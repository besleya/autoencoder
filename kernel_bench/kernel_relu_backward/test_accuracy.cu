// Test accuracy of ReLU backward implementations
// Compares original and thrust-based implementations against reference data

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Forward declarations of implementations
void relu_backward_original(int size, float* dz, const float* z, cudaStream_t stream);
void relu_backward_thrust(int size, float* dz, const float* z, cudaStream_t stream);

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

bool load_binary_file(const char* filename, float* buffer, int size) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return false;
    }
    size_t read = fread(buffer, sizeof(float), size, f);
    fclose(f);
    if (read != (size_t)size) {
        fprintf(stderr, "Error: Expected %d floats, got %zu in %s\n", size, read, filename);
        return false;
    }
    return true;
}

int get_size_from_meta() {
    FILE* f = fopen("meta.txt", "r");
    if (!f) {
        fprintf(stderr, "Error: Cannot open meta.txt\n");
        return -1;
    }
    int size = -1;
    fscanf(f, "size=%d", &size);
    fclose(f);
    return size;
}

bool compare_buffers(const float* expected, const float* actual, int size, 
                     float rel_tol, float abs_tol, const char* impl_name) {
    int errors = 0;
    float max_rel_err = 0.0f;
    float max_abs_err = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float abs_err = fabsf(expected[i] - actual[i]);
        float rel_err = 0.0f;
        if (fabsf(expected[i]) > abs_tol) {
            rel_err = abs_err / fabsf(expected[i]);
        }
        
        bool is_error = (abs_err > abs_tol && rel_err > rel_tol);
        if (is_error) {
            errors++;
            if (errors <= 5) {
                printf("  [%s] Error at idx %d: expected=%e, actual=%e, abs_err=%e, rel_err=%e\n",
                       impl_name, i, expected[i], actual[i], abs_err, rel_err);
            }
        }
        max_abs_err = fmaxf(max_abs_err, abs_err);
        max_rel_err = fmaxf(max_rel_err, rel_err);
    }
    
    printf("[%s] Max absolute error: %e\n", impl_name, max_abs_err);
    printf("[%s] Max relative error: %e\n", impl_name, max_rel_err);
    printf("[%s] Total errors: %d / %d\n", impl_name, errors, size);
    
    return errors == 0;
}

int main() {
    // Load metadata
    int size = get_size_from_meta();
    if (size <= 0) {
        fprintf(stderr, "Error: Invalid size from meta.txt\n");
        return 1;
    }
    printf("Test size: %d elements\n\n", size);
    
    // Allocate host buffers
    float* h_dz_init = (float*)malloc(size * sizeof(float));
    float* h_z = (float*)malloc(size * sizeof(float));
    float* h_expected = (float*)malloc(size * sizeof(float));
    float* h_result = (float*)malloc(size * sizeof(float));
    
    if (!h_dz_init || !h_z || !h_expected || !h_result) {
        fprintf(stderr, "Error: Host memory allocation failed\n");
        return 1;
    }
    
    // Load test data
    printf("Loading test data...\n");
    if (!load_binary_file("dz_init.bin", h_dz_init, size)) return 1;
    if (!load_binary_file("z.bin", h_z, size)) return 1;
    if (!load_binary_file("expected_dz.bin", h_expected, size)) return 1;
    printf("Test data loaded.\n\n");
    
    // Allocate device buffers
    float* d_dz = nullptr;
    float* d_z = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dz, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z, size * sizeof(float)));
    
    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    const float rel_tol = 1e-5f;
    const float abs_tol = 1e-6f;
    
    // Test original implementation
    printf("=== Testing Original Implementation ===\n");
    CUDA_CHECK(cudaMemcpyAsync(d_dz, h_dz_init, size * sizeof(float), 
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_z, h_z, size * sizeof(float), 
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    relu_backward_original(size, d_dz, d_z, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    CUDA_CHECK(cudaMemcpyAsync(h_result, d_dz, size * sizeof(float), 
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    bool orig_pass = compare_buffers(h_expected, h_result, size, rel_tol, abs_tol, "original");
    printf("Original: %s\n\n", orig_pass ? "PASS" : "FAIL");
    
    // Test thrust implementation
    printf("=== Testing Thrust Implementation ===\n");
    CUDA_CHECK(cudaMemcpyAsync(d_dz, h_dz_init, size * sizeof(float), 
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_z, h_z, size * sizeof(float), 
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    relu_backward_thrust(size, d_dz, d_z, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    CUDA_CHECK(cudaMemcpyAsync(h_result, d_dz, size * sizeof(float), 
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    bool thrust_pass = compare_buffers(h_expected, h_result, size, rel_tol, abs_tol, "thrust");
    printf("Thrust: %s\n\n", thrust_pass ? "PASS" : "FAIL");
    
    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_dz));
    CUDA_CHECK(cudaFree(d_z));
    free(h_dz_init);
    free(h_z);
    free(h_expected);
    free(h_result);
    
    printf("=== Summary ===\n");
    printf("Original: %s\n", orig_pass ? "PASS" : "FAIL");
    printf("Thrust: %s\n", thrust_pass ? "PASS" : "FAIL");
    
    return (orig_pass && thrust_pass) ? 0 : 1;
}
