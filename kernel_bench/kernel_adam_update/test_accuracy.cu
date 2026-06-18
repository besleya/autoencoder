// Accuracy verification test for both original and replacement implementations
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

// Forward declarations (defined in original.cu and replacement.cu)
void adam_update_original(int size, float* p, float* m, float* v, const float* g,
                          float lr_t, float beta1, float beta2, float eps,
                          cudaStream_t stream);

void adam_update_thrust(int size, float* p, float* m, float* v, const float* g,
                        float lr_t, float beta1, float beta2, float eps,
                        cudaStream_t stream);

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Tolerance levels for comparison
// ============================================================================
const float REL_TOL = 1e-5f;
const float ABS_TOL = 1e-6f;

// Compare two arrays with relative and absolute tolerance
bool compare_arrays(const float* computed, const float* expected, int size,
                    const char* name) {
    float max_rel_err = 0.0f;
    float max_abs_err = 0.0f;
    int error_count = 0;
    
    for (int i = 0; i < size; ++i) {
        float abs_err = std::abs(computed[i] - expected[i]);
        float rel_err = (expected[i] != 0.0f) ? abs_err / std::abs(expected[i]) : abs_err;
        
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
        
        if (abs_err > ABS_TOL && rel_err > REL_TOL) {
            error_count++;
            if (error_count <= 5) {  // Print first 5 mismatches
                printf("  %s[%d]: computed=%.6e, expected=%.6e, abs_err=%.6e, rel_err=%.6e\n",
                       name, i, computed[i], expected[i], abs_err, rel_err);
            }
        }
    }
    
    bool pass = (error_count == 0);
    printf("%s: %s (max_abs_err=%.2e, max_rel_err=%.2e, errors=%d/%d)\n",
           name, pass ? "PASS" : "FAIL", max_abs_err, max_rel_err, error_count, size);
    
    return pass;
}

// Test a single implementation
bool test_implementation(const char* impl_name,
                        void (*update_func)(int, float*, float*, float*, const float*,
                                          float, float, float, float, cudaStream_t),
                        int size,
                        const float* p_init, const float* m_init, const float* v_init,
                        const float* g,
                        const float* expected_p, const float* expected_m, const float* expected_v,
                        float lr_t, float beta1, float beta2, float eps) {
    printf("\n=== Testing %s ===\n", impl_name);
    
    // Allocate device memory
    float *d_p, *d_m, *d_v, *d_g;
    CUDA_CHECK(cudaMalloc(&d_p, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g, size * sizeof(float)));
    
    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_p, p_init, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m, m_init, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, v_init, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g, g, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Run the implementation
    CUDA_CHECK(cudaDeviceSynchronize());
    update_func(size, d_p, d_m, d_v, d_g, lr_t, beta1, beta2, eps, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    float* h_p = new float[size];
    float* h_m = new float[size];
    float* h_v = new float[size];
    
    CUDA_CHECK(cudaMemcpy(h_p, d_p, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_m, d_m, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v, d_v, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compare with expected values
    bool pass_p = compare_arrays(h_p, expected_p, size, "p");
    bool pass_m = compare_arrays(h_m, expected_m, size, "m");
    bool pass_v = compare_arrays(h_v, expected_v, size, "v");
    
    bool all_pass = pass_p && pass_m && pass_v;
    
    // Cleanup
    delete[] h_p;
    delete[] h_m;
    delete[] h_v;
    
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_g));
    
    return all_pass;
}

// ============================================================================
// Main test
// ============================================================================
int main() {
    printf("Adam Update Accuracy Test\n");
    printf("==========================\n");
    
    // Read metadata
    FILE* meta_file = fopen("meta.txt", "r");
    if (!meta_file) {
        fprintf(stderr, "Failed to open meta.txt\n");
        return 1;
    }
    
    int size;
    float lr_t, beta1, beta2, eps;
    fscanf(meta_file, "size=%d\n", &size);
    fscanf(meta_file, "lr_t=%f\n", &lr_t);
    fscanf(meta_file, "beta1=%f\n", &beta1);
    fscanf(meta_file, "beta2=%f\n", &beta2);
    fscanf(meta_file, "eps=%f\n", &eps);
    fclose(meta_file);
    
    printf("Parameters: size=%d, lr_t=%.2e, beta1=%.2f, beta2=%.4f, eps=%.2e\n",
           size, lr_t, beta1, beta2, eps);
    printf("Tolerances: rel=%.2e, abs=%.2e\n", REL_TOL, ABS_TOL);
    
    // Allocate host arrays
    float* p_init = new float[size];
    float* m_init = new float[size];
    float* v_init = new float[size];
    float* g = new float[size];
    float* expected_p = new float[size];
    float* expected_m = new float[size];
    float* expected_v = new float[size];
    
    // Read binary files
    auto read_file = [](const char* filename, float* data, int size) {
        FILE* f = fopen(filename, "rb");
        if (!f) {
            fprintf(stderr, "Failed to open %s\n", filename);
            exit(1);
        }
        size_t nread = fread(data, sizeof(float), size, f);
        if (nread != (size_t)size) {
            fprintf(stderr, "Error reading %s: expected %d floats, got %zu\n", filename, size, nread);
            exit(1);
        }
        fclose(f);
    };
    
    read_file("p_init.bin", p_init, size);
    read_file("m_init.bin", m_init, size);
    read_file("v_init.bin", v_init, size);
    read_file("g.bin", g, size);
    read_file("expected_p.bin", expected_p, size);
    read_file("expected_m.bin", expected_m, size);
    read_file("expected_v.bin", expected_v, size);
    
    // Test both implementations
    bool pass_original = test_implementation(
        "Original Kernel", adam_update_original,
        size, p_init, m_init, v_init, g,
        expected_p, expected_m, expected_v,
        lr_t, beta1, beta2, eps);
    
    bool pass_thrust = test_implementation(
        "Thrust Replacement", adam_update_thrust,
        size, p_init, m_init, v_init, g,
        expected_p, expected_m, expected_v,
        lr_t, beta1, beta2, eps);
    
    // Summary
    printf("\n=== Summary ===\n");
    printf("Original: %s\n", pass_original ? "PASS" : "FAIL");
    printf("Thrust:   %s\n", pass_thrust ? "PASS" : "FAIL");
    
    // Cleanup
    delete[] p_init;
    delete[] m_init;
    delete[] v_init;
    delete[] g;
    delete[] expected_p;
    delete[] expected_m;
    delete[] expected_v;
    
    return (pass_original && pass_thrust) ? 0 : 1;
}
