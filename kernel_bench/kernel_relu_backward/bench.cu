// Benchmark ReLU backward implementations
// 10 warmup iterations + 100 timed iterations
// Per-iteration: reset dz from dz_init, run kernel, measure time

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

// Forward declarations
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

int main() {
    // Load metadata
    int size = get_size_from_meta();
    if (size <= 0) {
        fprintf(stderr, "Error: Invalid size from meta.txt\n");
        return 1;
    }
    printf("Benchmark size: %d elements\n\n", size);
    
    // Allocate host buffers
    float* h_dz_init = (float*)malloc(size * sizeof(float));
    float* h_z = (float*)malloc(size * sizeof(float));
    
    if (!h_dz_init || !h_z) {
        fprintf(stderr, "Error: Host memory allocation failed\n");
        return 1;
    }
    
    // Load test data
    printf("Loading test data...\n");
    if (!load_binary_file("dz_init.bin", h_dz_init, size)) return 1;
    if (!load_binary_file("z.bin", h_z, size)) return 1;
    printf("Test data loaded.\n\n");
    
    // Allocate device buffers
    float* d_dz = nullptr;
    float* d_z = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dz, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z, size * sizeof(float)));
    
    // Copy z to device (constant)
    CUDA_CHECK(cudaMemcpy(d_z, h_z, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int WARMUP = 10;
    const int ITERATIONS = 100;
    
    std::vector<float> times_original;
    std::vector<float> times_thrust;
    
    // Benchmark original implementation
    printf("=== Benchmarking Original Implementation ===\n");
    printf("Warmup: %d iterations\n", WARMUP);
    for (int i = 0; i < WARMUP; i++) {
        CUDA_CHECK(cudaMemcpyAsync(d_dz, h_dz_init, size * sizeof(float), 
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        relu_backward_original(size, d_dz, d_z, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    printf("Timed iterations: %d\n", ITERATIONS);
    for (int i = 0; i < ITERATIONS; i++) {
        CUDA_CHECK(cudaMemcpyAsync(d_dz, h_dz_init, size * sizeof(float), 
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        CUDA_CHECK(cudaEventRecord(start, stream));
        relu_backward_original(size, d_dz, d_z, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_original.push_back(ms);
    }
    
    // Benchmark thrust implementation
    printf("\n=== Benchmarking Thrust Implementation ===\n");
    printf("Warmup: %d iterations\n", WARMUP);
    for (int i = 0; i < WARMUP; i++) {
        CUDA_CHECK(cudaMemcpyAsync(d_dz, h_dz_init, size * sizeof(float), 
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        relu_backward_thrust(size, d_dz, d_z, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    printf("Timed iterations: %d\n", ITERATIONS);
    for (int i = 0; i < ITERATIONS; i++) {
        CUDA_CHECK(cudaMemcpyAsync(d_dz, h_dz_init, size * sizeof(float), 
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        CUDA_CHECK(cudaEventRecord(start, stream));
        relu_backward_thrust(size, d_dz, d_z, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_thrust.push_back(ms);
    }
    
    // Compute statistics
    std::sort(times_original.begin(), times_original.end());
    std::sort(times_thrust.begin(), times_thrust.end());
    
    float median_orig = times_original[ITERATIONS / 2];
    float median_thrust = times_thrust[ITERATIONS / 2];
    
    float avg_orig = 0.0f;
    for (auto t : times_original) avg_orig += t;
    avg_orig /= ITERATIONS;
    
    float avg_thrust = 0.0f;
    for (auto t : times_thrust) avg_thrust += t;
    avg_thrust /= ITERATIONS;
    
    float speedup = median_orig / median_thrust;
    
    printf("\n=== Results ===\n");
    printf("Original - Median: %.6f ms, Average: %.6f ms\n", median_orig, avg_orig);
    printf("Thrust   - Median: %.6f ms, Average: %.6f ms\n", median_thrust, avg_thrust);
    printf("Speedup (median original / median thrust): %.2f x\n", speedup);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_dz));
    CUDA_CHECK(cudaFree(d_z));
    free(h_dz_init);
    free(h_z);
    
    return 0;
}
