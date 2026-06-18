/*
   BENCHMARK
   10 warmup + 100 timed iterations. Report median time and speedup.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

// Declare implementations
void fill_ones_thrust(int n, float* v, cudaStream_t stream);
void launch_kernel_fill_ones(int batch_size, float* d_ones);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

void launch_kernel_fill_ones_original(int n, float* d_v, cudaStream_t stream) {
    launch_kernel_fill_ones(n, d_v);
}

int main(int argc, char** argv) {
    int n = 4096;
    int warmup_iters = 10;
    int timed_iters = 100;
    
    printf("Benchmark: kernel_fill_ones\n");
    printf("n = %d, warmup = %d, timed = %d\n", n, warmup_iters, timed_iters);
    
    // Allocate device buffer
    float* d_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_v, n * sizeof(float)));
    
    // Create streams
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // ========== Benchmark Original Kernel ==========
    printf("\n--- Original Kernel (kernel_fill_ones) ---\n");
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        launch_kernel_fill_ones_original(n, d_v, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Timed
    std::vector<float> times_orig(timed_iters);
    for (int i = 0; i < timed_iters; i++) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start, stream));
        launch_kernel_fill_ones_original(n, d_v, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_orig[i] = ms;
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    std::sort(times_orig.begin(), times_orig.end());
    float median_orig = times_orig[timed_iters / 2];
    printf("Median time: %.6f ms\n", median_orig);
    
    // ========== Benchmark Thrust Replacement ==========
    printf("\n--- Thrust Replacement (fill_ones_thrust) ---\n");
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        fill_ones_thrust(n, d_v, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Timed
    std::vector<float> times_thrust(timed_iters);
    for (int i = 0; i < timed_iters; i++) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start, stream));
        fill_ones_thrust(n, d_v, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_thrust[i] = ms;
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    std::sort(times_thrust.begin(), times_thrust.end());
    float median_thrust = times_thrust[timed_iters / 2];
    printf("Median time: %.6f ms\n", median_thrust);
    
    // ========== Summary ==========
    float speedup = median_orig / median_thrust;
    printf("\n--- Summary ---\n");
    printf("Original median:      %.6f ms\n", median_orig);
    printf("Thrust median:        %.6f ms\n", median_thrust);
    printf("Speedup (orig/thrust): %.2fx\n", speedup);
    
    if (speedup > 1.0f) {
        printf("Result: Thrust is FASTER\n");
    } else {
        printf("Result: Original kernel is faster\n");
    }
    
    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_v));
    
    return 0;
}
