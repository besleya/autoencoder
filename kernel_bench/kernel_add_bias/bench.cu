// SPDX-License-Identifier: MIT
// Benchmark: timing comparison of original kernel vs. cuBLAS replacement

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

extern void add_bias_kernel(int out, int B, float* d_z, const float* d_b, cudaStream_t stream);
extern void add_bias_cublas(cublasHandle_t handle, int out, int B,
                            float* d_z, const float* d_b, const float* d_ones_B,
                            cudaStream_t stream);

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t err = (call); \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error: %d\n", err); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

bool load_file(const char* path, void* buf, size_t size) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    size_t read = fread(buf, 1, size, f);
    fclose(f);
    return read == size;
}

int main() {
    // Read metadata
    int out = 0, B = 0;
    FILE* meta_f = fopen("meta.txt", "r");
    if (!meta_f) {
        fprintf(stderr, "Cannot open meta.txt\n");
        return 1;
    }
    fscanf(meta_f, "out=%d\nB=%d\n", &out, &B);
    fclose(meta_f);
    
    printf("Benchmark parameters: out=%d, B=%d\n", out, B);
    
    // Allocate host buffers
    float* h_z_init = new float[out * B];
    float* h_b = new float[out];
    
    if (!load_file("z_init.bin", h_z_init, out * B * sizeof(float))) {
        fprintf(stderr, "Failed to load z_init.bin\n");
        return 1;
    }
    if (!load_file("b.bin", h_b, out * sizeof(float))) {
        fprintf(stderr, "Failed to load b.bin\n");
        return 1;
    }
    
    // Allocate device buffers
    float* d_b = nullptr;
    float* d_z_kernel = nullptr;
    float* d_z_cublas = nullptr;
    float* d_ones_B = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_b, out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z_kernel, out * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z_cublas, out * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ones_B, B * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_b, h_b, out * sizeof(float), cudaMemcpyHostToDevice));
    
    // Fill d_ones_B with 1.0
    cudaStream_t stream = 0;
    float* h_ones_B = new float[B];
    for (int i = 0; i < B; i++) {
        h_ones_B[i] = 1.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_ones_B, h_ones_B, B * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_ones_B;
    
    // Setup cuBLAS
    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    
    // Warm up
    const int warmup_iters = 10;
    const int timed_iters = 100;
    
    printf("Warmup phase (%d iterations)...\n", warmup_iters);
    for (int iter = 0; iter < warmup_iters; iter++) {
        CUDA_CHECK(cudaMemcpy(d_z_kernel, h_z_init, out * B * sizeof(float), cudaMemcpyHostToDevice));
        add_bias_kernel(out, B, d_z_kernel, d_b, stream);
        
        CUDA_CHECK(cudaMemcpy(d_z_cublas, h_z_init, out * B * sizeof(float), cudaMemcpyHostToDevice));
        add_bias_cublas(handle, out, B, d_z_cublas, d_b, d_ones_B, stream);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark original kernel
    printf("Timing original kernel (%d iterations)...\n", timed_iters);
    std::vector<float> times_kernel;
    for (int iter = 0; iter < timed_iters; iter++) {
        CUDA_CHECK(cudaMemcpy(d_z_kernel, h_z_init, out * B * sizeof(float), cudaMemcpyHostToDevice));
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start, stream));
        add_bias_kernel(out, B, d_z_kernel, d_b, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_kernel.push_back(ms);
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // Benchmark cuBLAS replacement
    printf("Timing cuBLAS replacement (%d iterations)...\n", timed_iters);
    std::vector<float> times_cublas;
    for (int iter = 0; iter < timed_iters; iter++) {
        CUDA_CHECK(cudaMemcpy(d_z_cublas, h_z_init, out * B * sizeof(float), cudaMemcpyHostToDevice));
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start, stream));
        add_bias_cublas(handle, out, B, d_z_cublas, d_b, d_ones_B, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_cublas.push_back(ms);
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // Compute medians
    std::sort(times_kernel.begin(), times_kernel.end());
    std::sort(times_cublas.begin(), times_cublas.end());
    
    float median_kernel = times_kernel[timed_iters / 2];
    float median_cublas = times_cublas[timed_iters / 2];
    float speedup = median_kernel / median_cublas;
    
    printf("\n=== Benchmark Results ===\n");
    printf("Original kernel median time: %.4f ms\n", median_kernel);
    printf("cuBLAS replacement median time: %.4f ms\n", median_cublas);
    printf("Speedup: %.2fx\n", speedup);
    
    // Cleanup
    delete[] h_z_init;
    delete[] h_b;
    
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_z_kernel));
    CUDA_CHECK(cudaFree(d_z_cublas));
    CUDA_CHECK(cudaFree(d_ones_B));
    CUBLAS_CHECK(cublasDestroy(handle));
    
    return 0;
}
