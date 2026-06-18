// SPDX-License-Identifier: MIT
// Accuracy test: compare original kernel vs. cuBLAS replacement

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

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

// Load binary data
bool load_file(const char* path, void* buf, size_t size) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        return false;
    }
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
    
    printf("Test parameters: out=%d, B=%d\n", out, B);
    
    // Allocate host buffers
    float* h_z_init = new float[out * B];
    float* h_b = new float[out];
    float* h_expected = new float[out * B];
    
    // Load reference data
    if (!load_file("z_init.bin", h_z_init, out * B * sizeof(float))) {
        fprintf(stderr, "Failed to load z_init.bin\n");
        return 1;
    }
    if (!load_file("b.bin", h_b, out * sizeof(float))) {
        fprintf(stderr, "Failed to load b.bin\n");
        return 1;
    }
    if (!load_file("expected_z.bin", h_expected, out * B * sizeof(float))) {
        fprintf(stderr, "Failed to load expected_z.bin\n");
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
    CUDA_CHECK(cudaMemcpy(d_z_kernel, h_z_init, out * B * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z_cublas, h_z_init, out * B * sizeof(float), cudaMemcpyHostToDevice));
    
    // Fill d_ones_B with 1.0
    cudaStream_t stream = 0;
    float* h_ones_B = new float[B];
    for (int i = 0; i < B; i++) {
        h_ones_B[i] = 1.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_ones_B, h_ones_B, B * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_ones_B;
    
    // Run original kernel
    printf("Running original kernel...\n");
    add_bias_kernel(out, B, d_z_kernel, d_b, stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run cuBLAS replacement
    printf("Running cuBLAS replacement...\n");
    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    add_bias_cublas(handle, out, B, d_z_cublas, d_b, d_ones_B, stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUBLAS_CHECK(cublasDestroy(handle));
    
    // Copy results back
    float* h_z_kernel = new float[out * B];
    float* h_z_cublas = new float[out * B];
    CUDA_CHECK(cudaMemcpy(h_z_kernel, d_z_kernel, out * B * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_z_cublas, d_z_cublas, out * B * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compare kernel vs expected
    float max_err_kernel = 0.0f;
    int err_count_kernel = 0;
    for (int i = 0; i < out * B; i++) {
        float diff = std::abs(h_z_kernel[i] - h_expected[i]);
        float rel_err = diff / (std::abs(h_expected[i]) + 1e-8);
        if (rel_err > 1e-5 || diff > 1e-6) {
            err_count_kernel++;
            max_err_kernel = std::max(max_err_kernel, rel_err);
        }
    }
    
    // Compare cublas vs expected
    float max_err_cublas = 0.0f;
    int err_count_cublas = 0;
    for (int i = 0; i < out * B; i++) {
        float diff = std::abs(h_z_cublas[i] - h_expected[i]);
        float rel_err = diff / (std::abs(h_expected[i]) + 1e-8);
        if (rel_err > 1e-5 || diff > 1e-6) {
            err_count_cublas++;
            max_err_cublas = std::max(max_err_cublas, rel_err);
        }
    }
    
    printf("\n=== Accuracy Results ===\n");
    printf("Original kernel vs expected: %d errors, max rel_err=%.2e\n", 
           err_count_kernel, max_err_kernel);
    printf("cuBLAS replacement vs expected: %d errors, max rel_err=%.2e\n", 
           err_count_cublas, max_err_cublas);
    
    if (err_count_kernel == 0 && err_count_cublas == 0) {
        printf("PASS: Both implementations match expected result.\n");
    } else {
        printf("FAIL: Accuracy mismatch detected.\n");
        return 1;
    }
    
    // Cleanup
    delete[] h_z_init;
    delete[] h_b;
    delete[] h_expected;
    delete[] h_z_kernel;
    delete[] h_z_cublas;
    
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_z_kernel));
    CUDA_CHECK(cudaFree(d_z_cublas));
    CUDA_CHECK(cudaFree(d_ones_B));
    
    return 0;
}
