// SPDX-License-Identifier: MIT
// Replacement using cuBLAS cublasSger: z += alpha * b * ones^T
// This implements: z[i,j] += b[i] for all i in [0, out), j in [0, B)

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>

// ============================================================================
// cuBLAS-based replacement
// ============================================================================

// Variant 1: Caller provides ones_vec (preferred for reuse)
void add_bias_cublas(cublasHandle_t handle, int out, int B, 
                      float* d_z, const float* d_b, const float* d_ones_B,
                      cudaStream_t stream) {
    // z += 1.0 * b * ones_B^T
    // cublasSger(handle, m, n, &alpha, x, incx, y, incy, A, lda)
    // x: vector of length m
    // y: vector of length n
    // A: matrix m x n (col-major)
    // A = alpha * x * y^T + A
    
    cublasSetStream(handle, stream);
    
    float one_f = 1.0f;
    cublasStatus_t stat = cublasSger(
        handle,
        out,          // m: leading dimension of z (and length of b)
        B,            // n: batch size (number of columns)
        &one_f,       // alpha
        (float*)d_b,  // x: bias vector of length out
        1,            // incx
        (float*)d_ones_B,  // y: ones vector of length B
        1,            // incy
        d_z,          // A: matrix out x B (col-major)
        out           // lda: leading dimension of z
    );
    
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSger failed: %d\n", stat);
        exit(EXIT_FAILURE);
    }
}

// Variant 2: Allocates and manages its own ones buffer (for standalone testing)
void add_bias_cublas_standalone(cublasHandle_t handle, int out, int B,
                                 float* d_z, const float* d_b,
                                 cudaStream_t stream) {
    static float* d_ones_cached = nullptr;
    static int cached_size = 0;
    
    // Allocate or reuse ones vector for this B
    if (d_ones_cached == nullptr || cached_size < B) {
        if (d_ones_cached != nullptr) {
            cudaFree(d_ones_cached);
        }
        cudaMalloc(&d_ones_cached, B * sizeof(float));
        
        // Fill with ones using host vector
        float* h_ones = new float[B];
        for (int i = 0; i < B; i++) {
            h_ones[i] = 1.0f;
        }
        cudaMemcpyAsync(d_ones_cached, h_ones, B * sizeof(float), 
                       cudaMemcpyHostToDevice, stream);
        delete[] h_ones;
        
        cached_size = B;
    }
    
    add_bias_cublas(handle, out, B, d_z, d_b, d_ones_cached, stream);
}
