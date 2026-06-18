// Original kernel_sparse_loss_and_grad + launcher
// Source: gpu_autoencoder.cu:59-79 (kernel definition), gpu_autoencoder.cu:334-343 (launcher)

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cub/cub.cuh>

// Helper kernel: Compute Frobenius norm squared of a_L and add to loss_acc
__global__ void kernel_add_frobenius_norm(
    int d0, int B, float* loss_acc, const float* a_L) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)d0 * B) return;
    
    float a_val = a_L[idx];
    atomicAdd(loss_acc, a_val * a_val);
}

// Compute loss and initialize loss gradient:
//   loss_acc = ||a_L||_F^2
//   d_grad_loss = a_L / B (for all entries)
//   For each sparse (r, j, v): d_grad_loss(r,j) = (a_L(r,j) - v) / B
//                              loss_acc += v^2 - 2*a_L(r,j)*v
// Then loss = loss_acc / (d0 * B)
//
// This kernel iterates through columns; each thread block handles one column.
__global__ void kernel_sparse_loss_and_grad(
    int d0, int B, float* d_grad_loss, float* loss_acc, 
    const float* a_L, const int32_t* col_ptr, const int32_t* row_idx,
    const float* values) {
    
    int j = blockIdx.x;
    if (j >= B) return;
    
    int start = col_ptr[j];
    int end = col_ptr[j + 1];
    
    for (int k = start + threadIdx.x; k < end; k += blockDim.x) {
        int r = row_idx[k];
        float v = values[k];
        float a_val = a_L[r * B + j];
        
        // d_grad_loss(r,j) = (a_L(r,j) - v) / B
        d_grad_loss[r * B + j] = (a_val - v) / B;
        
        // Accumulate loss correction
        float correction = v * v - 2.0f * a_val * v;
        atomicAdd(loss_acc, correction);
    }
}

// Launcher wrapper for testing
extern "C" void launch_kernel_sparse_loss_and_grad(
    int d0, int B, float* d_grad_loss, float* loss_acc,
    const float* a_L, const int32_t* col_ptr, const int32_t* row_idx,
    const float* values, cudaStream_t stream = 0) {
    
    // Step 1: Add Frobenius norm of a_L to loss_acc
    {
        int threads = 256;
        int blocks = (d0 * B + threads - 1) / threads;
        kernel_add_frobenius_norm<<<blocks, threads, 0, stream>>>(
            d0, B, loss_acc, a_L);
    }
    
    // Step 2: Process sparse entries and add corrections to loss
    kernel_sparse_loss_and_grad<<<B, 256, 0, stream>>>(
        d0, B, d_grad_loss, loss_acc, a_L, col_ptr, row_idx, values);
    
    // Step 3: Divide loss_acc by (d0 * B)
    // This is done in a small kernel since we need to modify a single value
    // For simplicity, we'll do it in a small 1x1 grid kernel
    // Actually, let's do this as a separate post-processing step in the test code
}
