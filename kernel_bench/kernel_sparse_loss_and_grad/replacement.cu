// Replacement kernel_sparse_loss_and_grad using CUDA libraries (thrust + CUB)
// Strategy:
//   - Frobenius norm: CUB DeviceReduce::Sum over transform iterator for ||a_L||_F^2
//   - Gradient: thrust::for_each over sparse entries to compute (A_L - Y) / B at sparse positions
//   - Loss: CUB DeviceReduce::Sum over a transform iterator that yields v^2 - 2*A_L*v per entry

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>
#include <cub/cub.cuh>
#include <cstdint>
#include <cstdio>

// ============================================================================
// Device functor for Frobenius norm computation
// Computes: a_L[i]^2 for each element
// ============================================================================
struct FrobeniusNormFunctor {
    const float* a_L;
    
    __device__ float operator()(int idx) const {
        float a_val = a_L[idx];
        return a_val * a_val;
    }
};

// ============================================================================
// Device functor for loss computation
// Computes: v^2 - 2*A_L[r,j]*v for each sparse entry
// ============================================================================
struct LossTermFunctor {
    int d0;
    const float* a_L;
    const int32_t* row_idx;
    const float* values;
    const int32_t* col_ptr;
    int B;
    
    __device__ float operator()(int k) const {
        // Find which column this entry belongs to: entry k belongs to column j if col_ptr[j] <= k < col_ptr[j+1]
        int j = 0;
        for (int i = 0; i < B; ++i) {
            if (col_ptr[i] <= k && k < col_ptr[i + 1]) {
                j = i;
                break;
            }
        }
        
        int r = row_idx[k];
        float v = values[k];
        float a_val = a_L[r * B + j];
        
        return v * v - 2.0f * a_val * v;
    }
};

// ============================================================================
// Small custom kernel for gradient update (thrust doesn't provide gather-compute-scatter easily)
// ============================================================================
__global__ void kernel_gradient_update_library(
    int d0, int B, int nnz, float* d_grad_loss,
    const float* a_L, const int32_t* col_ptr, const int32_t* row_idx,
    const float* values, float inv_B) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    
    // Binary search to find column j: entry idx belongs to column j if col_ptr[j] <= idx < col_ptr[j+1]
    int j = 0;
    for (int i = 0; i < B; ++i) {
        if (col_ptr[i] <= idx && idx < col_ptr[i + 1]) {
            j = i;
            break;
        }
    }
    
    int r = row_idx[idx];
    float v = values[idx];
    float a_val = a_L[r * B + j];
    
    // d_grad_loss(r,j) = (a_L(r,j) - v) / B
    d_grad_loss[r * B + j] = (a_val - v) * inv_B;
}

// ============================================================================
// Replacement launcher using CUB reduce for loss and custom kernel for gradient
// ============================================================================
extern "C" void launch_kernel_sparse_loss_and_grad_replacement(
    int d0, int B, float* d_grad_loss, float* loss_acc,
    const float* a_L, const int32_t* col_ptr, const int32_t* row_idx,
    const float* values, cudaStream_t stream = 0) {
    
    // Get nnz from col_ptr
    int nnz;
    cudaMemcpyAsync(&nnz, col_ptr + B, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    if (nnz == 0) {
        return;
    }
    
    float inv_B = 1.0f / B;
    
    // ========================================================================
    // Step 1: Compute and add Frobenius norm ||a_L||_F^2 to loss_acc
    // ========================================================================
    {
        FrobeniusNormFunctor frob_functor{a_L};
        auto frob_iter = thrust::make_transform_iterator(
            thrust::counting_iterator<int>(0),
            frob_functor);
        
        // Allocate temp storage for CUB
        size_t temp_storage_bytes = 0;
        void* d_temp_storage = nullptr;
        float* d_frob_result = nullptr;
        
        cudaMalloc(&d_frob_result, sizeof(float));
        
        cub::DeviceReduce::Sum(
            d_temp_storage, temp_storage_bytes,
            frob_iter, d_frob_result, d0 * B, stream);
        
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        
        cub::DeviceReduce::Sum(
            d_temp_storage, temp_storage_bytes,
            frob_iter, d_frob_result, d0 * B, stream);
        
        // Add Frobenius norm to loss_acc
        // Create a kernel to do: loss_acc += d_frob_result
        // For simplicity, we'll copy it to host, add, and copy back
        float h_frob;
        cudaMemcpyAsync(&h_frob, d_frob_result, sizeof(float),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        float h_loss_acc = 0.0f;
        cudaMemcpyAsync(&h_loss_acc, loss_acc, sizeof(float),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        h_loss_acc += h_frob;
        
        cudaMemcpyAsync(loss_acc, &h_loss_acc, sizeof(float),
                       cudaMemcpyHostToDevice, stream);
        
        cudaFree(d_temp_storage);
        cudaFree(d_frob_result);
    }
    
    // ========================================================================
    // Step 2: Update gradient at sparse positions using custom kernel
    // ========================================================================
    {
        int threads = 256;
        int blocks = (nnz + threads - 1) / threads;
        kernel_gradient_update_library<<<blocks, threads, 0, stream>>>(
            d0, B, nnz, d_grad_loss, a_L, col_ptr, row_idx, values, inv_B);
    }
    
    // ========================================================================
    // Step 3: Compute and add sparse loss correction using CUB DeviceReduce
    // ========================================================================
    {
        LossTermFunctor loss_functor{d0, a_L, row_idx, values, col_ptr, B};
        auto loss_iter = thrust::make_transform_iterator(
            thrust::counting_iterator<int>(0),
            loss_functor);
        
        // Allocate temp storage for CUB
        size_t temp_storage_bytes = 0;
        void* d_temp_storage = nullptr;
        float* d_loss_result = nullptr;
        
        cudaMalloc(&d_loss_result, sizeof(float));
        
        cub::DeviceReduce::Sum(
            d_temp_storage, temp_storage_bytes,
            loss_iter, d_loss_result, nnz, stream);
        
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        
        cub::DeviceReduce::Sum(
            d_temp_storage, temp_storage_bytes,
            loss_iter, d_loss_result, nnz, stream);
        
        // Add sparse correction to loss_acc
        float h_sparse_loss;
        cudaMemcpyAsync(&h_sparse_loss, d_loss_result, sizeof(float),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        float h_loss_acc;
        cudaMemcpyAsync(&h_loss_acc, loss_acc, sizeof(float),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        h_loss_acc += h_sparse_loss;
        
        cudaMemcpyAsync(loss_acc, &h_loss_acc, sizeof(float),
                       cudaMemcpyHostToDevice, stream);
        
        cudaFree(d_temp_storage);
        cudaFree(d_loss_result);
    }
}
