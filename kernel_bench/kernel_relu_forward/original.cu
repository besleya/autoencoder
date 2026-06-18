/**
 * Original kernel_relu_forward from /mnt/home/besleya/autoencoder/layer.cu (line 73)
 * 
 * Purpose: Forward pass for ReLU activation: a[idx] = max(z[idx], 0)
 */

#include <cuda_runtime.h>

/**
 * CUDA kernel for ReLU forward pass.
 * 
 * @param size    Number of elements
 * @param a       Output array (device memory)
 * @param z       Input array (device memory)
 */
__global__ void kernel_relu_forward(int size, float* a, const float* z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] = fmaxf(z[idx], 0.0f);
    }
}

/**
 * Host wrapper for original kernel (for testing/comparison).
 */
void relu_forward_original(int size, float* a, const float* z, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    kernel_relu_forward<<<grid, block, 0, stream>>>(size, a, z);
}
