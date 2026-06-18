// SOURCE: /mnt/home/besleya/autoencoder/layer.cu
// ReLU backward kernel (original implementation)
//
// Kernel definition: lines 79-86
// Launcher: lines 641-647 in _apply_activation_backward

#include <cuda_runtime.h>

// ReLU backward: dz[i] *= (z[i] > 0)
__global__ void kernel_relu_backward(int size, float* dz, const float* z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dz[idx] *= (z[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// Launcher wrapper for original kernel
void relu_backward_original(int size, float* dz, const float* z, cudaStream_t stream) {
    // Simplified grid/block configuration
    // In original code: get_grid_block(size, grid, block) is called
    // Using standard: 1 thread per element, max 1024 threads per block
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    dim3 grid(num_blocks), block(block_size);
    
    kernel_relu_backward<<<grid, block, 0, stream>>>(size, dz, z);
}
