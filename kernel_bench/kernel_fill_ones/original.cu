/* 
   ORIGINAL KERNEL
   Source: /mnt/home/besleya/autoencoder/layer.cu, lines ~89, ~330
   Kernel definition line 89, launch in _ensure_batch_buffers line 330
*/

#include <cuda_runtime.h>

// Fill vector with ones
__global__ void kernel_fill_ones(int n, float* v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        v[idx] = 1.0f;
    }
}

/* Launcher helper (from layer.cu context) */
void launch_kernel_fill_ones(int batch_size, float* d_ones) {
    // Simplified launcher; assumes extern get_grid_block is available
    // In original context, called as:
    //   dim3 grid, block;
    //   get_grid_block(batch_size, grid, block);
    //   kernel_fill_ones<<<grid, block>>>(batch_size, d_ones);
    
    // For standalone compilation, use simple 1D grid
    int block_size = 256;
    int grid_size = (batch_size + block_size - 1) / block_size;
    kernel_fill_ones<<<grid_size, block_size>>>(batch_size, d_ones);
}
