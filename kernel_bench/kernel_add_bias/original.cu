// SPDX-License-Identifier: MIT
// Source: /mnt/home/besleya/autoencoder/layer.cu around line 62
// Bias broadcast (column-major): z[i,j] += b[i] for all j

#include <cuda_runtime.h>
#include <cstdio>

// ============================================================================
// Original kernel: add_bias (kernel-based)
// ============================================================================

__global__ void kernel_add_bias(int out, int B, float* z, const float* b) {
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    if (bidx >= B) return;

    for (int i = tidx; i < out; i += blockDim.x) {
        z[i + (size_t)bidx * out] += b[i];
    }
}

// Launcher for the original kernel
void add_bias_kernel(int out, int B, float* d_z, const float* d_b, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(B);
    kernel_add_bias<<<grid, block, 0, stream>>>(out, B, d_z, d_b);
}
