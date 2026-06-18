// SPDX-License-Identifier: MIT
// original.cu — Original log_normalize_columns_kernel
// Source: gpu_data_loader.cu:41–65 (commit reference: gpu_data_loader.cu)
//
// Kernel: For each column j of a sparse CSC matrix:
//   1. Compute sum of all values in column j
//   2. If sum > 0, replace each value v with log1pf(scaler * v / sum)

__global__ void log_normalize_columns_kernel(int n_cols,
                                             float scaler,
                                             const int32_t* __restrict__ col_ptr,
                                             float* __restrict__ values) {
    int col = blockIdx.x;
    if (col >= n_cols) return;

    int start = col_ptr[col];
    int end   = col_ptr[col + 1];

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();

    // Phase 1: column sum via block-shared atomicAdd
    float thread_sum = 0.0f;
    for (int k = start + threadIdx.x; k < end; k += blockDim.x) {
        thread_sum += values[k];
    }
    if (thread_sum != 0.0f) atomicAdd(&s_sum, thread_sum);
    __syncthreads();

    float sum = s_sum;
    if (sum > 0.0f) {
        float inv = scaler / sum;
        for (int k = start + threadIdx.x; k < end; k += blockDim.x) {
            values[k] = log1pf(values[k] * inv);
        }
    }
}

// Launcher wrapper (from gpu_data_loader.cu:332, adapted for standalone use)
void log_normalize_columns_original(int n_cols,
                                   float scaler,
                                   const int32_t* col_ptr,
                                   float* values,
                                   cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(n_cols);
    log_normalize_columns_kernel<<<grid, block, 0, stream>>>(
        n_cols, scaler, col_ptr, values);
}
