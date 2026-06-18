// SPDX-License-Identifier: MIT
// replacement.cu — CUB + Thrust implementation of log_normalize_columns_kernel
//
// Strategy: Use CUB::DeviceSegmentedReduce::Sum to compute per-column sums,
// then thrust::transform with a column-index lookup functor to apply log normalization.
// Zero hand-written kernels.

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <cuda_runtime.h>
#include <cassert>

// Functor: Given an nnz index, find its column (via upper_bound on col_ptr),
// then apply log1pf(scaler * v / col_sums[col]) if col_sums[col] > 0
struct LogNormalizeTransform {
    float scaler;
    const int32_t* col_ptr;
    const float* col_sums;
    float* values;
    int n_cols;

    // Binary search: upper_bound on col_ptr to find column
    __device__ int find_column(int nnz_idx) const {
        // col_ptr[col] <= nnz_idx < col_ptr[col+1]
        // Find col such that col_ptr[col] <= nnz_idx < col_ptr[col+1]
        // Equivalently: find col = largest j where col_ptr[j] <= nnz_idx
        int lo = 0, hi = n_cols;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (col_ptr[mid] <= nnz_idx) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo - 1;  // Column index
    }

    __device__ void operator()(int nnz_idx) const {
        int col = find_column(nnz_idx);
        float s = col_sums[col];
        if (s > 0.0f) {
            float v = values[nnz_idx];
            values[nnz_idx] = log1pf(scaler * v / s);
        }
    }
};

// Main library function
void log_normalize_columns_lib(int n_cols,
                              float scaler,
                              const int32_t* col_ptr,
                              float* values,
                              cudaStream_t stream) {
    // Total nnz
    int nnz;
    cudaMemcpyAsync(&nnz, col_ptr + n_cols, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Allocate temp buffer for CUB
    void* d_temp = nullptr;
    size_t temp_bytes = 0;

    // Allocate d_col_sums (length n_cols, type float)
    float* d_col_sums;
    cudaMalloc(&d_col_sums, n_cols * sizeof(float));

    // Step 1: Compute per-column sums using CUB::DeviceSegmentedReduce::Sum
    // Signature: Sum(void* d_temp_storage, size_t& temp_storage_bytes,
    //               InputIteratorT d_in, OutputIteratorT d_out,
    //               int num_segments, OffsetIteratorT d_begin_offsets,
    //               OffsetIteratorT d_end_offsets, cudaStream_t stream)
    //
    // We use col_ptr as begin_offsets and (col_ptr + 1) as end_offsets.

    // Query workspace size
    cub::DeviceSegmentedReduce::Sum(d_temp, temp_bytes,
                                    values, d_col_sums, n_cols,
                                    col_ptr, col_ptr + 1, stream);

    // Allocate workspace
    cudaMalloc(&d_temp, temp_bytes);

    // Run reduction
    cub::DeviceSegmentedReduce::Sum(d_temp, temp_bytes,
                                    values, d_col_sums, n_cols,
                                    col_ptr, col_ptr + 1, stream);

    // Step 2: Transform values in-place
    // Apply LogNormalizeTransform functor to each nnz index
    LogNormalizeTransform xform{scaler, col_ptr, d_col_sums, values, n_cols};

    thrust::for_each(thrust::cuda::par.on(stream),
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(nnz),
                    xform);

    // Cleanup
    cudaFree(d_temp);
    cudaFree(d_col_sums);
}
