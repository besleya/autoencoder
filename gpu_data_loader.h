// SPDX-License-Identifier: MIT
// gpu_data_loader.h — sparse minibatch DataLoader for GPU-resident autoencoders.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

// A single mini-batch: sparse CSC of shape (m × B) resident on device.
// Pointers are owned by the GpuDataLoader and remain valid until next_batch()
// or next_chunk() is called.
//
// Usage: the caller MUST use the returned stream, or synchronize between
// batches. E.g., the autoencoder kernel should be launched on out->stream,
// or the caller should synchronize the AE's stream with out->stream after
// consuming each batch. Simplest: run all AE operations on the loader's stream.
struct SparseBatch {
    int m;  // features (rows), same as dataset
    int B;  // batch size (cols)
    int nnz;  // nonzeros in this batch
    const int32_t* d_col_ptr;  // device, length B+1
    const int32_t* d_row_idx;  // device, length nnz
    const float* d_values;  // device, length nnz
    cudaStream_t stream;  // stream on which the batch buffers are valid
};

class GpuDataLoader {
public:
    // rng is held BY REFERENCE; the loader consumes RNG state for
    // (a) per-epoch shuffle of file order, (b) per-chunk shuffle of column
    // order.
    // chunk_size = number of .1pz files loaded into one device-resident chunk.
    // batch_size = number of columns per mini-batch.
    GpuDataLoader(const std::vector<std::string>& paths, int chunk_size, int batch_size,
                  std::mt19937& rng);
    ~GpuDataLoader();

    // Number of features (set after first chunk is loaded). 0 before
    // begin_epoch.
    int m() const;

    // Start a new epoch: shuffles file paths (only if multiple files), resets
    // chunk/batch cursors. RNG state is consumed here.
    void begin_epoch();

    // Returns true if a batch is available; false when the epoch is exhausted.
    // On true, fills *out with device pointers for the current mini-batch.
    // Loads next chunk on demand. Drops any tail batch within a chunk that
    // would be smaller than batch_size (matches main.cpp semantics).
    bool next_batch(SparseBatch* out);

private:
    std::vector<std::string> file_paths_;
    int chunk_size_;
    int batch_size_;
    std::mt19937& rng_;

    int m_;  // number of features
    int current_chunk_idx_;  // index into shuffled file paths
    int current_batch_idx_;  // batch index within current chunk

    // Current chunk state
    int chunk_n_;  // total columns in current chunk
    int chunk_nnz_;  // total nonzeros in current chunk
    std::vector<int> chunk_order_;  // permutation of chunk columns
    std::vector<int> host_col_ptr_;  // host mirror of chunk's col_ptr

    // Device buffers for chunk (persistent storage)
    int32_t* d_chunk_row_idx_;
    float* d_chunk_values_;
    size_t d_chunk_capacity_;

    // Device buffers for batch (reused, sized to max batch nnz in chunk)
    int32_t* d_batch_col_ptr_;   // length batch_size + 1
    int32_t* d_batch_row_idx_;   // length max_batch_nnz
    float* d_batch_values_;      // length max_batch_nnz
    size_t d_batch_capacity_;    // max nnz per batch

    // Pinned host buffers
    int32_t* h_batch_col_ptr_;  // pinned, length batch_size + 1

    cudaStream_t stream_;

    // Helper: load next chunk from file_paths_[current_chunk_idx_..],
    // update chunk state
    void load_next_chunk_();

    // Helper: determine actual batch nnz given columns to extract
    int compute_batch_nnz_(const std::vector<int>& cols);
};
