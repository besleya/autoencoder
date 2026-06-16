// SPDX-License-Identifier: MIT
// gpu_data_loader.h — sparse minibatch DataLoader with async chunk prefetch.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

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
    // --- Configuration ---
    std::vector<std::string> file_paths_;
    int chunk_size_;
    int batch_size_;
    std::mt19937& rng_;
    int m_;  // number of features

    // --- ChunkSlot: double-buffered device storage ---
    struct ChunkSlot {
        int32_t* d_col_ptr  = nullptr;  // device, capacity cap_n+1
        int32_t* d_row_idx  = nullptr;  // device, capacity cap_nnz
        float*   d_values   = nullptr;  // device, capacity cap_nnz
        std::vector<int32_t> host_col_ptr;  // host mirror of col_ptr
        std::vector<int>     order;         // shuffle permutation (host)
        int     n            = 0;       // actual columns in this chunk
        int     nnz          = 0;       // actual nonzeros in this chunk
        size_t  cap_n        = 0;       // device buffer capacity (cols)
        size_t  cap_nnz      = 0;       // device buffer capacity (nnz)
        cudaEvent_t ready_evt = nullptr; // signaled after H2D + lognorm
        int     batches_remaining = 0;  // how many batches left in this slot
        int     current_batch_idx = 0;  // current batch index within slot
        bool    valid       = false;    // contents are loaded
        bool    eof         = false;    // no more data after this slot
    };
    ChunkSlot slots_[2];
    int       active_slot_ = 0;  // slot currently being consumed

    // --- Two CUDA streams ---
    cudaStream_t stream_;         // consumer stream (returned in SparseBatch)
    cudaStream_t loader_stream_;  // loader stream (used by prefetch worker)

    // --- Device buffers for batch (reused, sized to max batch nnz in chunk) ---
    int32_t* d_batch_col_ptr_;   // length batch_size + 1
    int32_t* d_batch_row_idx_;   // length max_batch_nnz
    float* d_batch_values_;      // length max_batch_nnz
    size_t d_batch_capacity_;    // max nnz per batch

    // Pinned host buffers
    int32_t* h_batch_col_ptr_;  // pinned, length batch_size + 1

    // --- Prefetch worker thread state ---
    struct LoadRequest {
        int slot_idx;
        std::vector<std::string> files;
    };
    std::thread worker_;
    std::mutex  mtx_;
    std::condition_variable cv_;
    std::queue<LoadRequest> work_queue_;
    std::atomic<bool> stop_{false};
    std::vector<std::string> file_order_for_epoch_;  // populated by begin_epoch
    size_t next_chunk_start_idx_ = 0;
    bool   epoch_eof_ = false;

    // --- Helper: load a chunk into the given slot (runs on worker thread) ---
    void load_chunk_into_slot_(const LoadRequest& req);

    // --- Helper: determine actual batch nnz given columns to extract ---
    int compute_batch_nnz_(const ChunkSlot& slot, const std::vector<int>& cols);

    // --- Worker thread entry point ---
    void worker_thread_();
};
