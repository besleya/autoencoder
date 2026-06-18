// SPDX-License-Identifier: MIT
// gpu_data_loader.h — async sparse minibatch GPU DataLoader with ring buffer.
//
// Implements DESIGN.md specification: two-thread async loader (T_CL, T_BB)
// with ring buffer (default depth=4) and chunk queue (depth=1).
// CPU-async H2D transfers + GPU lognorm kernel.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

// Concatenation policy for multi-file chunks.
enum class ConcatPolicy { CONCAT_HOST, POINTER_LIST };

// A single mini-batch: sparse CSC of shape (m × B) resident on device.
// All pointers are device-resident. The event is signaled after H2D + lognorm
// are enqueued (GPU signals when complete). Trainer pattern:
//   next_batch(&b);
//   cudaStreamWaitEvent(trainer_stream, b.ready_event, 0);  // GPU fence only
//   net.forward(b, trainer_stream);
//
// Lifetime: pointers and event remain valid until ring_depth-th subsequent
// next_batch() call (default K=4). Trainer holds at most 1 batch at a time.
struct SparseBatch {
    int m;              // features (rows), same as dataset
    int B;              // batch size (cols)
    int nnz;            // nonzeros in this batch
    const int32_t* d_col_ptr;   // device, length B+1
    const int32_t* d_row_idx;   // device, length nnz
    const float*   d_values;    // device, length nnz
    cudaEvent_t    ready_event; // signaled after H2D + lognorm; owned by slot
    bool           eof_after;   // true on the last batch of epoch
};

// Async sparse GPU DataLoader with prefetching.
//
// Architecture: Two worker threads (T_CL: chunk-loader, T_BB: batch-builder)
// synchronized via ring buffer and chunk queue. RNG is only touched in
// begin_epoch() on the main thread — not thread-safe during epochs.
class DataLoader {
public:
    // Constructor. Validates args; opens first file to peek m; creates stream;
    // spawns worker threads. Does NOT call begin_epoch.
    //
    // Arguments:
    //   paths:                vector of .1pz file paths (non-empty)
    //   chunk_size:           number of files per chunk (> 0)
    //   batch_size:           number of columns per batch (> 0)
    //   rng:                  PRNG for epoch shuffles (by reference; consumed in begin_epoch)
    //   policy:               CONCAT_HOST (default) or POINTER_LIST
    //   ring_depth:           ring buffer depth (default 4, > 0)
    //   n_concurrent_loaders: for OMP thread budgeting (default 1, > 0)
    //
    // Throws: std::runtime_error on invalid args or file I/O errors.
    DataLoader(const std::vector<std::string>& paths,
               int chunk_size, int batch_size,
               std::mt19937& rng,
               ConcatPolicy policy = ConcatPolicy::CONCAT_HOST,
               int ring_depth = 4,
               int n_concurrent_loaders = 1);

    ~DataLoader();

    // Non-copyable, non-movable (threads hold unique_ptr to Impl).
    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    // Begin a new epoch: shuffle file order, reset cursors, unblock worker threads.
    // Returns immediately (does NOT wait for first batch). Consumes RNG state.
    // Must be called exactly once per epoch before next_batch().
    void begin_epoch();

    // Fetch next batch. Returns true if a batch is available; false when the
    // epoch is exhausted. Fills *out with device pointers + event + eof flag.
    // On eof_after=true, the NEXT call to next_batch will return false.
    bool next_batch(SparseBatch* out);

    // Number of features (peeked from first file in constructor).
    int m() const;

    // CUDA stream on which H2D + lognorm are enqueued.
    cudaStream_t loader_stream() const;

public:
    struct Impl;

// ---------------------------------------------------------------------------
// Standalone log-normalization helper.
// Applies per-column log1p normalization in-place on device values.
// scaler=10000.0f matches the DataLoader's internal kSparseLogNormScaler.
// The kernel launch is identical to the one inside batch_builder_thread.
// ---------------------------------------------------------------------------
void log_normalize_csc_columns(int n_cols,
                                float scaler,
                                const int32_t* d_col_ptr,
                                float* d_values,
                                cudaStream_t stream);
    
private:
    std::unique_ptr<Impl> impl_;
};
