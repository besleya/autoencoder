// SPDX-License-Identifier: MIT
// gpu_data_loader.h — async sparse minibatch GPU DataLoader delegating to Ring.
//
// Implements PLAN.md specification: two-thread async loader (T_CL: chunk-loader,
// T_BB: batch-builder) that delegates slot/buffer/event management to an
// externally-owned Ring instance.
//
// The DataLoader is a producer for a single lane in the Ring. The trainer
// talks directly to the Ring for batch consumption.

#pragma once

#include "ring.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

// Concatenation policy for multi-file chunks.
enum class ConcatPolicy { CONCAT_HOST, POINTER_LIST };

// Async sparse GPU DataLoader delegating to Ring.
//
// Architecture: Two worker threads (T_CL: chunk-loader, T_BB: batch-builder)
// that delegate all slot/buffer/event management to an external Ring instance.
// RNG is only touched in begin_epoch() on the main thread — not thread-safe
// during epochs. Each DataLoader instance is a single producer lane in the Ring.
class DataLoader {
public:
    // Constructor. Validates args; opens first file to peek m and calls
    // ring->set_m(m); creates loader stream; spawns T_CL and T_BB threads.
    // Does NOT call begin_epoch.
    //
    // Arguments:
    //   paths:        non-empty vector of .1pz file paths
    //   chunk_size:   number of files per chunk (>0; recommend 64)
    //   batch_size:   columns per batch (>0; recommend 512)
    //   rng:          PRNG for epoch shuffles (by ref; consumed in begin_epoch)
    //   ring:         externally owned Ring; this loader produces into ring with lane_id
    //   lane_id:      lane index in the ring (0..ring->n_lanes()-1)
    //   policy:       CONCAT_HOST (default) or POINTER_LIST
    //   omp_threads:  parallel decode threads (default 16; 0 = use all available)
    //
    // Throws: std::runtime_error on invalid args or file I/O errors.
    DataLoader(const std::vector<std::string>& paths,
               int chunk_size,
               int batch_size,
               std::mt19937& rng,
               Ring* ring,
               int lane_id,
               ConcatPolicy policy = ConcatPolicy::CONCAT_HOST,
               int omp_threads = 16);

    ~DataLoader();

    // Non-copyable, non-movable (threads hold unique_ptr to Impl).
    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    // Start the data loader: shuffle file order once and unblock workers.
    // Called once before the training loop. Returns immediately.
    void start();

    // Number of features (peeked from first file in constructor).
    int m() const;

    // CUDA stream on which H2D + lognorm are enqueued.
    cudaStream_t loader_stream() const;

    // Lane ID (for reference, typically used by Ring methods).
    int lane_id() const;

public:
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// Standalone log-normalization helper (free function).
// Applies per-column log1p normalization in-place on device values.
// scaler=10000.0f matches the DataLoader's internal kSparseLogNormScaler.
// The kernel launch is identical to the one inside batch_builder_thread.
// ---------------------------------------------------------------------------
void log_normalize_csc_columns(int n_cols,
                                float scaler,
                                const int32_t* d_col_ptr,
                                float* d_values,
                                cudaStream_t stream);
