// SPDX-License-Identifier: MIT
// data_loader.h — Per-species batch producer. Passive, driven by Ring's dispatcher.

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <random>
#include <string>
#include <vector>

#include "slot.h"

// Forward declarations
class Batch;

// Concatenation policy for multi-file chunks.
enum class ConcatPolicy { CONCAT_HOST, POINTER_LIST };

// Per-species batch producer.
// Owns a fixed set of Slots and manages the on-disk file order and current chunk state.
// No internal threads; all work runs on Ring's pool worker threads.
class DataLoader {
public:
    // Constructor: store configuration, don't touch files or create Slots yet.
    // Defers feature_count peek and Slot creation to start().
    //
    // Arguments:
    //   species_name    - human-readable identifier
    //   file_paths      - non-empty vector of .1pz file paths
    //   chunk_size      - number of files per chunk (>0; recommend 64)
    //   batch_size      - columns per batch (>0; recommend 512)
    //   n_slots         - number of prefetch Slots (>0; recommend 4)
    //   rng             - PRNG for epoch shuffles (by ref)
    //   policy          - CONCAT_HOST (default) or POINTER_LIST
    //   omp_threads     - parallel decode threads (default 16; 0 = use all available)
    //   max_nnz_estimate - max expected nnz per batch (default batch_size * 2000)
    DataLoader(std::string species_name,
               std::vector<std::string> file_paths,
               int chunk_size,
               int batch_size,
               int n_slots,
               std::mt19937& rng,
               ConcatPolicy policy = ConcatPolicy::CONCAT_HOST,
               int omp_threads = 16,
               int max_nnz_estimate = -1);

    ~DataLoader();

    // Non-copyable, non-movable.
    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    // ========================================================================
    // Lifecycle
    // ========================================================================

    // Start the loader: shuffle file order once, peek feature count,
    // construct Slots, prepare to serve batches.
    void start();

    // Feature count (only valid after start()).
    int feature_count() const;

    // Species name.
    const std::string& species_name() const;

    // Epoch/pass counter (increments when file list wraps).
    int pass() const;

    // ========================================================================
    // Ring dispatcher interface
    // ========================================================================

    // Count free slots (state == FREE).
    bool has_free_slot() const;
    int free_slot_count() const;
    int ready_slot_count() const;

    // Atomically reserve a free slot (FREE → FILLING); returns nullptr if none.
    Slot* reserve_free_slot();

    // Fill a reserved slot: load chunk if needed, pack batch, construct Batch,
    // call batch->prepare(), transition slot FILLING → READY.
    // Runs on a Ring pool worker; returns immediately.
    void fill(Slot* slot);

    // Block (condition_variable) until any slot is READY, then move its Batch out
    // and return it. Returns nullptr if none (only on shutdown). The Batch's
    // destructor will later flip the slot back to FREE.
    std::unique_ptr<Batch> take_ready_batch();

private:
    // ========================================================================
    // Private data structures (matching gpu_data_loader logic)
    // ========================================================================

    // Per-file decoded data (POINTER_LIST policy).
    struct DecodedFile {
        std::vector<int32_t> col_ptr;   // length n_cols+1
        std::vector<int32_t> row_idx;   // length nnz
        std::vector<float> values;      // length nnz
        int n_cols = 0;                 // number of columns in this file
        int m = 0;                      // number of rows (features)
    };

    // Unified chunk data (concatenated all files into one CSC).
    struct ChunkDataConcat {
        std::vector<int32_t> col_ptr;   // length total_cols+1
        std::vector<int32_t> row_idx;   // length total_nnz
        std::vector<float> values;      // length total_nnz
        int n_cols = 0;                 // total columns across all files
        int m = 0;                      // number of rows (features)
    };

    // Chunk data (union of both policies).
    struct ChunkData {
        ConcatPolicy policy;
        ChunkDataConcat concat;         // used for CONCAT_HOST
        std::vector<DecodedFile> files; // used for POINTER_LIST
        std::vector<std::pair<int, int>> column_permutation; // (file_idx, local_col)
        int n_cols = 0;
        int m = 0;
    };

    // ========================================================================
    // Private methods
    // ========================================================================

    // Ensure a chunk is loaded into memory. If current chunk is exhausted,
    // decode the next chunk_size files, shuffle columns, reshuffle at epoch wrap.
    // Must be called with chunk_mtx_ locked.
    void ensure_chunk_loaded_locked();

    // Pack the next batch_size columns from the current chunk into host-side CSC.
    // Fills col_ptr, row_idx, values vectors and sets nnz and is_chunk_end.
    void pack_batch_host_csc(std::vector<int32_t>& col_ptr,
                              std::vector<int32_t>& row_idx,
                              std::vector<float>& values,
                              int& nnz,
                              bool& is_chunk_end);

    // Check if any slot is in READY state.
    bool any_slot_ready() const;

    // Find first slot in READY state; return its index or -1.
    int find_ready_slot() const;

    // Find the index of a slot in the slots_ vector; return -1 if not found.
    int find_slot_index(Slot* slot) const;

    // ========================================================================
    // Private data
    // ========================================================================

    std::string species_name_;
    std::vector<std::string> file_paths_;
    int chunk_size_;
    int batch_size_;
    int n_slots_;
    ConcatPolicy policy_;
    int omp_threads_;
    int max_nnz_estimate_;

    // RNG reference (used for shuffling, not thread-safe during epochs).
    std::mt19937& rng_;

    // Feature count (set in start()).
    std::atomic<int> m_{-1};

    // Pass counter (incremented when file list wraps).
    std::atomic<int> pass_{0};

    // Slots and per-slot Batches.
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<std::unique_ptr<Batch>> batch_per_slot_;

    // Chunk state (current chunk in memory, column permutation, cursor).
    std::mutex chunk_mtx_;
    std::unique_ptr<ChunkData> current_chunk_;
    int col_cursor_ = 0;               // current column offset in current_chunk_
    int file_cursor_ = 0;              // current file index in shuffled order
    std::vector<std::string> file_order_; // shuffled file paths

    // Slot state synchronization (for take_ready_batch blocking).
    std::mutex slot_mtx_;
    std::condition_variable slot_cv_;
    bool started_ = false;
};
