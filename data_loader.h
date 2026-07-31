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
class Ring;

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
    //   rng             - PRNG for epoch shuffles (by value; owned copy per loader)
    //   policy          - CONCAT_HOST (default) or POINTER_LIST
    //   omp_threads     - parallel decode threads (default 16; 0 = use all available)
    //   max_nnz_estimate - max expected nnz per batch (default batch_size * 2000)
    //   double_buffer_chunks - enable async decode of next chunk (default true)
    DataLoader(std::string species_name,
               std::vector<std::string> file_paths,
               int chunk_size,
               int batch_size,
               int n_slots,
               std::mt19937 rng,
               ConcatPolicy policy = ConcatPolicy::CONCAT_HOST,
               int omp_threads = 16,
               int max_nnz_estimate = -1,
               bool double_buffer_chunks = true);

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

    // Epoch counter (increments when file list wraps).
    int epoch() const;

    // ========================================================================
    // Ring back-reference (called by Ring::add_loader)
    // ========================================================================

    // Set the Ring reference. Called by Ring::add_loader() after registration.
    // DataLoader uses this to access Ring::decode_pool() for submitting decode tasks.
    void set_ring(Ring* ring);

    // ========================================================================
    // Ring dispatcher interface — blocking variant
    // ========================================================================

    // Block (via CUDA event wait) until this loader's next slot (in rotation) is empty,
    // then atomically reserve it (FREE → FILLING). Strict rotation: always the same
    // sequence. Blocks the calling thread (dispatcher) until THIS slot is ready.
    Slot* reserve_slot();

    // Count free slots (state == FREE).
    bool has_free_slot() const;
    int free_slot_count() const;
    int ready_slot_count() const;

    // Fill a reserved slot: load chunk if needed, pack batch, construct Batch,
    // call batch->prepare(), transition slot FILLING → READY.
    // Runs on a Ring pool worker; returns immediately.
    void fill(Slot* slot);

    // Block (condition_variable) until next slot in rotation is READY, then move its Batch out
    // and return it. Returns nullptr if none (only on shutdown). The Batch's
    // destructor will later flip the slot back to FREE.
    std::unique_ptr<Batch> take_ready_batch();

private:
    // ========================================================================
    // Private data structures
    // ========================================================================

    // Unified chunk data (concatenated all files into one CSC matrix).
    // Matches the Chunk struct in batch.h: stores uint32_t (raw from read_1pz).
    struct Chunk {
        int m = 0;                       // rows/features
        int n = 0;                       // total columns across concatenated files
        std::vector<uint32_t> col_ptr;   // length n+1, cumulative nnz offsets
        std::vector<uint32_t> row_idx;   // length nnz, row indices
        std::vector<uint32_t> values;    // length nnz, raw counts (from read_1pz)
    };

    // ========================================================================
    // Private methods
    // ========================================================================

    // Ensure a chunk is loaded into memory. If current chunk is exhausted,
    // decode the next chunk_size files, shuffle columns, reshuffle at epoch wrap.
    // Must be called with chunk_mtx_ locked.
    void advance_chunk();

    // Start async decode of next chunk. Called from DataLoader::fill()
    // when this batch claims the chunk's last columns. Non-blocking: submits tasks
    // to decode_pool_ and stores the future. Guarded by chunk_mtx_.
    void decode_next_chunk();

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
    bool double_buffer_chunks_;

    // RNG owned by this loader (seeded per-loader for determinism).
    std::mt19937 rng_;

    // Ring back-reference (set by Ring::add_loader).
    Ring* ring_ = nullptr;

    // Feature count (set in start()).
    std::atomic<int> m_{-1};

    // Epoch counter (incremented when file list wraps).
    std::atomic<int> epoch_{0};

    // Slots and per-slot Batches.
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<std::unique_ptr<Batch>> batch_per_slot_;

    // Slot rotation indices (single-writer rule: only dispatcher writes fill_idx_,
    // only trainer writes consume_idx_).
    int fill_idx_ = 0;              // next slot to fill (dispatcher writes this)
    int consume_idx_ = 0;           // next slot to consume (trainer writes this)

    // Chunk state (current chunk in memory, column permutation, cursor).
    std::mutex chunk_mtx_;
    std::shared_ptr<const Chunk> current_chunk_;
    std::shared_future<std::shared_ptr<const Chunk>> next_chunk_future_;
    bool next_chunk_submitted_ = false;  // guard against double-submit
    int col_cursor_ = 0;               // current column offset in current_chunk_
    int file_cursor_ = 0;              // current file index in shuffled order
    std::vector<std::string> file_order_; // shuffled file paths

    // Slot state synchronization (for take_ready_batch blocking).
    std::mutex slot_mtx_;
    std::condition_variable slot_cv_;
    bool started_ = false;
};
