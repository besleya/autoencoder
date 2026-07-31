// SPDX-License-Identifier: MIT
// data_loader.cu — Per-species batch producer implementation.

#include "data_loader.h"
#include "batch.h"
#include "slot.h"
#include "validate_1pz.h"

#include <singlet/pileup/pz_reader.h>
#include <singlet/gpu/core/types.h>

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <sstream>
#include <stdexcept>

// ============================================================================
// CUDA error checking
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::ostringstream oss; \
        oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
            << cudaGetErrorString(err); \
        throw std::runtime_error(oss.str()); \
    } \
} while (0)

#define CUDA_CHECK_SILENT(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err)); \
    } \
} while (0)

// ============================================================================
// Public API: Constructor and destructor
// ============================================================================

DataLoader::DataLoader(std::string species_name,
                       std::vector<std::string> file_paths,
                       int chunk_size,
                       int batch_size,
                       int n_slots,
                       std::mt19937 rng,
                       ConcatPolicy policy,
                       int omp_threads,
                       int max_nnz_estimate,
                       bool double_buffer_chunks)
    : species_name_(std::move(species_name)),
      file_paths_(std::move(file_paths)),
      chunk_size_(chunk_size),
      batch_size_(batch_size),
      n_slots_(n_slots),
      policy_(policy),
      omp_threads_(omp_threads > 0 ? omp_threads : 16),
      max_nnz_estimate_(max_nnz_estimate),
      double_buffer_chunks_(double_buffer_chunks),
      rng_(rng) {
    // Validate arguments
    if (file_paths_.empty()) {
        throw std::runtime_error("DataLoader: no files provided");
    }
    if (chunk_size <= 0) {
        throw std::runtime_error("DataLoader: chunk_size must be > 0");
    }
    if (batch_size <= 0) {
        throw std::runtime_error("DataLoader: batch_size must be > 0");
    }
    if (n_slots <= 0) {
        throw std::runtime_error("DataLoader: n_slots must be > 0");
    }

    // Set reasonable default max_nnz_estimate if not provided
    if (max_nnz_estimate_ <= 0) {
        max_nnz_estimate_ = batch_size_ * 2000;
    }
}

DataLoader::~DataLoader() {
    // Batch destructors (if any) will mark slots FREE.
    // No cleanup needed here.
}

// ============================================================================
// Lifecycle: start() and feature_count()
// ============================================================================

void DataLoader::start() {
    // Peek first file for feature count
    try {
        PZHeader hdr = validate_1pz(file_paths_[0]);
        m_ = hdr.m;
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << file_paths_[0] << ": " << e.what();
        throw std::runtime_error(oss.str());
    }

    // Create Slots now that we know the feature count
    slots_.resize(n_slots_);
    batch_per_slot_.resize(n_slots_);
    for (int i = 0; i < n_slots_; ++i) {
        slots_[i] = std::make_unique<Slot>(batch_size_ + 1, max_nnz_estimate_);
        batch_per_slot_[i] = nullptr;
    }

    // Shuffle file order
    file_order_ = file_paths_;
    std::shuffle(file_order_.begin(), file_order_.end(), rng_);
    file_cursor_ = 0;
    col_cursor_ = 0;
    current_chunk_ = nullptr;

    // Mark as started
    {
        std::lock_guard<std::mutex> lock(slot_mtx_);
        started_ = true;
    }
    slot_cv_.notify_all();
}

int DataLoader::feature_count() const {
    assert(m_ >= 0 && "feature_count() called before start()");
    return m_;
}

const std::string& DataLoader::species_name() const {
    return species_name_;
}

int DataLoader::epoch() const {
    return epoch_;
}

// ============================================================================
// Ring back-reference
// ============================================================================

void DataLoader::set_ring(Ring* ring) {
    if (!ring) {
        throw std::runtime_error("DataLoader::set_ring: null ring pointer");
    }
    ring_ = ring;
}

// ============================================================================
// Slot rotation interface (blocking)
// ============================================================================

Slot* DataLoader::reserve_slot() {
    assert(ring_ != nullptr && "reserve_slot() called before set_ring()");
    assert(fill_idx_ >= 0 && fill_idx_ < (int)slots_.size());

    // Get the next slot in rotation
    Slot* slot = slots_[fill_idx_].get();

    // Block until this specific slot is EMPTY
    slot->await_empty();

    // Atomically transition from EMPTY -> FILLING
    slot->fill();

    // Advance to next slot in rotation
    fill_idx_ = (fill_idx_ + 1) % (int)slots_.size();

    return slot;
}

// ============================================================================
// Slot counting and reservation
// ============================================================================

bool DataLoader::has_free_slot() const {
    for (const auto& slot : slots_) {
        if (slot->state() == Slot::State::kEmpty) {
            return true;
        }
    }
    return false;
}

int DataLoader::free_slot_count() const {
    int count = 0;
    for (const auto& slot : slots_) {
        if (slot->state() == Slot::State::kEmpty) {
            ++count;
        }
    }
    return count;
}

int DataLoader::ready_slot_count() const {
    int count = 0;
    for (const auto& slot : slots_) {
        if (slot->state() == Slot::State::kFull) {
            ++count;
        }
    }
    return count;
}

// ============================================================================
// Batch filling: the main pool-worker entry point
// ============================================================================

void DataLoader::fill(Slot* slot) {
    assert(slot != nullptr);
    assert(ring_ != nullptr && "fill() called before set_ring()");

    // Step 1: Claim next batch's column slice and determine if last (short critical section)
    std::shared_ptr<const Chunk> chunk_snapshot;
    std::vector<int> column_indices;
    bool is_chunk_end = false;

    {
        std::lock_guard<std::mutex> lock(chunk_mtx_);

        // Loop until we claim a full batch, skipping trailing partial batches
        while (true) {
            // Ensure we have a current chunk
            if (!current_chunk_ || col_cursor_ >= current_chunk_->n) {
                ensure_chunk_loaded_locked();
            }

            int remaining = current_chunk_->n - col_cursor_;

            // Skip trailing partial batch: if fewer than batch_size columns remain,
            // advance to end of chunk and loop to load next chunk
            if (remaining < batch_size_) {
                col_cursor_ = current_chunk_->n;
                continue;  // Loop back to load next chunk
            }

            // We have >= batch_size columns remaining; claim exactly batch_size
            chunk_snapshot = current_chunk_;
            for (int i = 0; i < batch_size_; ++i) {
                column_indices.push_back(col_cursor_ + i);
            }
            col_cursor_ += batch_size_;

            // Check if this is the last full slice of this chunk
            // (i.e., after claiming this batch, fewer than batch_size columns remain)
            int remaining_after = current_chunk_->n - col_cursor_;
            is_chunk_end = (remaining_after < batch_size_);

            break;  // Exit loop; we have a full batch
        }
    }  // Release lock here

    // Step 2: Construct Batch
    auto batch = std::make_unique<Batch>(
        slot,
        species_name_,
        chunk_snapshot,
        column_indices,
        is_chunk_end);

    // Step 3: Prepare batch (gather, normalize, copy to device)
    batch->prepare();

    // Step 4: If this is the last full slice of the chunk, kick off next chunk's decode
    if (is_chunk_end) {
        std::lock_guard<std::mutex> lock(chunk_mtx_);
        start_next_chunk_decode_async();
    }

    // Step 6: Store batch for consumer
    {
        std::lock_guard<std::mutex> lock(slot_mtx_);
        int slot_idx = find_slot_index(slot);
        batch_per_slot_[slot_idx] = std::move(batch);
    }

    // Step 7: Notify consumer
    {
        std::lock_guard<std::mutex> lock(slot_mtx_);
        slot_cv_.notify_all();
    }
}

// Helper: find the index of a slot in the slots_ vector
int DataLoader::find_slot_index(Slot* slot) const {
    for (int i = 0; i < n_slots_; ++i) {
        if (slots_[i].get() == slot) {
            return i;
        }
    }
    return -1;
}

// ============================================================================
// Batch consumer interface
// ============================================================================

std::unique_ptr<Batch> DataLoader::take_ready_batch() {
    assert(consume_idx_ >= 0 && consume_idx_ < (int)slots_.size());

    // Get the next slot in rotation
    Slot* slot = slots_[consume_idx_].get();

    // Block until this specific slot is READY (transitions from FILLING -> READY)
    slot->await_full();

    // Find the batch stored in this slot (should be non-null at this point)
    std::unique_lock<std::mutex> lock(slot_mtx_);
    auto batch = std::move(batch_per_slot_[consume_idx_]);
    batch_per_slot_[consume_idx_] = nullptr;

    // Advance to next slot in rotation
    consume_idx_ = (consume_idx_ + 1) % (int)slots_.size();

    lock.unlock();

    // Notify fill() workers that a slot might be free now
    slot_cv_.notify_all();

    // The batch's destructor will call slot->mark_empty() when consumed
    return batch;
}

// ============================================================================
// Chunk loading and async decode (private helpers)
// ============================================================================

void DataLoader::ensure_chunk_loaded_locked() {
    // If current chunk still has columns remaining, nothing to do
    if (current_chunk_ && col_cursor_ < current_chunk_->n) {
        return;
    }

    // Need to load a new chunk
    auto new_chunk = std::make_shared<Chunk>();

    // Handle epoch boundary: if file_cursor_ >= file_order_.size(), reshuffle and bump epoch
    if (file_cursor_ >= (int)file_order_.size()) {
        std::shuffle(file_order_.begin(), file_order_.end(), rng_);
        file_cursor_ = 0;
        epoch_++;
    }

    // Determine which files to decode for this chunk
    int chunk_end = std::min(file_cursor_ + chunk_size_, (int)file_order_.size());
    int n_files = chunk_end - file_cursor_;

    // Try to get next chunk from async decode if available and double-buffering was enabled
    if (next_chunk_future_.valid() && double_buffer_chunks_) {
        try {
            auto decoded = next_chunk_future_.get();
            if (decoded) {
                current_chunk_ = decoded;
                col_cursor_ = 0;
                next_chunk_submitted_ = false;
                file_cursor_ = chunk_end;
                return;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[DataLoader] Error getting next chunk: %s\n", e.what());
        }
    }

    // Synchronous inline decode: read and decode all files for this chunk
    std::vector<singlet::pz::ReadResult> decoded_files(n_files);

    bool read_error = false;
    std::string error_msg;

    #pragma omp parallel for schedule(dynamic) num_threads(omp_threads_)
    for (int i = file_cursor_; i < chunk_end; ++i) {
        int idx = i - file_cursor_;
        try {
            decoded_files[idx] = singlet::pz::read_1pz(file_order_[i]);
            if (decoded_files[idx].m != (uint32_t)m_) {
                throw std::runtime_error("feature count mismatch");
            }
        } catch (const std::exception& e) {
            #pragma omp critical
            {
                if (!read_error) {
                    read_error = true;
                    error_msg = std::string(file_order_[i]) + ": " + e.what();
                }
            }
        }
    }

    if (read_error) {
        std::fprintf(stderr, "[DataLoader] FATAL: %s\n", error_msg.c_str());
        std::terminate();
    }

    // Compute cumulative nnz and column offsets
    std::vector<uint32_t> nnz_off(n_files + 1, 0);
    std::vector<uint32_t> col_off(n_files + 1, 0);
    uint64_t total_nnz = 0;
    int total_cols = 0;

    for (int i = 0; i < n_files; ++i) {
        nnz_off[i] = total_nnz;
        col_off[i] = total_cols;
        total_nnz += decoded_files[i].nnz;
        total_cols += decoded_files[i].n;
    }
    nnz_off[n_files] = total_nnz;
    col_off[n_files] = total_cols;

    // Overflow guard: check that total_nnz fits in int32_t
    if (total_nnz > INT32_MAX) {
        std::fprintf(stderr, "[DataLoader] FATAL: chunk total nnz %lu exceeds INT32_MAX\n", total_nnz);
        std::terminate();
    }

    new_chunk->m = m_;
    new_chunk->n = total_cols;
    new_chunk->col_ptr.resize(total_cols + 1);
    new_chunk->row_idx.resize(total_nnz);
    new_chunk->values.resize(total_nnz);

    // Parallel concatenation: copy and rebase col_ptr, copy row_idx and values
    #pragma omp parallel for schedule(dynamic) num_threads(omp_threads_)
    for (int i = 0; i < n_files; ++i) {
        const auto& r = decoded_files[i];
        uint32_t col_base = col_off[i];
        uint32_t nnz_base = nnz_off[i];

        // Copy and rebase col_ptr (add nnz_base offset to account for concatenation)
        for (uint32_t c = 0; c <= r.n; ++c) {
            new_chunk->col_ptr[col_base + c] = r.indptr[c] + nnz_base;
        }

        // Copy row_idx and values
        for (uint64_t j = 0; j < r.nnz; ++j) {
            new_chunk->row_idx[nnz_base + j] = r.indices[j];
            new_chunk->values[nnz_base + j] = r.data[j];
        }
    }

    // Normalize col_ptr to be 0-based
    uint32_t base = new_chunk->col_ptr[0];
    for (auto& c : new_chunk->col_ptr) {
        c -= base;
    }

    current_chunk_ = new_chunk;
    col_cursor_ = 0;
    file_cursor_ = chunk_end;
    next_chunk_submitted_ = false;
}

void DataLoader::start_next_chunk_decode_async() {
    // Called from Batch's after_gather callback when the last batch of a chunk is gathered.
    // Submits async decode tasks to ring_->decode_pool() for the next chunk.
    // Stores the resulting future in next_chunk_future_.
    // Must be called with chunk_mtx_ locked.

    assert(ring_ != nullptr && "start_next_chunk_decode_async() called before set_ring()");
    assert(!next_chunk_submitted_ && "start_next_chunk_decode_async() called twice");

    // Guard against double-submit
    next_chunk_submitted_ = true;

    // Determine next chunk's files
    int next_file_cursor = file_cursor_;

    // Handle epoch boundary
    if (next_file_cursor >= (int)file_order_.size()) {
        // This shouldn't happen if called correctly, but guard anyway
        return;
    }

    int next_chunk_end = std::min(next_file_cursor + chunk_size_, (int)file_order_.size());
    int n_files = next_chunk_end - next_file_cursor;

    // Create a lambda to decode all files for next chunk
    auto decode_fn = [this, next_file_cursor, next_chunk_end]() -> std::shared_ptr<const Chunk> {
        // Read all files
        std::vector<singlet::pz::ReadResult> decoded_files(next_chunk_end - next_file_cursor);

        bool read_error = false;
        std::string error_msg;

        #pragma omp parallel for schedule(dynamic) num_threads(omp_threads_)
        for (int i = next_file_cursor; i < next_chunk_end; ++i) {
            int idx = i - next_file_cursor;
            try {
                decoded_files[idx] = singlet::pz::read_1pz(file_order_[i]);
                if (decoded_files[idx].m != (uint32_t)m_) {
                    throw std::runtime_error("feature count mismatch");
                }
            } catch (const std::exception& e) {
                #pragma omp critical
                {
                    if (!read_error) {
                        read_error = true;
                        error_msg = std::string(file_order_[i]) + ": " + e.what();
                    }
                }
            }
        }

        if (read_error) {
            std::fprintf(stderr, "[DataLoader] FATAL in async decode: %s\n", error_msg.c_str());
            std::terminate();
        }

        // Concatenate into Chunk
        auto new_chunk = std::make_shared<Chunk>();

        // Compute offsets
        std::vector<uint32_t> nnz_off(decoded_files.size() + 1, 0);
        std::vector<uint32_t> col_off(decoded_files.size() + 1, 0);
        uint64_t total_nnz = 0;
        int total_cols = 0;

        for (size_t i = 0; i < decoded_files.size(); ++i) {
            nnz_off[i] = total_nnz;
            col_off[i] = total_cols;
            total_nnz += decoded_files[i].nnz;
            total_cols += decoded_files[i].n;
        }
        nnz_off[decoded_files.size()] = total_nnz;
        col_off[decoded_files.size()] = total_cols;

        // Overflow guard
        if (total_nnz > INT32_MAX) {
            std::fprintf(stderr, "[DataLoader] FATAL: chunk total nnz %lu exceeds INT32_MAX\n", total_nnz);
            std::terminate();
        }

        new_chunk->m = m_;
        new_chunk->n = total_cols;
        new_chunk->col_ptr.resize(total_cols + 1);
        new_chunk->row_idx.resize(total_nnz);
        new_chunk->values.resize(total_nnz);

        // Parallel concatenation
        #pragma omp parallel for schedule(dynamic) num_threads(omp_threads_)
        for (size_t i = 0; i < decoded_files.size(); ++i) {
            const auto& r = decoded_files[i];
            uint32_t col_base = col_off[i];
            uint32_t nnz_base = nnz_off[i];

            for (uint32_t c = 0; c <= r.n; ++c) {
                new_chunk->col_ptr[col_base + c] = r.indptr[c] + nnz_base;
            }

            for (uint64_t j = 0; j < r.nnz; ++j) {
                new_chunk->row_idx[nnz_base + j] = r.indices[j];
                new_chunk->values[nnz_base + j] = r.data[j];
            }
        }

        // Normalize col_ptr
        uint32_t base = new_chunk->col_ptr[0];
        for (auto& c : new_chunk->col_ptr) {
            c -= base;
        }

        return new_chunk;
    };

    // Submit the task to decode_pool and store the future
    auto future = ring_->decode_pool().submit_task(decode_fn);
    next_chunk_future_ = future;
}
