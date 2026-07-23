// SPDX-License-Identifier: MIT
// data_loader.cu — Per-species batch producer implementation.

#include "data_loader.h"
#include "batch.h"
#include "slot.h"

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
// Saturate cast helpers (file-scope with atomic warning flags)
// ============================================================================

static std::atomic<bool> saturate_warned_idx{false};
static std::atomic<bool> saturate_warned_val{false};

static inline int32_t saturate_cast_u32_to_i32(uint32_t v) {
    if (v > static_cast<uint32_t>(INT32_MAX)) {
        bool exp = false;
        if (saturate_warned_idx.compare_exchange_strong(exp, true)) {
            std::fprintf(stderr, "[DataLoader] WARNING: uint32 index %u > INT32_MAX; truncating.\n", v);
        }
        return INT32_MAX;
    }
    return static_cast<int32_t>(v);
}

static inline float saturate_cast_u32_to_f32(uint32_t v) {
    if (v > (1u << 24)) {
        bool exp = false;
        if (saturate_warned_val.compare_exchange_strong(exp, true)) {
            std::fprintf(stderr, "[DataLoader] WARNING: uint32 value %u > 2^24; fp32 cast loses precision.\n", v);
        }
    }
    return static_cast<float>(v);
}

// ============================================================================
// Public API: Constructor and destructor
// ============================================================================

DataLoader::DataLoader(std::string species_name,
                       std::vector<std::string> file_paths,
                       int chunk_size,
                       int batch_size,
                       int n_slots,
                       std::mt19937& rng,
                       ConcatPolicy policy,
                       int omp_threads,
                       int max_nnz_estimate)
    : species_name_(std::move(species_name)),
      file_paths_(std::move(file_paths)),
      chunk_size_(chunk_size),
      batch_size_(batch_size),
      n_slots_(n_slots),
      policy_(policy),
      omp_threads_(omp_threads > 0 ? omp_threads : 16),
      max_nnz_estimate_(max_nnz_estimate),
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
        singlet::pz::ReadResult r = singlet::pz::read_1pz(file_paths_[0]);
        m_ = r.m;
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << file_paths_[0] << ": " << e.what();
        throw std::runtime_error(oss.str());
    }

    // Create Slots now that we know the feature count
    slots_.resize(n_slots_);
    batch_per_slot_.resize(n_slots_);
    for (int i = 0; i < n_slots_; ++i) {
        slots_[i] = std::make_unique<Slot>(m_, batch_size_, max_nnz_estimate_);
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

int DataLoader::pass() const {
    return pass_;
}

// ============================================================================
// Slot counting and reservation
// ============================================================================

bool DataLoader::has_free_slot() const {
    for (const auto& slot : slots_) {
        if (slot->state() == Slot::State::FREE) {
            return true;
        }
    }
    return false;
}

int DataLoader::free_slot_count() const {
    int count = 0;
    for (const auto& slot : slots_) {
        if (slot->state() == Slot::State::FREE) {
            ++count;
        }
    }
    return count;
}

int DataLoader::ready_slot_count() const {
    int count = 0;
    for (const auto& slot : slots_) {
        if (slot->state() == Slot::State::READY) {
            ++count;
        }
    }
    return count;
}

Slot* DataLoader::reserve_free_slot() {
    std::lock_guard<std::mutex> lock(slot_mtx_);
    for (int i = 0; i < n_slots_; ++i) {
        if (slots_[i]->state() == Slot::State::FREE) {
            slots_[i]->mark_filling();
            return slots_[i].get();
        }
    }
    return nullptr;
}

// ============================================================================
// Batch filling: the main pool-worker entry point
// ============================================================================

void DataLoader::fill(Slot* slot) {
    assert(slot != nullptr);
    assert(slot->state() == Slot::State::FILLING);

    // Find which slot index this is (for batch_per_slot_ indexing)
    int slot_idx = -1;
    for (int i = 0; i < n_slots_; ++i) {
        if (slots_[i].get() == slot) {
            slot_idx = i;
            break;
        }
    }
    assert(slot_idx >= 0);

    // Ensure a chunk is loaded
    {
        std::lock_guard<std::mutex> lock(chunk_mtx_);
        ensure_chunk_loaded_locked();
    }

    // Pack the next batch from the current chunk
    std::vector<int32_t> col_ptr;
    std::vector<int32_t> row_idx;
    std::vector<float> values;
    int nnz = 0;
    bool is_chunk_end = false;

    pack_batch_host_csc(col_ptr, row_idx, values, nnz, is_chunk_end);

    // Construct Batch (which copies CSC data into slot's pinned buffers)
    auto batch = std::make_unique<Batch>(
        slot,
        species_name_,
        col_ptr.data(),
        row_idx.data(),
        values.data(),
        m_,
        batch_size_,
        nnz,
        is_chunk_end);

    // Prepare the batch (H2D transfer, log-norm kernel, event record, mark READY)
    batch->prepare();

    // Store the batch for later consumption
    {
        std::lock_guard<std::mutex> lock(slot_mtx_);
        int slot_idx = find_slot_index(slot);
        batch_per_slot_[slot_idx] = std::move(batch);
    }

    // Notify take_ready_batch() that a slot is now READY
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

bool DataLoader::any_slot_ready() const {
    for (const auto& slot : slots_) {
        if (slot->state() == Slot::State::READY) {
            return true;
        }
    }
    return false;
}

int DataLoader::find_ready_slot() const {
    for (int i = 0; i < n_slots_; ++i) {
        if (slots_[i]->state() == Slot::State::READY && batch_per_slot_[i]) {
            return i;
        }
    }
    return -1;
}

std::unique_ptr<Batch> DataLoader::take_ready_batch() {
    std::unique_lock<std::mutex> lock(slot_mtx_);

    // Block until any slot is READY or we shut down
    slot_cv_.wait(lock, [this]() { return any_slot_ready() || !started_; });

    if (!started_) {
        return nullptr;  // Shutdown signal
    }

    // Find a READY slot with a non-null Batch
    int idx = find_ready_slot();
    if (idx < 0) {
        return nullptr;  // No ready batch
    }

    // Move the Batch out and return it
    auto batch = std::move(batch_per_slot_[idx]);
    batch_per_slot_[idx] = nullptr;

    // Notify fill() workers that a slot is now free
    slot_cv_.notify_all();

    return batch;
}

// ============================================================================
// Chunk loading and packing (private helpers)
// ============================================================================

void DataLoader::ensure_chunk_loaded_locked() {
    // If we have a current chunk with remaining columns, nothing to do
    if (current_chunk_ && col_cursor_ < current_chunk_->n_cols) {
        return;
    }

    // Need to load a new chunk
    // Check if we've exhausted all files; if so, reshuffle and bump pass
    if (file_cursor_ >= (int)file_order_.size()) {
        std::shuffle(file_order_.begin(), file_order_.end(), rng_);
        file_cursor_ = 0;
        pass_++;
    }

    // Decode the next chunk_size files
    int chunk_end = std::min(file_cursor_ + chunk_size_, (int)file_order_.size());
    std::vector<singlet::pz::ReadResult> decoded_files;
    decoded_files.resize(chunk_end - file_cursor_);

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

    // Build ChunkData based on policy
    current_chunk_ = std::make_unique<ChunkData>();
    current_chunk_->policy = policy_;
    current_chunk_->m = m_;

    if (policy_ == ConcatPolicy::CONCAT_HOST) {
        // Compute offsets
        std::vector<int> nnz_off(decoded_files.size() + 1, 0);
        std::vector<int> col_off(decoded_files.size() + 1, 0);
        int total_nnz = 0;
        int total_cols = 0;
        for (size_t i = 0; i < decoded_files.size(); ++i) {
            nnz_off[i] = total_nnz;
            col_off[i] = total_cols;
            total_nnz += decoded_files[i].nnz;
            total_cols += decoded_files[i].n;
        }
        nnz_off[decoded_files.size()] = total_nnz;
        col_off[decoded_files.size()] = total_cols;

        current_chunk_->concat.n_cols = total_cols;
        current_chunk_->n_cols = total_cols;

        // Allocate unified buffers
        current_chunk_->concat.col_ptr.resize(total_cols + 1);
        current_chunk_->concat.row_idx.resize(total_nnz);
        current_chunk_->concat.values.resize(total_nnz);

        // Parallel copy+cast
        #pragma omp parallel for schedule(dynamic) num_threads(omp_threads_)
        for (size_t i = 0; i < decoded_files.size(); ++i) {
            auto& r = decoded_files[i];
            int col_base = col_off[i];
            int nnz_base = nnz_off[i];

            // Copy and cast col_ptr
            for (uint32_t c = 0; c <= r.n; ++c) {
                current_chunk_->concat.col_ptr[col_base + c] =
                    saturate_cast_u32_to_i32(r.indptr[c]) + nnz_base;
            }

            // Copy and cast row_idx and values
            for (uint64_t j = 0; j < r.nnz; ++j) {
                current_chunk_->concat.row_idx[nnz_base + j] =
                    saturate_cast_u32_to_i32(r.indices[j]);
                current_chunk_->concat.values[nnz_base + j] =
                    saturate_cast_u32_to_f32(r.data[j]);
            }
        }

        // Fix col_ptr base (0-based for this chunk)
        int base = current_chunk_->concat.col_ptr[0];
        for (int& c : current_chunk_->concat.col_ptr) {
            c -= base;
        }

    } else {
        // POINTER_LIST: store per-file decoded arrays
        current_chunk_->files.resize(decoded_files.size());

        #pragma omp parallel for schedule(dynamic) num_threads(omp_threads_)
        for (size_t i = 0; i < decoded_files.size(); ++i) {
            auto& r = decoded_files[i];
            auto& df = current_chunk_->files[i];

            df.n_cols = r.n;
            df.m = r.m;
            df.col_ptr.resize(r.n + 1);
            df.row_idx.resize(r.nnz);
            df.values.resize(r.nnz);

            for (uint32_t c = 0; c <= r.n; ++c) {
                df.col_ptr[c] = saturate_cast_u32_to_i32(r.indptr[c]);
            }
            for (uint64_t j = 0; j < r.nnz; ++j) {
                df.row_idx[j] = saturate_cast_u32_to_i32(r.indices[j]);
                df.values[j] = saturate_cast_u32_to_f32(r.data[j]);
            }
        }

        // Compute total columns
        int total_cols = 0;
        for (auto& df : current_chunk_->files) {
            total_cols += df.n_cols;
        }
        current_chunk_->n_cols = total_cols;
    }

    // Reset column cursor for this chunk
    col_cursor_ = 0;

    // Advance file cursor
    file_cursor_ = chunk_end;

    // For POINTER_LIST, generate and shuffle column permutation
    if (policy_ == ConcatPolicy::POINTER_LIST) {
        current_chunk_->column_permutation.clear();
        for (size_t fi = 0; fi < current_chunk_->files.size(); ++fi) {
            for (int ci = 0; ci < current_chunk_->files[fi].n_cols; ++ci) {
                current_chunk_->column_permutation.push_back({(int)fi, ci});
            }
        }
        // Shuffle columns using RNG
        std::mt19937 col_rng(rng_());
        std::shuffle(current_chunk_->column_permutation.begin(),
                     current_chunk_->column_permutation.end(), col_rng);
    }

void DataLoader::pack_batch_host_csc(std::vector<int32_t>& col_ptr,
                                      std::vector<int32_t>& row_idx,
                                      std::vector<float>& values,
                                      int& nnz,
                                      bool& is_chunk_end) {
    assert(current_chunk_ != nullptr);

    // Ensure we're not past the end
    if (col_cursor_ >= current_chunk_->n_cols) {
        throw std::runtime_error("DataLoader: col_cursor past end of chunk (no more batches)");
    }

    col_ptr.resize(batch_size_ + 1);
    col_ptr[0] = 0;
    nnz = 0;

    if (policy_ == ConcatPolicy::CONCAT_HOST) {
        // Extract batch_size columns from concatenated chunk
        for (int b = 0; b < batch_size_ && col_cursor_ + b < current_chunk_->n_cols; ++b) {
            int global_col = col_cursor_ + b;
            int start = current_chunk_->concat.col_ptr[global_col];
            int end = current_chunk_->concat.col_ptr[global_col + 1];
            int col_nnz = end - start;
            nnz += col_nnz;
            col_ptr[b + 1] = nnz;
        }
        // Pad with zeros if we reached end of chunk
        for (int b = (current_chunk_->n_cols - col_cursor_); b < batch_size_; ++b) {
            col_ptr[b + 1] = nnz;
        }

        // Allocate and fill row_idx and values
        row_idx.resize(nnz);
        values.resize(nnz);
        int offset = 0;
        for (int b = 0; b < batch_size_ && col_cursor_ + b < current_chunk_->n_cols; ++b) {
            int global_col = col_cursor_ + b;
            int start = current_chunk_->concat.col_ptr[global_col];
            int end = current_chunk_->concat.col_ptr[global_col + 1];
            int col_nnz = end - start;
            std::memcpy(row_idx.data() + offset,
                        current_chunk_->concat.row_idx.data() + start,
                        col_nnz * sizeof(int32_t));
            std::memcpy(values.data() + offset,
                        current_chunk_->concat.values.data() + start,
                        col_nnz * sizeof(float));
            offset += col_nnz;
        }

    } else {
        // POINTER_LIST: extract batch_size columns from per-file arrays
        for (int b = 0; b < batch_size_ && col_cursor_ + b < current_chunk_->n_cols; ++b) {
            auto [fi, ci] = current_chunk_->column_permutation[col_cursor_ + b];
            auto& file = current_chunk_->files[fi];
            int start = file.col_ptr[ci];
            int end = file.col_ptr[ci + 1];
            int col_nnz = end - start;
            nnz += col_nnz;
            col_ptr[b + 1] = nnz;
        }
        // Pad with zeros if we reached end
        for (int b = (current_chunk_->n_cols - col_cursor_); b < batch_size_; ++b) {
            col_ptr[b + 1] = nnz;
        }

        // Allocate and fill row_idx and values
        row_idx.resize(nnz);
        values.resize(nnz);
        int offset = 0;
        for (int b = 0; b < batch_size_ && col_cursor_ + b < current_chunk_->n_cols; ++b) {
            auto [fi, ci] = current_chunk_->column_permutation[col_cursor_ + b];
            auto& file = current_chunk_->files[fi];
            int start = file.col_ptr[ci];
            int end = file.col_ptr[ci + 1];
            int col_nnz = end - start;
            std::memcpy(row_idx.data() + offset, file.row_idx.data() + start,
                        col_nnz * sizeof(int32_t));
            std::memcpy(values.data() + offset, file.values.data() + start,
                        col_nnz * sizeof(float));
            offset += col_nnz;
        }
    }

    // Determine chunk_end: true if this batch consumed the last columns of the chunk
    is_chunk_end = (col_cursor_ + batch_size_ >= current_chunk_->n_cols);

    // Advance column cursor for next batch
    col_cursor_ += batch_size_;
}
