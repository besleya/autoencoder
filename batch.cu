// SPDX-License-Identifier: MIT
// batch.cu — Implementation of Batch class. Handles gathering, type-casting, log-norm, and H2D transfer.

#include "batch.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <utility>
#include <sstream>
#include <stdexcept>

// ============================================================================
// CUDA error checking
// ============================================================================

#define BATCH_CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::ostringstream oss; \
        oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
            << cudaGetErrorString(err); \
        throw std::runtime_error(oss.str()); \
    } \
} while (0)

// ============================================================================
// CPU gather+normalize function: fused type-cast and log-normalization
// ============================================================================

namespace {

static void gather_normalize(
    const uint32_t* src_col_ptr,
    const uint32_t* src_row_idx,
    const uint32_t* src_values,
    const std::vector<int>& column_indices,
    int32_t* dst_col_ptr,
    int32_t* dst_row_idx,
    float* dst_values,
    float scale) {

    const int B = static_cast<int>(column_indices.size());

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < B; ++j) {
        const uint32_t col_idx = column_indices[j];
        const uint32_t src_start = src_col_ptr[col_idx];
        const uint32_t src_end = src_col_ptr[col_idx + 1];
        const int32_t dst_start = dst_col_ptr[j];
        const int32_t dst_end = dst_col_ptr[j + 1];

        float col_sum = 0.0f;

        // First pass: cast+copy and accumulate sum
        for (uint32_t src_k = src_start, dst_k = dst_start; src_k < src_end; ++src_k, ++dst_k) {
            dst_row_idx[dst_k] = static_cast<int32_t>(src_row_idx[src_k]);
            float val = static_cast<float>(src_values[src_k]);
            dst_values[dst_k] = val;
            col_sum += val;
        }

        // Avoid division by zero; if sum is zero, leave values as-is
        if (col_sum > 0.0f) {
            float mult = scale / col_sum;

            // Second pass: apply log1p transformation in place
            for (int32_t dst_k = dst_start; dst_k < dst_end; ++dst_k) {
                dst_values[dst_k] = std::log1p(dst_values[dst_k] * mult);
            }
        }
    }
}

}  // namespace

// ============================================================================
// Batch implementation
// ============================================================================

Batch::Batch(Slot* slot,
             std::string species_name,
             const Chunk* chunk,
             std::vector<int> column_indices,
             bool chunk_end,
             float scale)
    : slot_(slot),
      species_name_(std::move(species_name)),
      chunk_(chunk),
      column_indices_(std::move(column_indices)),
      B_(static_cast<int>(column_indices_.size())),
      chunk_end_(chunk_end),
      scale_(scale) {
    
    if (!slot_) {
        throw std::runtime_error("Batch constructor: slot must not be null");
    }

    if (!chunk_) {
        throw std::runtime_error("Batch constructor: chunk must not be null");
    }

    if (slot_->state() != Slot::State::kEmpty) {
        throw std::runtime_error("Batch: slot already has an active batch");
    }

    m_ = chunk_->m;
    slot_->fill();
}

// Move constructor
Batch::Batch(Batch&& other) noexcept
    : slot_(other.slot_),
      species_name_(std::move(other.species_name_)),
      chunk_(other.chunk_),
      column_indices_(std::move(other.column_indices_)),
      m_(other.m_),
      B_(other.B_),
      nnz_(other.nnz_),
      chunk_end_(other.chunk_end_),
      scale_(other.scale_),
      consumer_stream_(other.consumer_stream_) {
    other.slot_ = nullptr;
    other.chunk_ = nullptr;
}

// Move assignment
Batch& Batch::operator=(Batch&& other) noexcept {
    if (this != &other) {
        // Release current slot (if any)
        if (slot_) {
            slot_->mark_empty(consumer_stream_);
        }
        // Transfer ownership
        slot_ = other.slot_;
        species_name_ = std::move(other.species_name_);
        chunk_ = other.chunk_;
        column_indices_ = std::move(other.column_indices_);
        m_ = other.m_;
        B_ = other.B_;
        nnz_ = other.nnz_;
        chunk_end_ = other.chunk_end_;
        scale_ = other.scale_;
        consumer_stream_ = other.consumer_stream_;
        other.slot_ = nullptr;
        other.chunk_ = nullptr;
    }
    return *this;
}

// Destructor
Batch::~Batch() {
    if (slot_) {
        slot_->mark_empty(consumer_stream_);
    }
}

// Builds this batch's own col_ptr (prefix sum) into slot's pinned col_ptr buffer.
// Returns total nnz.
int Batch::layout() {
    // Ensure slot has capacity for col_ptr (B_+1 entries) and row_idx/values sizes not yet known.
    slot_->grow(B_ + 1, 0);

    // Single pass: write prefix sum directly into slot's pinned col_ptr buffer.
    int32_t* dst_col_ptr = slot_->pinned_col_ptr();
    dst_col_ptr[0] = 0;
    for (int j = 0; j < B_; ++j) {
        const int col_idx = column_indices_[j];
        const uint32_t nnz_col = chunk_->col_ptr[col_idx + 1] - chunk_->col_ptr[col_idx];
        dst_col_ptr[j + 1] = dst_col_ptr[j] + static_cast<int32_t>(nnz_col);
    }
    const int total_nnz = dst_col_ptr[B_];

    // Now grow row_idx/values buffers to fit; col_ptr capacity already satisfied so it's untouched.
    // Note: new API takes only 2 args (col_cap, nnz_cap) since row_idx/values share nnz capacity.
    slot_->grow(B_ + 1, total_nnz);

    return total_nnz;
}

// Copies col_ptr, row_idx, values to device and records ready event.
void Batch::to_device() {
    // Note: An agent wanted to make the slot's stream wait for its own empty event
    // before copying, but the slot's buffers belong to this Batch as long as they
    // are tied together, so we assume the buffers are safe
    BATCH_CUDA_CHECK(cudaMemcpyAsync(
        slot_->device_col_ptr(),
        slot_->pinned_col_ptr(),
        sizeof(int32_t) * (B_ + 1),
        cudaMemcpyHostToDevice,
        slot_->stream()
    ));

    BATCH_CUDA_CHECK(cudaMemcpyAsync(
        slot_->device_row_idx(),
        slot_->pinned_row_idx(),
        sizeof(int32_t) * nnz_,
        cudaMemcpyHostToDevice,
        slot_->stream()
    ));

    BATCH_CUDA_CHECK(cudaMemcpyAsync(
        slot_->device_values(),
        slot_->pinned_values(),
        sizeof(float) * nnz_,
        cudaMemcpyHostToDevice,
        slot_->stream()
    ));

    slot_->mark_ready();
}

// Prepare: gather, normalize, and ship to device.
void Batch::prepare() {
    if (!slot_) {
        throw std::runtime_error("Batch::prepare: batch is not bound (may have been moved)");
    }

    if (!chunk_) {
        throw std::runtime_error("Batch::prepare: chunk is not bound");
    }

    nnz_ = layout();

    gather_normalize(chunk_->col_ptr.data(), chunk_->row_idx.data(), chunk_->values.data(),
                     column_indices_, slot_->pinned_col_ptr(), slot_->pinned_row_idx(),
                     slot_->pinned_values(), scale_);

    to_device();
}

// Accessors
const std::string& Batch::species_name() const {
    return species_name_;
}

bool Batch::chunk_end() const {
    return chunk_end_;
}

int Batch::m() const {
    return m_;
}

int Batch::B() const {
    return B_;
}

int Batch::nnz() const {
    return nnz_;
}

cudaEvent_t Batch::ready_event() const {
    if (!slot_) {
        throw std::runtime_error("Batch::ready_event: batch is not bound (may have been moved)");
    }
    return slot_->ready_event();
}

SparseView Batch::sparse_view() const {
    if (!slot_) {
        throw std::runtime_error("Batch::sparse_view: batch is not bound (may have been moved)");
    }

    return SparseView{
        m_,
        B_,
        nnz_,
        slot_->device_col_ptr(),
        slot_->device_row_idx(),
        slot_->device_values()
    };
}

void Batch::set_consumer_stream(cudaStream_t stream) {
    consumer_stream_ = stream;
}
