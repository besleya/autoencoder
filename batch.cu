// SPDX-License-Identifier: MIT
// batch.cu — Implementation of Batch class. Handles H2D transfer, log-norm, and event recording.

#include "batch.h"

#include <cuda_runtime.h>
#include <cstring>
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
// Log-normalization kernel (copied from gpu_data_loader.cu)
// ============================================================================

static constexpr float kBatchLogNormScaler = 10000.0f;

__global__ void batch_log_normalize_columns_kernel(int n_cols,
                                                    float scaler,
                                                    const int32_t* __restrict__ col_ptr,
                                                    float* __restrict__ values) {
    int col = blockIdx.x;
    if (col >= n_cols) return;

    int start = col_ptr[col];
    int end   = col_ptr[col + 1];

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();

    // Phase 1: column sum via block-shared atomicAdd
    float thread_sum = 0.0f;
    for (int k = start + threadIdx.x; k < end; k += blockDim.x) {
        thread_sum += values[k];
    }
    if (thread_sum != 0.0f) atomicAdd(&s_sum, thread_sum);
    __syncthreads();

    float sum = s_sum;
    if (sum > 0.0f) {
        float inv = scaler / sum;
        for (int k = start + threadIdx.x; k < end; k += blockDim.x) {
            values[k] = log1pf(values[k] * inv);
        }
    }
}

// ============================================================================
// Batch implementation
// ============================================================================

Batch::Batch(Slot* slot,
             std::string species_name,
             const int32_t* col_ptr,
             const int32_t* row_idx,
             const float* values,
             int m,
             int B,
             int nnz,
             bool chunk_end)
    : slot_(slot),
      species_name_(std::move(species_name)),
      m_(m),
      B_(B),
      nnz_(nnz),
      chunk_end_(chunk_end) {
    
    if (!slot_) {
        throw std::runtime_error("Batch constructor: slot must not be null");
    }

    // Ensure slot has capacity for this batch
    slot_->ensure_capacity(B + 1, nnz, nnz);

    // Copy CSC data from host into slot's pinned buffers
    std::memcpy(slot_->pinned_col_ptr(), col_ptr, sizeof(int32_t) * (B + 1));
    std::memcpy(slot_->pinned_row_idx(), row_idx, sizeof(int32_t) * nnz);
    std::memcpy(slot_->pinned_values(), values, sizeof(float) * nnz);
}

// Move constructor
Batch::Batch(Batch&& other) noexcept
    : slot_(other.slot_),
      species_name_(std::move(other.species_name_)),
      m_(other.m_),
      B_(other.B_),
      nnz_(other.nnz_),
      chunk_end_(other.chunk_end_) {
    other.slot_ = nullptr;
}

// Move assignment
Batch& Batch::operator=(Batch&& other) noexcept {
    if (this != &other) {
        // Release current slot (if any)
        if (slot_) {
            slot_->mark_free();
        }
        // Transfer ownership
        slot_ = other.slot_;
        species_name_ = std::move(other.species_name_);
        m_ = other.m_;
        B_ = other.B_;
        nnz_ = other.nnz_;
        chunk_end_ = other.chunk_end_;
        other.slot_ = nullptr;
    }
    return *this;
}

// Destructor
Batch::~Batch() {
    if (slot_) {
        slot_->mark_free();
    }
}

// Prepare: ship to GPU, log-normalize, record event, mark ready
void Batch::prepare() {
    if (!slot_) {
        throw std::runtime_error("Batch::prepare: batch is not bound (may have been moved)");
    }

    // Issue H2D transfers on the slot's stream
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

    // Launch log-normalize kernel on the slot's stream
    // One block per column; each block computes the sum and applies the transform
    batch_log_normalize_columns_kernel<<<B_, 256, 0, slot_->stream()>>>(
        B_,
        kBatchLogNormScaler,
        slot_->device_col_ptr(),
        slot_->device_values()
    );

    // Record the ready event on the slot's stream
    BATCH_CUDA_CHECK(cudaEventRecord(slot_->ready_event(), slot_->stream()));

    // Mark the slot READY (atomic state transition)
    slot_->mark_ready();
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

// Build and return a SparseBatch struct for downstream consumers
SparseBatch Batch::sparse_view() const {
    if (!slot_) {
        throw std::runtime_error("Batch::sparse_view: batch is not bound (may have been moved)");
    }

    return SparseBatch{
        m_,
        B_,
        nnz_,
        slot_->device_col_ptr(),
        slot_->device_row_idx(),
        slot_->device_values(),
        slot_->ready_event(),
        chunk_end_
    };
}
