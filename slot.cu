// SPDX-License-Identifier: MIT
// slot.cu — implementation of Slot (one prefetch buffer).

#include "slot.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

// ============================================================================
// CUDA error checking
// ============================================================================

#define SLOT_CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::ostringstream oss; \
        oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
            << cudaGetErrorString(err); \
        throw std::runtime_error(oss.str()); \
    } \
} while (0)

// ============================================================================
// Constructor
// ============================================================================

Slot::Slot(int m, int batch_size, int max_nnz_estimate)
    : m_(m), state_(State::FREE) {
    
    // Initial capacities
    int col_capacity = batch_size + 1;
    int row_capacity = max_nnz_estimate;
    int val_capacity = max_nnz_estimate;

    // Allocate pinned (host) buffers
    SLOT_CUDA_CHECK(cudaMallocHost(&h_col_ptr_, col_capacity * sizeof(int32_t)));
    SLOT_CUDA_CHECK(cudaMallocHost(&h_row_idx_, row_capacity * sizeof(int32_t)));
    SLOT_CUDA_CHECK(cudaMallocHost(&h_values_, val_capacity * sizeof(float)));

    h_col_capacity_ = col_capacity;
    h_row_capacity_ = row_capacity;
    h_val_capacity_ = val_capacity;

    // Allocate device buffers
    SLOT_CUDA_CHECK(cudaMalloc(&d_col_ptr_, col_capacity * sizeof(int32_t)));
    SLOT_CUDA_CHECK(cudaMalloc(&d_row_idx_, row_capacity * sizeof(int32_t)));
    SLOT_CUDA_CHECK(cudaMalloc(&d_values_, val_capacity * sizeof(float)));

    d_col_capacity_ = col_capacity;
    d_row_capacity_ = row_capacity;
    d_val_capacity_ = val_capacity;

    // Create stream
    SLOT_CUDA_CHECK(cudaStreamCreate(&stream_));

    // Create event with timing disabled
    SLOT_CUDA_CHECK(cudaEventCreateWithFlags(&ready_event_, cudaEventDisableTiming));
}

// ============================================================================
// Destructor
// ============================================================================

Slot::~Slot() {
    // Destroy CUDA resources in reverse order of creation.
    // Tolerate null pointers (already freed).

    if (ready_event_) {
        cudaEventDestroy(ready_event_);
        ready_event_ = nullptr;
    }

    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }

    // Free device buffers
    if (d_col_ptr_) {
        cudaFree(d_col_ptr_);
        d_col_ptr_ = nullptr;
    }
    if (d_row_idx_) {
        cudaFree(d_row_idx_);
        d_row_idx_ = nullptr;
    }
    if (d_values_) {
        cudaFree(d_values_);
        d_values_ = nullptr;
    }

    // Free pinned (host) buffers
    if (h_col_ptr_) {
        cudaFreeHost(h_col_ptr_);
        h_col_ptr_ = nullptr;
    }
    if (h_row_idx_) {
        cudaFreeHost(h_row_idx_);
        h_row_idx_ = nullptr;
    }
    if (h_values_) {
        cudaFreeHost(h_values_);
        h_values_ = nullptr;
    }
}

// ============================================================================
// State management
// ============================================================================

Slot::State Slot::state() const {
    return state_.load(std::memory_order_acquire);
}

void Slot::mark_filling() {
    State prior = state_.exchange(State::FILLING, std::memory_order_acq_rel);
    assert(prior == State::FREE && "mark_filling: prior state must be FREE");
}

void Slot::mark_ready() {
    State prior = state_.exchange(State::READY, std::memory_order_acq_rel);
    assert(prior == State::FILLING && "mark_ready: prior state must be FILLING");
}

void Slot::mark_free() {
    State prior = state_.exchange(State::FREE, std::memory_order_acq_rel);
    assert(prior == State::READY && "mark_free: prior state must be READY");
}

// ============================================================================
// Buffer management
// ============================================================================

void Slot::ensure_capacity(int cap_col, int cap_row, int cap_val) {
    // Reallocate col_ptr if needed
    if (cap_col > h_col_capacity_) {
        if (h_col_ptr_) SLOT_CUDA_CHECK(cudaFreeHost(h_col_ptr_));
        SLOT_CUDA_CHECK(cudaMallocHost(&h_col_ptr_, cap_col * sizeof(int32_t)));
        h_col_capacity_ = cap_col;

        if (d_col_ptr_) SLOT_CUDA_CHECK(cudaFree(d_col_ptr_));
        SLOT_CUDA_CHECK(cudaMalloc(&d_col_ptr_, cap_col * sizeof(int32_t)));
        d_col_capacity_ = cap_col;
    }

    // Reallocate row_idx if needed
    if (cap_row > h_row_capacity_) {
        if (h_row_idx_) SLOT_CUDA_CHECK(cudaFreeHost(h_row_idx_));
        SLOT_CUDA_CHECK(cudaMallocHost(&h_row_idx_, cap_row * sizeof(int32_t)));
        h_row_capacity_ = cap_row;

        if (d_row_idx_) SLOT_CUDA_CHECK(cudaFree(d_row_idx_));
        SLOT_CUDA_CHECK(cudaMalloc(&d_row_idx_, cap_row * sizeof(int32_t)));
        d_row_capacity_ = cap_row;
    }

    // Reallocate values if needed
    if (cap_val > h_val_capacity_) {
        if (h_values_) SLOT_CUDA_CHECK(cudaFreeHost(h_values_));
        SLOT_CUDA_CHECK(cudaMallocHost(&h_values_, cap_val * sizeof(float)));
        h_val_capacity_ = cap_val;

        if (d_values_) SLOT_CUDA_CHECK(cudaFree(d_values_));
        SLOT_CUDA_CHECK(cudaMalloc(&d_values_, cap_val * sizeof(float)));
        d_val_capacity_ = cap_val;
    }
}

// ============================================================================
// Buffer accessors
// ============================================================================

int32_t* Slot::pinned_col_ptr() const {
    return h_col_ptr_;
}

int32_t* Slot::pinned_row_idx() const {
    return h_row_idx_;
}

float* Slot::pinned_values() const {
    return h_values_;
}

int32_t* Slot::device_col_ptr() const {
    return d_col_ptr_;
}

int32_t* Slot::device_row_idx() const {
    return d_row_idx_;
}

float* Slot::device_values() const {
    return d_values_;
}

// ============================================================================
// CUDA resource accessors
// ============================================================================

cudaStream_t Slot::stream() const {
    return stream_;
}

cudaEvent_t Slot::ready_event() const {
    return ready_event_;
}
