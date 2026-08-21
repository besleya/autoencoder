// SPDX-License-Identifier: MIT
// slot.cu — implementation of Slot (one prefetch buffer).

#include <sstream>
#include <stdexcept>

#include "slot.h"

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
// Helper: likely/unlikely for branch hints
// ============================================================================

// Wrap __builtin_expect for readability; growth is the unlikely path.
inline bool unlikely(bool cond) {
    return __builtin_expect(cond, 0);
}

// ============================================================================
// Constructor
// ============================================================================

Slot::Slot(int col_cap, int nnz_cap)
    : state_(State::kEmpty) {
    
    col_capacity_ = col_cap;
    nnz_capacity_ = nnz_cap;

    // Allocate pinned (host) buffers
    SLOT_CUDA_CHECK(cudaMallocHost(&h_col_ptr_, col_cap * sizeof(int32_t)));
    SLOT_CUDA_CHECK(cudaMallocHost(&h_row_idx_, nnz_cap * sizeof(int32_t)));
    SLOT_CUDA_CHECK(cudaMallocHost(&h_values_, nnz_cap * sizeof(float)));

    // Allocate device buffers
    SLOT_CUDA_CHECK(cudaMalloc(&d_col_ptr_, col_cap * sizeof(int32_t)));
    SLOT_CUDA_CHECK(cudaMalloc(&d_row_idx_, nnz_cap * sizeof(int32_t)));
    SLOT_CUDA_CHECK(cudaMalloc(&d_values_, nnz_cap * sizeof(float)));

    // Create stream
    SLOT_CUDA_CHECK(cudaStreamCreate(&stream_));

    // Create events with timing disabled
    SLOT_CUDA_CHECK(cudaEventCreateWithFlags(&ready_event_, cudaEventDisableTiming));
    SLOT_CUDA_CHECK(cudaEventCreateWithFlags(&empty_event_, cudaEventDisableTiming));

    // Record empty_event_ once up front so the very first fill never blocks.
    // This is a trivial synchronization point before any real work.
    SLOT_CUDA_CHECK(cudaEventRecord(empty_event_, stream_));
}

// ============================================================================
// Destructor
// ============================================================================

Slot::~Slot() {
    // Destroy CUDA resources in reverse order. Tolerate nulls.

    if (empty_event_) {
        cudaEventDestroy(empty_event_);
        empty_event_ = nullptr;
    }

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
    State curr = state_.load(std::memory_order_acquire);

    // Lazily upgrade kFilling -> kFull by polling ready_event_.
    if (curr == State::kFilling) {
        cudaError_t err = cudaEventQuery(ready_event_);
        if (err == cudaSuccess) {
            // Event is signaled, upgrade to kFull.
            state_.store(State::kFull, std::memory_order_release);
            return State::kFull;
        } else if (err == cudaErrorNotReady) {
            // Event not yet signaled, stay in kFilling.
            return State::kFilling;
        } else {
            // Some other CUDA error; just return current state and let caller handle it.
            return State::kFilling;
        }
    }

    return curr;
}

void Slot::fill() {
    // kEmpty -> kFilling. No assert (Batch validates before calling).
    state_.store(State::kFilling, std::memory_order_release);
}

void Slot::mark_ready() {
    // Record ready_event_ on stream_. Does NOT mutate state directly.
    SLOT_CUDA_CHECK(cudaEventRecord(ready_event_, stream_));
}

void Slot::mark_empty(cudaStream_t consumer_stream) {
    // Record empty_event_ on the consumer's stream, then set state_ = kEmpty.
    SLOT_CUDA_CHECK(cudaEventRecord(empty_event_, consumer_stream));
    state_.store(State::kEmpty, std::memory_order_release);
}

// ============================================================================
// Waiting
// ============================================================================

void Slot::await_full() {
    SLOT_CUDA_CHECK(cudaEventSynchronize(ready_event_));
}

void Slot::await_full(cudaStream_t stream) {
    SLOT_CUDA_CHECK(cudaStreamWaitEvent(stream, ready_event_, 0));
}

void Slot::await_empty() {
    SLOT_CUDA_CHECK(cudaEventSynchronize(empty_event_));
}

void Slot::await_empty(cudaStream_t stream) {
    SLOT_CUDA_CHECK(cudaStreamWaitEvent(stream, empty_event_, 0));
}

// ============================================================================
// Buffer management — growth helpers
// ============================================================================

void Slot::grow_col(int col_cap) {
    // Synchronize before freeing: ensure consumer is done reading old device buffers.
    SLOT_CUDA_CHECK(cudaEventSynchronize(empty_event_));

    // Free and reallocate pinned col_ptr.
    if (h_col_ptr_) {
        SLOT_CUDA_CHECK(cudaFreeHost(h_col_ptr_));
    }
    SLOT_CUDA_CHECK(cudaMallocHost(&h_col_ptr_, col_cap * sizeof(int32_t)));

    // Free and reallocate device col_ptr.
    if (d_col_ptr_) {
        SLOT_CUDA_CHECK(cudaFree(d_col_ptr_));
    }
    SLOT_CUDA_CHECK(cudaMalloc(&d_col_ptr_, col_cap * sizeof(int32_t)));

    col_capacity_ = col_cap;
}

void Slot::grow_nnz(int nnz_cap) {
    // Synchronize before freeing: ensure consumer is done reading old device buffers.
    SLOT_CUDA_CHECK(cudaEventSynchronize(empty_event_));

    // Free and reallocate pinned row_idx.
    if (h_row_idx_) {
        SLOT_CUDA_CHECK(cudaFreeHost(h_row_idx_));
    }
    SLOT_CUDA_CHECK(cudaMallocHost(&h_row_idx_, nnz_cap * sizeof(int32_t)));

    // Free and reallocate device row_idx.
    if (d_row_idx_) {
        SLOT_CUDA_CHECK(cudaFree(d_row_idx_));
    }
    SLOT_CUDA_CHECK(cudaMalloc(&d_row_idx_, nnz_cap * sizeof(int32_t)));

    // Free and reallocate pinned values.
    if (h_values_) {
        SLOT_CUDA_CHECK(cudaFreeHost(h_values_));
    }
    SLOT_CUDA_CHECK(cudaMallocHost(&h_values_, nnz_cap * sizeof(float)));

    // Free and reallocate device values.
    if (d_values_) {
        SLOT_CUDA_CHECK(cudaFree(d_values_));
    }
    SLOT_CUDA_CHECK(cudaMalloc(&d_values_, nnz_cap * sizeof(float)));

    nnz_capacity_ = nnz_cap;
}

void Slot::grow(int col_cap, int nnz_cap) {
    // Check col_ptr capacity; growth is unlikely.
    if (unlikely(col_cap > col_capacity_)) {
        grow_col(col_cap);
    }

    // Check nnz capacity (row_idx and values); growth is unlikely.
    if (unlikely(nnz_cap > nnz_capacity_)) {
        grow_nnz(nnz_cap);
    }
}

// ============================================================================
// Capacity accessors
// ============================================================================

int Slot::col_capacity() const {
    return col_capacity_;
}

int Slot::nnz_capacity() const {
    return nnz_capacity_;
}

// ============================================================================
// Buffer accessors (with debug-only write guard)
// ============================================================================

int32_t* Slot::pinned_col_ptr() const {
    assert(state() != State::kEmpty);
    return h_col_ptr_;
}

int32_t* Slot::pinned_row_idx() const {
    assert(state() != State::kEmpty);
    return h_row_idx_;
}

float* Slot::pinned_values() const {
    assert(state() != State::kEmpty);
    return h_values_;
}

int32_t* Slot::device_col_ptr() const {
    assert(state() != State::kEmpty);
    return d_col_ptr_;
}

int32_t* Slot::device_row_idx() const {
    assert(state() != State::kEmpty);
    return d_row_idx_;
}

float* Slot::device_values() const {
    assert(state() != State::kEmpty);
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

cudaEvent_t Slot::empty_event() const {
    return empty_event_;
}
