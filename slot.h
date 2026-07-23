// SPDX-License-Identifier: MIT
// slot.h — One prefetch buffer slot. Owns GPU/pinned memory and manages state.

#pragma once

#include <cuda_runtime.h>
#include <atomic>
#include <cassert>
#include <cstdint>

class Slot {
public:
    enum class State { FREE, FILLING, READY };

    // Constructor: allocate pinned + device buffers and CUDA resources.
    // Initial capacities: col_ptr = batch_size+1, row_idx = max_nnz_estimate, values = max_nnz_estimate.
    Slot(int m, int batch_size, int max_nnz_estimate);

    // Destructor: free all resources.
    ~Slot();

    // Non-copyable, non-movable (stable owned resource).
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    Slot(Slot&&) = delete;
    Slot& operator=(Slot&&) = delete;

    // ========================================================================
    // State management
    // ========================================================================

    // Atomic load of current state.
    State state() const;

    // Transition FREE → FILLING. Asserts prior state == FREE.
    void mark_filling();

    // Transition FILLING → READY. Asserts prior state == FILLING.
    void mark_ready();

    // Transition READY → FREE. Asserts prior state == READY.
    void mark_free();

    // ========================================================================
    // Buffer management
    // ========================================================================

    // Ensure pinned + device buffers have capacity. No-op if already sufficient.
    // Reallocates both pinned and device if any buffer is too small.
    void ensure_capacity(int cap_col, int cap_row, int cap_val);

    // Pinned (host) buffer accessors.
    int32_t* pinned_col_ptr() const;
    int32_t* pinned_row_idx() const;
    float* pinned_values() const;

    // Device buffer accessors.
    int32_t* device_col_ptr() const;
    int32_t* device_row_idx() const;
    float* device_values() const;

    // ========================================================================
    // CUDA resource accessors
    // ========================================================================

    // Dedicated stream for this slot's H2D + log-norm work.
    cudaStream_t stream() const;

    // Ready event: recorded at end of Batch::prepare().
    cudaEvent_t ready_event() const;

    // ========================================================================
    // FUTURE: await_ready() blocking synchronization
    // ========================================================================
    //
    // DESIGN NOTE (not yet implemented):
    //
    // A future public method await_ready() will provide a trainer-friendly way to
    // block until this Slot's Batch is fully materialized and ready to consume.
    // HOWEVER, it CANNOT be naively implemented as a simple cudaEventSynchronize()
    // or cudaStreamWaitEvent() on ready_event_, because that event object is
    // REUSED across Batch cycles.
    //
    // The problem: when a new Batch is bound to this Slot (the Batch is newly
    // constructed/attached), the ready_event_ will still READ as "already
    // completed" from its PREVIOUS cycle. This is dangerous — the trainer might
    // incorrectly think the new Batch is ready when it is not. The event only
    // becomes valid again when cudaEventRecord() is called at the very end of
    // Batch::prepare() — after all the substantial CPU work: gathering columns,
    // type-casting, log-normalizing, and issuing the H2D cudaMemcpyAsync calls.
    //
    // The solution: when a new Batch is bound (at Batch construction/attachment,
    // before any CPU prep work starts), the Slot MUST be told explicitly that it
    // is now in a "FILLING" state via a CPU-visible flag — NOT by waiting on the
    // CUDA event. This flag can be set atomically or via condition variable.
    //
    // The future await_ready() will:
    //   1. Wait for that flag to leave "FILLING" state (i.e. once the Batch's
    //      prepare() has recorded the event)
    //   2. Only THEN call cudaEventSynchronize() on the event to ensure all
    //      H2D transfers and GPU work are complete.
    //
    // This prevents naive implementations that would incorrectly sync on a stale
    // event from a previous Batch cycle.

private:
    std::atomic<State> state_;

    // Pinned (host) buffer pointers and capacities.
    int32_t* h_col_ptr_ = nullptr;
    int32_t* h_row_idx_ = nullptr;
    float* h_values_ = nullptr;
    int h_col_capacity_ = 0;
    int h_row_capacity_ = 0;
    int h_val_capacity_ = 0;

    // Device buffer pointers and capacities.
    int32_t* d_col_ptr_ = nullptr;
    int32_t* d_row_idx_ = nullptr;
    float* d_values_ = nullptr;
    int d_col_capacity_ = 0;
    int d_row_capacity_ = 0;
    int d_val_capacity_ = 0;

    // CUDA resources.
    cudaStream_t stream_ = nullptr;
    cudaEvent_t ready_event_ = nullptr;

    int m_;  // Feature dimension (for reference).
};
