// SPDX-License-Identifier: MIT
// slot.h — One prefetch buffer slot. Owns GPU/pinned memory and manages state.

#pragma once

#include <cuda_runtime.h>
#include <atomic>
#include <cassert>
#include <cstdint>

class Slot {
public:
    enum class State { kEmpty, kFilling, kFull };

    // Allocates pinned + device buffers at the given capacities, creates the
    // stream and both events. empty_event_ is recorded once, up front, so
    // the very first fill never has to wait. Initial state: kEmpty.
    Slot(int col_cap, int nnz_cap);

    ~Slot();

    // Non-copyable, non-movable.
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    Slot(Slot&&) = delete;
    Slot& operator=(Slot&&) = delete;

    // ---- State ----
    // Lazily upgrades kFilling -> kFull by polling ready_event_ (cheap,
    // non-blocking cudaEventQuery). Never blocks. kEmpty/kFull are read as-is.
    State state() const;

    // kEmpty -> kFilling. No assert (Batch already validated via throw).
    void fill();

    // Records ready_event_ on stream_. Call this after issuing the last H2D
    // cudaMemcpyAsync for the batch. Does not touch state_ directly — state()
    // observes completion lazily.
    void mark_ready();

    // Records empty_event_ on the caller-supplied consumer stream, then sets
    // state_ = kEmpty immediately (CPU-instant). Call this when discarding
    // the batch, after issuing (not necessarily completing) all reads of
    // device buffers on consumer_stream.
    void mark_empty(cudaStream_t consumer_stream);

    // ---- Waiting ----
    // CPU-blocking: cudaEventSynchronize. GPU-ordering: cudaStreamWaitEvent
    // (enqueues a wait on the given stream; never blocks the calling thread).
    void await_full();
    void await_full(cudaStream_t stream);
    void await_empty();
    void await_empty(cudaStream_t stream);

    // ---- Buffer management ----
    // Grows pinned+device buffers if requested capacities exceed current
    // ones; no-op otherwise. Only valid while state() != kEmpty.
    void grow(int col_cap, int nnz_cap);

    int col_capacity() const;
    int nnz_capacity() const;

    // ---- Buffer accessors ----
    // Each includes assert(state() != State::kEmpty) in debug builds only.
    int32_t* pinned_col_ptr() const;
    int32_t* pinned_row_idx() const;
    float*   pinned_values()  const;
    int32_t* device_col_ptr() const;
    int32_t* device_row_idx() const;
    float*   device_values()  const;

    // ---- CUDA resource accessors ----
    cudaStream_t stream() const;
    cudaEvent_t  ready_event() const;
    cudaEvent_t  empty_event() const;

private:
    // Helper functions for growth (called only when growth is needed).
    void grow_col(int col_cap);
    void grow_nnz(int nnz_cap);

    // mutable: state() is logically const but lazily upgrades kFilling -> kFull.
    mutable std::atomic<State> state_;

    int32_t* h_col_ptr_ = nullptr;
    int32_t* h_row_idx_ = nullptr;
    float*   h_values_  = nullptr;
    int32_t* d_col_ptr_ = nullptr;
    int32_t* d_row_idx_ = nullptr;
    float*   d_values_  = nullptr;

    int col_capacity_ = 0;
    int nnz_capacity_ = 0;

    cudaStream_t stream_ = nullptr;
    cudaEvent_t  ready_event_ = nullptr;
    cudaEvent_t  empty_event_ = nullptr;
};
