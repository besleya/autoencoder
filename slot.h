// SPDX-License-Identifier: MIT
// slot.h — One prefetch buffer slot. Owns GPU/pinned memory and manages state.

#pragma once

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <mutex>

#include <cuda_runtime.h>

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

    // Atomically kEmpty -> kFilling. Returns false if the slot was not kEmpty,
    // i.e. somebody has already reserved it. Exactly one caller can win, so a
    // slot can never be handed to two fill workers.
    bool try_reserve();

    // Non-promoting read of the state machine. state() polls ready_event_ and
    // will report kFull for a slot whose ready_event_ was never recorded;
    // raw_state() reports what was actually stored.
    State raw_state() const;

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

    // Wakes every thread blocked in await_full()/await_empty() on this slot so
    // shutdown can make progress. Those waits return without their condition
    // having been met.
    void abort_waits();

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

    // CPU-side handshake. A CUDA event answers "has the recorded work
    // finished?", which is not the same question as "is this slot mine now?":
    // an event that has not been recorded for the current fill reports success
    // immediately, so it cannot be used to establish ownership. The mutex and
    // condition variable below own that question; the events are only waited on
    // once the matching record is known to have happened.
    mutable std::mutex mtx_;
    mutable std::condition_variable cv_;
    std::atomic<bool> ready_recorded_{false};
    std::atomic<bool> aborted_{false};

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
