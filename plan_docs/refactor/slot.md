# Slot — Design & Implementation Spec (superseding all prior drafts)

> Status: **SPEC — not yet implemented.** This supersedes the previous
> FREE/FILLING/READY draft below it in history. Do not rely on `slot.h`/`slot.cu`
> as currently checked in — they implement the old, incomplete design (single
> event, six capacity fields, no overwrite-hazard protection). This document is
> the source of truth going forward. `HANDOFF.md` is a hint only and may be stale.

One prefetch buffer. Owns reusable pinned-host and device buffers sized to hold
a cuSPARSE CSC object (`col_ptr`, `row_idx`, `values`). Tracks its own state and
exposes it. Does **no** CPU math and issues **no** memcpy itself — `Batch` (or
whatever driver code fills/drains the slot) does that work using buffers and
CUDA resources borrowed from the `Slot`. This mirrors the ownership split
already established for `Batch` (`Batch` owns no memory; `Slot` owns
everything) — see [batch.md](batch.md).

## States

```
empty → filling → full → empty → filling → full → ...
```

- **empty**: no active batch. Free to be loaded with the next batch. Slots
  start here.
- **filling**: Slot has an assigned `Batch`. Reached **immediately** when the
  Ring/DataLoader issues a request to fill the slot — even if no worker thread
  is available yet to actually prepare the batch, the slot sits in `filling`
  for as long as it takes for a thread to pick up the work. This is a
  CPU-visible flag, not tied to any GPU completion.
- **full**: the batch has been completely prepared and its `cudaMemcpyAsync`
  calls to the GPU have all been **issued** (enqueued) on the slot's stream.
  This transition is also CPU-visible and does **not** imply the copies have
  *finished* on the GPU — only that the CPU-side prep work is done and the
  work is queued. The trainer must wait on `filled_event_` before it is safe
  to read device buffers.

Both transitions are driven by CPU-visible atomic state (matching the existing
"immediate FILLING" behavior already documented for the old design) — GPU
completion is tracked separately via the two events below.

## Owned resources

- Pinned host buffers: `h_col_ptr_`, `h_row_idx_`, `h_values_`.
- Device buffers: `d_col_ptr_`, `d_row_idx_`, `d_values_`.
- Two capacity attributes (not six — host and device capacity are always kept
  in lock-step, and `row_idx`/`values` always share one nnz-sized capacity):
  - `col_capacity_` — capacity of `col_ptr` (needs `B + 1` entries).
  - `nnz_capacity_` — shared capacity of `row_idx` and `values` (needs `nnz`
    entries each).
- `cudaStream_t stream_` — the slot's own dedicated stream, used for all
  H2D copies issued while filling. Owning one stream per slot is what lets
  multiple slots' prep work overlap on the GPU.
- Two events, each reused across every fill/drain cycle of this slot:
  - `cudaEvent_t filled_event_` — recorded on `stream_` at the moment the slot
    transitions `filling → full`. The trainer waits on this before reading
    device buffers.
  - `cudaEvent_t empty_event_` — recorded on the **consumer's** stream at the
    moment the slot transitions `full → empty`. The next producer waits on
    this before overwriting device buffers, and `ensure_capacity()` waits on
    it before freeing them.

### Why `empty_event_` exists (bug fixed vs. the old design)

The old single-event design let a new `Batch` immediately start overwriting a
slot's device buffers as soon as it was marked `FILLING`, with no guarantee
that a **previous** trainer read of those same buffers (e.g. a kernel still
running on the trainer's own stream) had finished. `empty_event_` closes that
race: it is recorded, by the consumer, on the consumer's stream when it is
done with the batch, and the producer's stream is told to wait on it
(`cudaStreamWaitEvent`) before any H2D copy touches the buffers again. Because
`cudaStreamWaitEvent` is a cheap async enqueue, this costs nothing when the
consumer is already done, and correctly stalls the producer stream (without
blocking any CPU thread) when it isn't.

This also fixes a latent bug in the current `slot.cu`: `ensure_capacity()`
calls `cudaFree`/`cudaFreeHost` unconditionally when growing, with nothing
guaranteeing the GPU has actually finished reading the old device buffer.
`cudaStreamWaitEvent` only orders *stream operations*, not host-issued
`cudaFree` calls, so the new spec requires `ensure_capacity()` to
`cudaEventSynchronize(empty_event_)` before freeing anything (see below).

## Interface

```cpp
class Slot {
public:
    enum class State { kEmpty, kFilling, kFull };

    // Allocates pinned + device buffers at the given capacities, creates the
    // stream and both events (empty_event_ starts in an "already recorded/
    // completed" state so the first fill never blocks). Initial state: kEmpty.
    Slot(int col_capacity, int nnz_capacity);

    ~Slot();  // frees device + pinned buffers, destroys stream + both events

    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    Slot(Slot&&) = delete;
    Slot& operator=(Slot&&) = delete;

    // ---- State ----
    State state() const;  // atomic load, memory_order_acquire

    // empty -> filling. Asserts prior state == kEmpty.
    // Also enqueues cudaStreamWaitEvent(stream_, empty_event_) on stream_ so
    // that any producer work queued afterwards (memcpys) cannot start on the
    // GPU until the previous consumer's reads have completed. Safe/cheap to
    // call even when nothing needs waiting for.
    void mark_filling();

    // filling -> full. Asserts prior state == kFilling.
    // Records filled_event_ on stream_ (this must be called AFTER all H2D
    // cudaMemcpyAsync calls for this batch have been issued on stream_, so
    // that filled_event_ only completes once they have).
    void mark_full();

    // full -> empty. Asserts prior state == kFull.
    // Records empty_event_ on the CALLER-SUPPLIED consumer_stream (the
    // trainer's own stream) — this must be called AFTER the consumer has
    // issued all its reads of the device buffers on that stream.
    void mark_empty(cudaStream_t consumer_stream);

    // ---- Buffer management ----
    // Grows pinned+device buffers if requested capacities exceed current
    // ones; no-op otherwise. Only valid to call while state() == kFilling
    // (asserts). If it needs to grow, synchronizes on empty_event_ before
    // freeing old buffers (see hazard note above).
    void ensure_capacity(int new_col_capacity, int new_nnz_capacity);

    int col_capacity() const;
    int nnz_capacity() const;

    // ---- Buffer accessors ----
    int32_t* pinned_col_ptr() const;
    int32_t* pinned_row_idx() const;
    float*   pinned_values()  const;
    int32_t* device_col_ptr() const;
    int32_t* device_row_idx() const;
    float*   device_values()  const;

    // ---- CUDA resource accessors ----
    cudaStream_t stream() const;        // slot's own stream (producer side)
    cudaEvent_t  filled_event() const;  // wait on this before reading device buffers
    cudaEvent_t  empty_event() const;   // recorded by consumer; do not record elsewhere

private:
    std::atomic<State> state_;

    int32_t* h_col_ptr_ = nullptr;
    int32_t* h_row_idx_ = nullptr;
    float*   h_values_  = nullptr;
    int32_t* d_col_ptr_ = nullptr;
    int32_t* d_row_idx_ = nullptr;
    float*   d_values_  = nullptr;

    int col_capacity_ = 0;
    int nnz_capacity_ = 0;

    cudaStream_t stream_ = nullptr;
    cudaEvent_t  filled_event_ = nullptr;
    cudaEvent_t  empty_event_  = nullptr;
};
```

## Usage protocol (who calls what, and when)

Producer side (whatever code fills the slot — currently `Batch`, see
[batch.md](batch.md)):

1. Ring/DataLoader decides to fill this slot → calls `slot->mark_filling()`.
   This is CPU-instant; the slot may sit in `kFilling` indefinitely before a
   worker thread is free.
2. Worker thread does CPU gather/log-norm into pinned buffers, calling
   `slot->ensure_capacity(...)` first if the batch is larger than current
   capacity.
3. Worker thread issues the 3 `cudaMemcpyAsync` H2D copies on `slot->stream()`.
4. Worker thread calls `slot->mark_full()`. (This records `filled_event_` on
   `stream_` internally — callers never call `cudaEventRecord` directly.)

Consumer side (trainer / Ring):

5. Trainer checks `slot->state() == kFull` (CPU-visible) before trusting the
   event at all — mirrors the "check FILLING flag before trusting a possibly
   stale event" rule from the original design notes.
6. Trainer does `cudaStreamWaitEvent(trainer_stream, slot->filled_event())` (or
   `cudaEventSynchronize` if it needs a CPU-side block) before touching device
   buffers, then reads them (kernels on `trainer_stream`).
7. When done issuing all its reads, trainer calls
   `slot->mark_empty(trainer_stream)`. (This records `empty_event_` on
   `trainer_stream` internally.)

This makes the full cycle symmetric: two CPU-instant state flags
(`mark_filling`/`mark_full`/`mark_empty` all transition state immediately) and
two GPU-side completion events (`filled_event_`, `empty_event_`) that gate the
*next* stage's async GPU work without ever blocking a CPU thread in the common
case.

## Invariants

- State transitions are strictly ordered: `kEmpty → kFilling → kFull → kEmpty`.
  Any other transition asserts.
- Only one `Batch` (or equivalent driver) is bound to a slot at a time.
- Device/pinned buffers are never freed while a GPU operation might still be
  reading/writing them — enforced by synchronizing on `empty_event_` inside
  `ensure_capacity()` before any `cudaFree`/`cudaFreeHost`.
- `filled_event_` is only ever recorded on `stream_`, only from
  `mark_full()`.
- `empty_event_` is only ever recorded on the consumer-supplied stream, only
  from `mark_empty()`.
- `ensure_capacity()` may only be called while `state() == kFilling`.

## Changelog vs. previous `slot.h`/`slot.cu` (currently checked in)

| Old | New |
|---|---|
| States `FREE/FILLING/READY` | States `kEmpty/kFilling/kFull` |
| One event (`ready_event_`) | Two events (`filled_event_`, `empty_event_`) |
| Six capacity ints (`h_col_capacity_`, `h_row_capacity_`, `h_val_capacity_`, `d_col_capacity_`, `d_row_capacity_`, `d_val_capacity_`) | Two (`col_capacity_`, `nnz_capacity_`) |
| No protection against overwriting device buffers still being read by a lagging consumer | `mark_filling()` enqueues `cudaStreamWaitEvent` on `empty_event_` |
| `ensure_capacity()` frees buffers with no GPU-completion guarantee (latent bug) | `ensure_capacity()` syncs on `empty_event_` before freeing |
| Constructor took `(m, batch_size, max_nnz_estimate)` | Constructor takes `(col_capacity, nnz_capacity)` — caller computes `batch_size + 1` |
| `mark_ready()`/`mark_free()` were bare state setters; `Batch` called `cudaEventRecord` separately | `mark_full()`/`mark_empty()` do the state transition **and** the event record together, removing a foot-gun (impossible to mark full/empty without recording) |

## Unit test plan

All tests require a CUDA device, so they run only via SLURM on the GPU
cluster (never locally). Proposed home: `tests/slot/test_slot.cu`, built and
run by a new `validate_slot.slurm` (modeled on the existing
`validate_race_fix.slurm`). Tests to implement:

1. **Initial state** — construct a `Slot`; assert `state() == kEmpty`.
2. **Happy-path cycle** — `mark_filling()` → `mark_full()` →
   `mark_empty(stream)` → back to `kEmpty`; assert state after each call;
   repeat for several cycles (catches any accidental one-shot/stale-flag bug).
3. **Illegal transitions assert/abort** — e.g. calling `mark_full()` from
   `kEmpty`, or `mark_empty()` from `kFilling`. Run as a death test (expect
   process abort via `assert`), one case per illegal edge.
4. **`ensure_capacity` no-op when sufficient** — capture buffer pointers,
   call `ensure_capacity()` with capacities ≤ current, assert pointers
   unchanged.
5. **`ensure_capacity` grows when needed** — call with larger capacities,
   assert `col_capacity()`/`nnz_capacity()` updated and pointers changed
   (new allocation), and that previously written data is not required to
   survive (grow is destructive, batch will re-`layout()`).
6. **`ensure_capacity` only legal while filling** — assert/abort if called
   while `kEmpty` or `kFull`.
7. **End-to-end data integrity** — write known CSC data into pinned buffers,
   run through the full fill sequence (`mark_filling` → memcpy H2D →
   `mark_full`), wait on `filled_event()`, copy device buffers back to host,
   compare against the source data.
8. **Overwrite-hazard regression test** — the test that matters most for the
   new design: launch a long-running dummy kernel on a *consumer* stream that
   reads the slot's current device buffer (busy-loop kernel, e.g. spins
   ~50ms), call `mark_empty(consumer_stream)` immediately after launching it
   (event recorded right after the kernel, so it won't complete until the
   kernel does), then immediately start a new fill cycle
   (`mark_filling()` → H2D copies on `stream_` → `mark_full()`). Verify via a
   value written by the "reader" kernel (e.g. into a small canary device
   buffer) that the reader kernel's read happened *before* the new H2D copy
   landed — e.g. by having the reader kernel record what it read and
   comparing against the pre-overwrite value. This directly exercises the
   `cudaStreamWaitEvent(stream_, empty_event_)` protection added in
   `mark_filling()`.
9. **`ensure_capacity` does not free while consumer still reading** —
   companion to #8: start a slow reader kernel on a consumer stream, call
   `mark_empty()` right after launch, then trigger a fill cycle whose batch
   is bigger than current capacity (forcing `ensure_capacity` to grow, i.e.
   free+realloc). Should not crash / should not trip
   `compute-sanitizer`/`cuda-memcheck` use-after-free — verified by running
   the test binary under `compute-sanitizer` in the SLURM job.
10. **Destructor safety** — destroy a `Slot` immediately after construction
    (never filled) and after a full cycle; both should clean up without
    leaks (verified via `compute-sanitizer --leak-check=full` in the SLURM
    job).

## Open questions / assumptions made (flag if wrong)

- Assumed the consumer (trainer/Ring) always has its own dedicated stream to
  pass into `mark_empty()`. If the trainer instead uses the default stream or
  no stream at all, `mark_empty()`'s signature/semantics need revisiting.
- Assumed `Batch` (not `Slot`) still owns the actual `cudaMemcpyAsync` calls
  and `layout()`/`gather_normalize()` CPU work, per `batch.md`'s existing
  ownership split. `Slot` only exposes buffers/stream/events and manages
  state + capacity.
- Assumed row_idx and values always grow together (single `nnz_capacity_`)
  since they are always the same length (`nnz`) — confirmed by both
  `batch.cu`'s `layout()` (writes both at same indices) and the CSC format
  itself.
