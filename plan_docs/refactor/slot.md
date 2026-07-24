# Slot — Design & Implementation Spec (v2, supersedes v1 below in history)

> Status: **SPEC — approved, ready for implementation.** This supersedes the
> checked-in `slot.h`/`slot.cu` (old FREE/FILLING/READY, single event, six
> capacity fields) and the v1 draft of this document. `HANDOFF.md` is a hint
> only and may be stale. `batch.md` is not modified by this doc, but this doc
> does authorize small edits to `batch.h`/`batch.cu` (see "Batch changes").

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
- **filling**: slot has an assigned `Batch`. Reached **immediately** (CPU-
  instant) when the producer decides to fill this slot — even before any
  worker thread starts real work, and stays here for the whole gather/
  normalize/H2D-issue sequence.
- **full**: **not** reached until the batch's `cudaMemcpyAsync` calls have
  actually **completed** on the GPU. `ready_event` is signaled at exactly
  this point. `state()` reflects `kFull` once `ready_event` is observed
  signaled — i.e. `full` is a true completion signal, not just "copies were
  issued."

`empty` is reached again once the trainer has consumed the active `Batch` —
concretely, once the `Batch` is discarded. `empty_event` is issued
(recorded) at that point, on the consumer's stream.

So `filling → empty` transitions are CPU-instant, driven directly by producer/
consumer code. The `filling → full` transition is different: it is only ever
*observed* to have happened by polling/waiting on `ready_event`; nothing sets
it eagerly.

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
  - `cudaEvent_t ready_event_` — recorded on `stream_`, right after the last
    H2D `cudaMemcpyAsync` for the current batch. Signals `full`.
  - `cudaEvent_t empty_event_` — recorded on the **consumer-supplied**
    stream when the batch is discarded. Signals `empty` is safe to build on
    (device buffers no longer being read).

### Why two events, and why `full` needs a real completion signal

`empty_event_` protects the **device** buffer: a new H2D copy must not start
until the previous consumer's reads of that device memory (kernels enqueued
on the consumer's own stream) have actually finished. The producer enqueues
`cudaStreamWaitEvent(stream_, empty_event_)` before its own H2D copies so the
GPU — not the CPU — enforces the order; this costs nothing when the consumer
is already done and correctly stalls the producer stream (no CPU thread ever
blocks) when it isn't.

`ready_event_`/`full` protects the **host pinned** buffer: while a
`cudaMemcpyAsync` H2D copy is in flight, the pinned source buffer it reads
from must not be overwritten. If `full` became true the instant the copies
were *issued* (the old design), nothing would stop a subsequent producer
from starting to gather/normalize a new batch into the same pinned memory
before the in-flight copy had actually finished reading it. Because the
whole cycle is `filling → full → empty → filling`, and `full`/`empty` can
only be reached in that order, requiring `full` to mean "copy actually
complete" transitively guarantees that whenever a new `filling` phase begins
writing to the pinned buffer, the previous copy from that same buffer is
long since finished. This is why `full` must be a genuine completion signal
and not a CPU-issued flag.

This also fixes a latent bug in the currently checked-in `slot.cu`:
`ensure_capacity()` calls `cudaFree`/`cudaFreeHost` unconditionally when
growing, with no guarantee the GPU has finished touching the old buffers.
`cudaStreamWaitEvent` only orders *stream operations*, not host-issued
`cudaFree` calls, so `ensure_capacity()` must `cudaEventSynchronize()` on the
relevant event before freeing anything it isn't certain is idle (see
interface below).

## Interface

```cpp
class Slot {
public:
    enum class State { kEmpty, kFilling, kFull };

    // Allocates pinned + device buffers at the given capacities, creates the
    // stream and both events. empty_event_ is recorded once, up front, so
    // the very first fill never has to wait. Initial state: kEmpty.
    Slot(int col_cap, int nnz_cap);

    ~Slot();  // frees device + pinned buffers, destroys stream + both events

    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    Slot(Slot&&) = delete;
    Slot& operator=(Slot&&) = delete;

    // ---- State ----
    // Lazily upgrades kFilling -> kFull by polling ready_event_ (cheap,
    // non-blocking cudaEventQuery). Never blocks. kEmpty/kFull are read as-is.
    State state() const;

    // kEmpty -> kFilling. No assert (Batch already validated via throw —
    // see "Batch changes"); Slot itself does not enforce 1:1.
    void fill();

    // Records ready_event_ on stream_. Call this after issuing the last H2D
    // cudaMemcpyAsync for the batch. Does not touch state_ directly — state()
    // observes completion lazily (see above).
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
    // ones; no-op otherwise. Only valid while state() != kEmpty (i.e. during
    // an active fill). If growth is needed, synchronizes on empty_event_
    // before freeing old buffers (see hazard note above). Branches are
    // ordered/annotated so growth is the unlikely path (see "Optimization
    // notes" below) — col growth is expected almost never, nnz growth rarely.
    void grow(int col_cap, int nnz_cap);

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
    cudaStream_t stream() const;       // slot's own stream (producer side)
    cudaEvent_t  ready_event() const;
    cudaEvent_t  empty_event() const;

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
    cudaEvent_t  ready_event_ = nullptr;
    cudaEvent_t  empty_event_ = nullptr;
};
```

### Debug-only write guard (recommendation, not enforced in release)

Per the invariant "buffers are only written by the active batch, and only
during `filling`": rather than adding runtime cost in release builds,
recommend wrapping the four pinned/device *mutable* accessors' precondition
in an `assert(state() != State::kEmpty)` compiled only under `!defined(NDEBUG)`.
This catches misuse in debug/test builds (including the SLURM-run unit
tests) at negligible cost, without slowing down the release path. It cannot
catch every misuse (e.g. it won't distinguish `filling` from a stale `full`
that a caller forgot to re-check), but it catches the most common mistake
(writing to a slot nobody has called `fill()` on yet).

## Batch changes (small, within `batch.h`/`batch.cu`)

- `Batch`'s constructor must check `slot->state() == Slot::State::kEmpty`
  and `throw std::runtime_error(...)` (or a small dedicated exception type)
  if not — this replaces relying on `Slot` to assert. `Slot` does not
  enforce 1:1 binding itself.
- After the throw-check passes, `Batch`'s constructor calls `slot->fill()`
  (renamed from `mark_filling()`).
- `Batch::to_device()`: before issuing the 3 `cudaMemcpyAsync` calls, add
  `slot->await_empty(slot->stream())` (GPU-side, non-blocking) so the H2D
  copies are ordered after the previous consumer's reads. After issuing the
  3 copies, call `slot->mark_ready()` (renamed from `mark_full()`) instead
  of recording the event itself.
- `Batch`'s destructor: call `slot->mark_empty(<consumer stream>)` instead
  of `mark_free()`. The consumer stream is whatever stream the trainer reads
  device buffers on — needs to be threaded into `Batch` (constructor
  parameter or setter) if not already available; check current `Batch`
  fields for a trainer/consumer stream before adding a new one.
- These are the only changes needed; the rest of `Batch` (layout, gather,
  normalize, move semantics) is unaffected.

## Usage protocol

Producer side (`Batch`):

1. `Batch` ctor: check `slot->state() == kEmpty`, throw if not, else call
   `slot->fill()`. CPU-instant; slot may sit in `kFilling` indefinitely
   before a worker thread is free to do the real work.
2. Worker thread does CPU gather/log-norm into pinned buffers, calling
   `slot->grow(...)` first if the batch exceeds current capacity.
3. `to_device()`: `slot->await_empty(slot->stream())`, then issue the 3
   `cudaMemcpyAsync` H2D copies on `slot->stream()`, then `slot->mark_ready()`.

Consumer side (trainer/Ring):

4. Poll/block on `slot->state() == kFull` (this is a true completion signal
   — no separate event wait is required just to trust it), or call
   `slot->await_full()`/`slot->await_full(stream)` directly if blocking or
   ordering is preferred over polling.
5. Read device buffers via kernels on the consumer's own stream.
6. When done issuing all reads (not necessarily their completion), discard
   the `Batch`; its destructor calls `slot->mark_empty(consumer_stream)`.

## Invariants

- State transitions occur only in the order `kEmpty → kFilling → kFull →
  kEmpty`. `kFilling → kFull` is observed (via `ready_event_`), never
  set directly; `kFull → kEmpty`/`kEmpty → kFilling` are CPU-instant.
- Buffers are only written while `state() != kEmpty` (ideally exactly
  `kFilling`) — see debug-only write guard above.
- Device/pinned buffers are never freed while a GPU operation might still be
  reading/writing them — `grow()` synchronizes on `empty_event_` before any
  `cudaFree`/`cudaFreeHost`.
- `ready_event_` is only ever recorded on `stream_`, only from
  `mark_ready()`.
- `empty_event_` is only ever recorded on the caller-supplied stream, only
  from `mark_empty()`.
- `grow()` may only be called while `state() != kEmpty`.
- `Slot` does not enforce single-ownership of a `Batch`; `Batch` enforces it
  by throwing in its constructor.

## Optimization notes (`grow()`)

Given: column-capacity growth is expected **almost never**; nnz-capacity
growth is expected **seldom** (both after the first few batches settle into
steady state). `grow()` should be written so the compiler's branch layout
favors the no-growth path:

- Use `if (col_cap > col_capacity_) [[unlikely]] { ... }` and
  `if (nnz_cap > nnz_capacity_) [[unlikely]] { ... }` (C++20 attribute; this
  repo builds with `-std=c++17` via nvcc/g++, so use
  `__builtin_expect(cond, 0)` instead, wrapped in a small local helper, or
  confirm `[[unlikely]]` is accepted by the pinned nvcc version before using
  it — flag this as a build-verification step).
- Keep the common (no-growth) path as a couple of cheap integer comparisons
  and nothing else — no function calls, no loops — so it inlines trivially.
- Isolate the actual realloc logic (event sync + free + alloc, for both the
  col arrays and the nnz-sized arrays) into two small helper functions
  (`grow_col()`, `grow_nnz()`) called only from the unlikely branches, per
  the "functions should do one thing" rule — keeps `grow()` itself a short
  dispatcher.

## Q&A

**Why one stream per slot (vs. one per DataLoader, vs. a single shared
transfer stream)?**

- **Per-slot stream (chosen)**: lets N slots' H2D copies (and any future
  per-slot GPU prep work) issue and progress concurrently/out-of-order with
  respect to each other — the GPU scheduler can overlap them subject to
  actual hardware queue/copy-engine limits. This maximizes overlap between
  "slot A's copy is in flight" and "slot B is layout()-ing on CPU and about
  to issue its own copy." Cost: one stream + two events per slot (cheap;
  streams/events are lightweight CUDA objects), and `cudaStreamWaitEvent`
  cross-stream bookkeeping (also cheap).
- **Per-DataLoader stream**: would serialize all of a species' slots onto
  one stream — slot B's copy could not start until slot A's prior work
  issued on that same stream is done being *enqueued* in order, reducing
  the achievable overlap between a loader's own slots, defeating a chunk of
  the reason to have multiple prefetch slots per loader in the first place.
- **Single global transfer stream**: serializes *all* species' transfers
  through one queue — simplest, lowest resource usage, but removes overlap
  entirely and makes one slow prep (e.g. a big batch needing `grow()`) a
  head-of-line blocker for every other loader's H2D copy. Given H2D copies
  themselves are typically bandwidth-bound on a shared PCIe/NVLink path
  regardless of how many streams issue them, the main loss here vs.
  per-slot streams is really about CPU-side issue-order serialization, not
  GPU copy-engine parallelism — but per-slot is still preferred since it's
  cheap and removes a class of false CPU-side dependencies for zero cost.
- Net: per-slot is the right default given slots are few (single digits per
  loader) and streams/events are cheap; recommend keeping it.

**Unit test framework?**

No Catch2/GoogleTest in this repo; existing convention
(`kernel_bench/*/test_accuracy.cu`, `tests/validate/validate.cpp`) is
hand-rolled: plain functions, `CUDA_CHECK`-style macros, `fprintf(stderr, ...)`
+ `exit(1)` on failure. Recommend continuing that convention rather than
introducing a new dependency (nothing to `apt`/`pip` install on the cluster,
no build-system changes): a small `tests/slot/test_slot.cu` with one function
per test case, each returning `bool`/printing `[PASS]`/`[FAIL] <reason>` with
the test name, and a `main()` that runs them all, counts failures, and
returns nonzero if any failed — mirroring `test_accuracy.cu`'s pass/fail line
format so log-grepping tools already used in this repo (e.g. `run_tests.sh`,
the `grep` in `validate_race_fix.slurm`) keep working unmodified.

## Unit test plan

All tests require a CUDA device — run only via SLURM (never locally). Home:
`tests/slot/test_slot.cu`, driven by `tests/slot/run_tests.sh` (mirrors
`kernel_bench/*/run_tests.sh`), launched from a new **`validate_slot.sh`**
SLURM script (`.sh` extension per convention — modeled on
`validate_race_fix.slurm`'s structure, just renamed/adapted).

1. **Initial state** — construct a `Slot`; assert `state() == kEmpty`.
2. **Happy-path cycle** — `fill()` → issue dummy H2D copies → `mark_ready()`
   → poll `state()` until `kFull` (bounded retry loop) → `mark_empty(stream)`
   → back to `kEmpty`. Repeat for several cycles (catches stale-flag bugs).
3. **`full` requires real completion, not just issue** — issue a slow
   (~50ms) dummy `cudaMemcpyAsync` (e.g. large buffer) immediately followed
   by `mark_ready()`; assert `state()` is **not yet** `kFull` right after
   `mark_ready()` returns (still `kFilling`), then assert it becomes `kFull`
   only after the copy actually finishes (poll or `await_full()`).
4. **`grow()` no-op when sufficient** — capture buffer pointers, call
   `grow()` with capacities ≤ current, assert pointers unchanged.
5. **`grow()` grows when needed** — call with larger capacities, assert
   `col_capacity()`/`nnz_capacity()` updated and pointers changed.
6. **`grow()` only legal while non-empty** — assert/abort if called while
   `kEmpty`.
7. **End-to-end data integrity** — write known CSC data into pinned buffers
   during `filling`, run the full fill sequence, wait for `kFull`, copy
   device buffers back to host, compare against source data.
8. **Overwrite-hazard test — device buffer** — launch a slow reader kernel
   on a consumer stream that reads the slot's current device buffer into a
   canary location, call `mark_empty(consumer_stream)` immediately after
   launching it, then immediately start a new fill cycle including its H2D
   copies. Verify the canary shows the *old* data (i.e. the reader read
   before the new copy landed), exercising `await_empty()` inside
   `to_device()`.
9. **Overwrite-hazard test — host buffer** — the one explicitly requested:
   issue a slow H2D copy from the pinned buffer, call `mark_ready()`, then
   — while `state()` is still `kFilling` (not yet `kFull`) — attempt to
   write new data into the pinned buffer from a second simulated "producer"
   and confirm (in a debug build, via the write-guard assert) that this is
   caught, or, if choosing not to exercise the assert path, confirm via a
   canary value that the in-flight copy transferred the *original* data,
   not data written after `mark_ready()`. Should also verify that once
   `state() == kFull`, buffers are not written again until a full
   `empty → filling` cycle has occurred.
10. **`grow()` does not free while consumer still reading** — start a slow
    reader kernel on a consumer stream, `mark_empty()` right after launch,
    then trigger a fill cycle whose batch is bigger than current capacity
    (forcing `grow()` to realloc). Must not crash and must be clean under
    `compute-sanitizer` (run as part of the SLURM job).
11. **Constructor throw on double-fill (in `Batch`, not `Slot`)** — construct
    a `Batch` bound to a slot already in `kFilling`/`kFull`; assert it
    throws. This is a `Batch`-level test but belongs alongside the Slot
    suite since it validates the contract described in "Batch changes."
12. **Destructor safety** — destroy a `Slot` immediately after construction
    (never filled) and after a full cycle; both clean up without leaks
    (`compute-sanitizer --leak-check=full`).

## Changelog vs. previous `slot.h`/`slot.cu` and vs. v1 of this doc

| Old (`slot.h`/`slot.cu`) | v1 of this doc | v2 (this version) |
|---|---|---|
| States `FREE/FILLING/READY` | `kEmpty/kFilling/kFull`, CPU-instant | `kEmpty/kFilling/kFull`; `kFull` is a real GPU-completion signal, not CPU-instant |
| One event (`ready_event_`) | Two events (`filled_event_`, `empty_event_`) | Two events, renamed back to `ready_event_`/`empty_event_` |
| Six capacity ints | Two (`col_capacity_`, `nnz_capacity_`) | Same |
| No device-overwrite protection | `mark_filling()` enqueues the wait | `await_empty(stream)` called explicitly by `Batch::to_device()`, not hidden inside `fill()` |
| No host-buffer-overwrite protection | Not addressed | Addressed by making `full` a true completion signal (see rationale above) |
| `ensure_capacity()` frees with no completion guarantee | Same fix (sync before free) | Same, renamed `grow()`, with `[[unlikely]]`-style branch hints |
| `mark_ready()`/`mark_free()` bare setters, `Batch` recorded events separately | `mark_full()`/`mark_empty()` bundle record + transition | `mark_ready()` only records (no state mutation — `full` is observed, not set); `mark_empty()` still bundles record + CPU-instant transition |
| Slot asserts on illegal `Batch` binding | Same | `Slot` does **not** enforce 1:1; `Batch` throws instead |

## Open questions — resolved

- Consumer always has its own dedicated stream, passed into `mark_empty()`/
  `await_empty(stream)`. **Confirmed.**
- `Batch` still owns the actual `cudaMemcpyAsync` calls. **Confirmed** — see
  "Batch changes" for the small additions needed on top of that.
- Single `nnz_capacity_` shared by `row_idx`/`values` is sufficient.
  **Confirmed.**

## Remaining open item

- `[[unlikely]]` is a C++20 attribute; this repo builds with `-std=c++17`.
  Need to verify during implementation whether the pinned `nvcc`/host
  compiler accepts it anyway (many compilers accept it as an extension pre-
  C++20) or whether to fall back to `__builtin_expect`. Flagging this now so
  Haiku doesn't have to guess — if it doesn't compile, fall back silently to
  `__builtin_expect(condition, 0)`.
