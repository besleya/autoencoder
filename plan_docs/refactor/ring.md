# Ring

Scheduler and thread-pool owner. Cycles DataLoaders, keeps prefetch queues warm, hands ready batches to the trainer. Runs its own dispatcher thread so `main()` can `ring.start()` and go on to build the trainer while prefetch is already warming up.

## Responsibilities

- Own **two** CPU thread pools (see "Two pools" below).
- Own (or hold refs to) the set of registered `DataLoader`s.
- Dispense batches: next species, next batch, in strict cyclic order. Never skip a lane; block and wait for it.
- Enforce back-pressure: never submit a prepare-task for a `DataLoader` unless there is a specific empty `Slot` reserved for it.
- Serve the trainer: `next_ready_batch()` returns the next completed `Batch` in scheduling order.

## Non-responsibilities

- Ring does not own slot buffers. `DataLoader` does.
- Ring does not know about layers, autoencoders, or loss.
- Ring does not perform log-norm, gather, or memcpy — that's inside `Batch::prepare()` / `DataLoader::fill()`, run on pool workers.
- Ring does not decode files itself — it only owns the pool that decode tasks run on.

## Two pools (design decision — read this)

A single "one task per batch" pool does **not** parallelize chunk decode: decode is one call inside `fill()`, so it occupies exactly one pool worker no matter how many files it reads. Since most fills are cheap (gather from an already-decoded chunk) but chunk boundaries are the real blocker (reading dozens of files), we split into two pools with very different jobs:

| Pool | Purpose | Size | Who submits to it |
|---|---|---|---|
| `fill_pool_` | Runs `DataLoader::fill(slot)` — the usual case is cheap (gather + H2D). | small, e.g. `2 × n_loaders` | Ring's dispatcher thread |
| `decode_pool_` | Runs one task **per file** when a chunk needs decoding. | large, e.g. `std::thread::hardware_concurrency()` | Any `DataLoader`, from inside a `fill_pool_` worker |

Both are `BS::thread_pool` (vendored single-header library, see below). Ring owns both and exposes the decode pool so `DataLoader`s can use it:

```cpp
class Ring {
 public:
  Ring(int n_fill_workers, int n_decode_workers = std::thread::hardware_concurrency());
  BS::thread_pool& decode_pool();   // pass this into each DataLoader's constructor
  ...
};
```

**Why this works well:** when a `fill_pool_` worker hits a chunk boundary, it submits one task per file to `decode_pool_` and blocks on the futures. A blocked thread uses ~0 CPU, so it's fine for `fill_pool_` to be small and for several loaders to be "stuck" waiting on decode at once — the real CPU-heavy work (parsing files) is concentrated on `decode_pool_`, sized to the whole machine, and shared fairly across every loader that needs it. You get maximum CPU utilization exactly when it matters (chunk loads) without any manual per-loader thread budgeting.

**Why `BS::thread_pool` over hand-rolled:** you approved vendoring it. It's a single header (no build system changes beyond an include path), gives `submit_task()` returning a `std::future`/`std::shared_future` for free, and removes the queue/mutex/condvar boilerplate the current hand-rolled pool has. Action item: vendor `BS_thread_pool.hpp` under e.g. `third_party/`, add its directory to the Makefile's include path. (Not done in this pass — no build happens locally; this is a note for whoever does the build on the GPU machine.)

## Registration

```cpp
void add_loader(DataLoader* loader);       // Ring stores the pointer; loader must outlive Ring
```

Order of registration = round-robin order. Register all loaders **before** `start()`.

## Scheduler thread

`ring.start()` launches one dedicated **dispatcher thread** (separate from both pools). Its loop:

```
for loader in loaders (round-robin, forever):
    slot = loader.reserve_free_slot_blocking()   # blocks until the loader's *next* slot is empty
    fill_pool_.submit_task([loader, slot] { loader.fill(slot); })
```

`loader.fill(slot)` is where all real work happens: chunk load (if needed, via `decode_pool_`), Batch construction, log-norm, H2D, event record. On completion the Slot transitions filling→ready. This runs on a `fill_pool_` worker, **not** the dispatcher thread — the dispatcher must stay free to keep reserving/submitting for other loaders.

**Strict rotation — never skip.** If the current loader's next slot isn't empty yet, the dispatcher **blocks** on it (see `reserve_free_slot_blocking()` in [data_loader.md](data_loader.md) — it's a single `Slot::await_empty()` call, no polling). It does not advance to the next loader. This is the opposite of the current Ring implementation, which skips empty lanes in round-robin mode — that behavior is a bug and must not be carried forward.

**Why block on the current loader rather than skip?** Simpler; and if one loader is falling behind, throttling the others prevents unbounded queue growth.

## Trainer interface

```cpp
std::unique_ptr<Batch> next_ready_batch();   // blocks until the next-in-cycle loader has a ready batch
```

Selection policy for **which ready batch** to hand to the trainer: round-robin, in the *same order* the dispatcher submitted. Ring keeps a `next_consume_idx_`. `next_ready_batch()`:
1. Calls `loaders_[next_consume_idx_].take_ready_batch()` — blocks until that loader's *next* slot (in its own fixed rotation) is full.
2. Advances `next_consume_idx_` (round-robin).

Like the dispatcher, the consumer never skips. If the next-in-cycle loader has no ready batch yet, `next_ready_batch()` blocks on it — even if a later loader has one ready.

Consumer signals the slot free via `Batch`'s destructor (see [batch.md](batch.md)), which calls `slot->mark_empty(consumer_stream)`.

> **Bug to NOT reproduce.** The current Ring's round-robin mode *skips* a lane if it has no ready slot. That is a bug. The new `Ring::next_ready_batch()` must **block** on `loaders_[next_consume_idx_]` until that specific loader has a ready batch. Never advance the cursor past a species that isn't ready.

## ⚠️ Single-writer rule for the round-robin indices

`next_dispatch_idx_` inside the dispatcher loop and `next_consume_idx_` in `next_ready_batch()` are each touched by exactly **one** thread — the dispatcher thread, and whichever thread calls `next_ready_batch()` (normally the single trainer thread) respectively. Because each index has exactly one writer, they need **no mutex or atomic** — do not add one "to be safe"; it would be dead weight and could mask a bug if a second thread ever calls `next_ready_batch()`.

**Warning:** if you ever call `next_ready_batch()` from more than one thread, this invariant breaks silently (two threads could interleave and desynchronize `next_consume_idx_` from the slot rotation each `DataLoader` expects). Enforce single-caller with an assert (e.g. check trainer thread id) if this is a concern.

## Shutdown

```cpp
void shutdown();
```

1. Signal the dispatcher thread to exit its loop (e.g. `shutting_down_.store(true)`; dispatcher checks between rotations and after each blocking wait — the blocking waits themselves need a way to wake on shutdown, see warning below).
2. Drain and join both pools (`BS::thread_pool` supports `wait()` / destructor drains automatically).
3. Join the dispatcher thread.
4. Slots left in `filling` state are safe: their events will fire on GPU; buffers stay alive because `DataLoader` owns them.

**⚠️ Warning — shutdown during a blocking wait.** `reserve_free_slot_blocking()` and `take_ready_batch()` block on CUDA events / futures that may never resolve if the pipeline is being torn down mid-flight (e.g. a species genuinely has no more data). Plan for this explicitly: give these blocking calls a way to be interrupted (e.g. check `shutting_down_` in a loop with a bounded `cudaEventQuery` poll as the *shutdown-only* fallback, or have `Ring::shutdown()` mark all loaders "closing" so in-flight waits return a sentinel). Do not let `shutdown()` hang forever waiting on GPU work that will never come.

## Bounds and invariants

- **Back-pressure**: because the dispatcher only submits a fill task once it has reserved a specific empty slot, in-flight fill tasks never exceed the total slot budget across all loaders.
- **Fairness**: strict round-robin dispatch and strict round-robin consume mean each species gets exactly its share of batches. No lane is starved.
- **Order preservation**: `DataLoader` slots are consumed in the exact same fixed rotation they were filled in (see [data_loader.md](data_loader.md) "Slot rotation" section) — this is what guarantees batches are dispensed in the order they were shuffled, with no reordering possible.
- **Trainer never waits unnecessarily**: as long as prefetch is faster than consume, `next_ready_batch()` returns immediately.

## Testability

Follow the existing `tests/slot/test_slot.cu` convention: plain `bool test_*()` functions, `[PASS]`/`[FAIL]` printed to stdout/stderr, `main()` aggregates and returns 0/1. New tests live in `tests/ring/`.

- Register 2 fake loaders (stub `DataLoader` with in-memory fake data) with 4 slots each; verify the dispatcher never submits a fill task before a slot is actually empty.
- Verify `next_ready_batch()` returns loader-A, loader-B, loader-A, loader-B, … in order across many calls.
- Verify order-preservation end to end: feed a fake loader batches tagged 0,1,2,...,N; verify `next_ready_batch()` returns them in exactly that order even if a later slot happens to finish filling first (simulate by adding artificial variable delay per slot).
- Verify decode-pool sharing: two loaders both trigger a chunk decode "at the same time" (simulate with a barrier); verify total concurrently-running decode tasks never exceeds `n_decode_workers`.
- Kill-and-restart: `shutdown()` from an arbitrary state leaves no hanging threads and no leaked CUDA events.
