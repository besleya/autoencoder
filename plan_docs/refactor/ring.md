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

## Dynamic pool rebalancing (watch idle count, move threads between pools)

`BS::thread_pool` supports this directly — confirmed via its docs: `get_tasks_running()`, `get_tasks_queued()`, `get_thread_count()` for monitoring, and `reset(n)` to safely resize a pool on-the-fly.

**`reset(n)` semantics (confirmed from the library docs):** it waits for **all** tasks to finish — both those *currently running* and any still *waiting in the queue* — then destroys the pool and relaunches it with `n` threads. This means calling `reset()` while `fill_pool_` has a backlog of queued fill tasks will block until that whole backlog drains, not just the in-flight ones. This is why rebalancing must only happen from the dispatcher thread, between cycles, never from inside a task running on the pool being resized.

**Design, per explicit correction from the user:** Ring does **not** increment/decrement pool sizes by a fixed step. Instead it watches `fill_pool_`'s idle thread count and, once that count has been consistently high (or consistently zero) for several checks, computes the target size directly and jumps to it in one `reset()` call. Each pool is independently capped at `std::thread::hardware_concurrency()` — that's a per-pool ceiling, not a combined budget across both pools (verified against the user's wording: "**Neither** thread pool should have more threads than hardware concurrency").

```cpp
// Called once per dispatch cycle, not per task — rebalancing more often than that is noise.
void Ring::maybe_rebalance() {
  const std::size_t hw = std::thread::hardware_concurrency();
  const std::size_t fill_threads = fill_pool_.get_thread_count();
  const std::size_t fill_idle = fill_threads - fill_pool_.get_tasks_running();

  if (fill_idle > 0) {
    idle_streak_++;
    busy_streak_ = 0;
  } else if (fill_pool_.get_tasks_queued() > 0) {
    // fully busy AND backlog waiting — a real signal, not just "got lucky this instant"
    busy_streak_++;
    idle_streak_ = 0;
  } else {
    idle_streak_ = busy_streak_ = 0;   // fully busy but no backlog: no evidence either way
  }

  if (idle_streak_ >= kStreakThreshold && fill_threads > kMinFillThreads) {
    // Usually idle: shrink fill_pool_ by the idle amount, grow decode_pool_ by the same,
    // capped so decode_pool_ doesn't exceed hardware_concurrency.
    std::size_t move = std::min(fill_idle, fill_threads - kMinFillThreads);
    move = std::min(move, hw - decode_pool_.get_thread_count());
    if (move > 0) {
      fill_pool_.reset(fill_threads - move);
      decode_pool_.reset(decode_pool_.get_thread_count() + move);
    }
    idle_streak_ = 0;
  } else if (busy_streak_ >= kStreakThreshold && fill_threads < hw) {
    // Usually fully busy with a backlog: grow fill_pool_ by stealing from decode_pool_,
    // capped so fill_pool_ doesn't exceed hardware_concurrency and decode_pool_ keeps its floor.
    std::size_t room = hw - fill_threads;
    std::size_t available = decode_pool_.get_thread_count() - kMinDecodeThreads;
    std::size_t move = std::min(room, available);
    if (move > 0) {
      decode_pool_.reset(decode_pool_.get_thread_count() - move);
      fill_pool_.reset(fill_threads + move);
    }
    busy_streak_ = 0;
  }
}
```

**⚠️ Warnings — read before implementing:**
- `reset()` **blocks the calling thread** until all of that pool's tasks — running and queued — finish. Never call it from a thread that itself needs to keep dispatching (i.e. do this from the dispatcher thread only, between round-robin cycles, never from inside a fill or decode task).
- **Hysteresis via streak counters** (`idle_streak_`/`busy_streak_`, threshold e.g. 5 consecutive cycles) is required — a single noisy observation must not trigger a `reset()`. Reset both streaks to 0 after any rebalance, or on an ambiguous observation, so a brief blip doesn't accumulate toward a future trigger.
- **Floors**: never shrink either pool below `kMinFillThreads` / `kMinDecodeThreads` (e.g. 1 each).
- **Ceilings**: never grow either pool above `hardware_concurrency()` — checked independently per pool, not against the combined total (the two pools' sizes may sum to more or less than `hardware_concurrency()`; that's fine and expected as threads move between them).
- This is a tuning nice-to-have, not a correctness requirement. If it turns out to add more complexity than value in practice, it's safe to ship fixed-size pools first and add rebalancing later — flag this to the user if it's cut for time.

## Registration

```cpp
void add_loader(DataLoader* loader);       // Ring stores the pointer; loader must outlive Ring
```

Order of registration = round-robin order. Register all loaders **before** `start()`. `add_loader()` also calls `loader->set_ring(this)`, giving the `DataLoader` a back-reference so it can reach `decode_pool()` and anything else it needs from `Ring` — this replaces threading a decode-pool reference through every `DataLoader` constructor:

```cpp
void Ring::add_loader(DataLoader* loader) {
  loaders_.push_back(loader);
  loader->set_ring(this);
}
```

## Scheduler thread

`ring.start()` launches one dedicated **dispatcher thread** (separate from both pools). Its loop:

```
for loader in loaders (round-robin, forever):
    slot = loader.reserve_slot()   # blocks until the loader's *next* slot is empty
    fill_pool_.submit_task([loader, slot] { loader.fill(slot); })
    maybe_rebalance()             // once per full cycle, see "Dynamic pool rebalancing" below
```

`loader.fill(slot)` is where all real work happens: chunk load (if needed, via `decode_pool_`), Batch construction, log-norm, H2D, event record. On completion the Slot transitions filling→ready. This runs on a `fill_pool_` worker, **not** the dispatcher thread — the dispatcher must stay free to keep reserving/submitting for other loaders.

**Strict rotation — never skip.** If the current loader's next slot isn't empty yet, the dispatcher **blocks** on it (see `reserve_slot()` in [data_loader.md](data_loader.md) — it's a single `Slot::await_empty()` call, no polling). It does not advance to the next loader. This is the opposite of the current Ring implementation, which skips empty lanes in round-robin mode — that behavior is a bug and must not be carried forward.

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

**Confirmed with the user:** exactly one thread (the dispatcher) calls `reserve_slot()`, and exactly one thread (the trainer) calls `next_ready_batch()`, for the lifetime of the program. If that ever changes, the single-writer assumption above must be revisited first.

**Warning:** if you ever call `next_ready_batch()` from more than one thread, this invariant breaks silently (two threads could interleave and desynchronize `next_consume_idx_` from the slot rotation each `DataLoader` expects). Enforce single-caller with an assert (e.g. check trainer thread id) if this is a concern.

## Shutdown

```cpp
void shutdown();
```

1. Signal the dispatcher thread to exit its loop (e.g. `shutting_down_.store(true)`; dispatcher checks between rotations and after each blocking wait — the blocking waits themselves need a way to wake on shutdown, see warning below).
2. Drain and join both pools (`BS::thread_pool` supports `wait()` / destructor drains automatically).
3. Join the dispatcher thread.
4. Slots left in `filling` state are safe: their events will fire on GPU; buffers stay alive because `DataLoader` owns them.

**⚠️ Warning — shutdown during a blocking wait.** `reserve_slot()` and `take_ready_batch()` block on CUDA events / futures that may never resolve if the pipeline is being torn down mid-flight (e.g. a species genuinely has no more data). Plan for this explicitly: give these blocking calls a way to be interrupted (e.g. check `shutting_down_` in a loop with a bounded `cudaEventQuery` poll as the *shutdown-only* fallback, or have `Ring::shutdown()` mark all loaders "closing" so in-flight waits return a sentinel). Do not let `shutdown()` hang forever waiting on GPU work that will never come.

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
