# Ring

Scheduler and thread-pool owner. Cycles DataLoaders, keeps prefetch queues warm, hands ready batches to the trainer.

## Responsibilities

- Own the CPU thread pool.
- Own (or hold refs to) the set of registered `DataLoader`s.
- Decide **which DataLoader** to schedule next (MVP: strict round-robin).
- Enforce back-pressure: never submit a prepare-task to the pool if the target `DataLoader` has no free `Slot`.
- Serve the trainer: `next_ready_batch()` returns the next completed `Batch` in scheduling order.

## Non-responsibilities

- Ring does not own slot buffers. `DataLoader` does.
- Ring does not know about layers, autoencoders, or loss.
- Ring does not perform log-norm or memcpy — that's inside `Batch::prepare()`, run on pool workers.

## Constructor

```cpp
Ring(int n_worker_threads);
```

Creates a thread pool of `n_worker_threads` CPU workers. Empty loader list.

## Registration

```cpp
void add_loader(DataLoader* loader);       // Ring stores the pointer; loader must outlive Ring
```

Order of registration = round-robin order.

## Scheduler thread

Ring runs one dedicated **dispatcher thread** (separate from the pool). Its loop:

```
for loader in loaders (round-robin, forever):
    wait until loader.has_free_slot()      # blocks — back-pressure
    slot = loader.reserve_free_slot()      # atomically marks it "filling"
    pool.submit( [loader, slot] { loader.fill(slot); } )
```

`loader.fill(slot)` is where all real work happens: chunk load (if needed), Batch construction, log-norm, H2D, event record. On completion the Slot transitions filling→ready.

**Why block on the current loader rather than skip?** Simpler; and if any one loader is falling behind, throttling the others prevents unbounded queue growth in the pool. The alternative (skip and revisit) can be added later if profiling shows imbalance hurts throughput.

## Trainer interface

```cpp
Batch next_ready_batch();      // blocks until any registered loader has a ready slot
```

Selection policy for **which ready batch** to hand to the trainer: also round-robin, in the same order the dispatcher submitted. This preserves the invariant "the next Batch from the next species is ready when the trainer requests it" — the dispatcher and the consumer walk the loader list in lockstep.

Concretely, Ring keeps a `next_consume_idx`. `next_ready_batch()`:
1. Waits on `loaders_[next_consume_idx]` for a ready slot.
2. Constructs (or reveals) the `Batch` for that slot.
3. Advances `next_consume_idx` (round-robin).

Consumer signals the slot free via `Batch`'s consume/destruct path (see [batch.md](batch.md)).

> **Bug to NOT reproduce.** The current Ring's round-robin mode *skips* a lane if it has no ready slot. That is a bug. The new `Ring::next_ready_batch()` must **block** on `loaders_[next_consume_idx]` until that specific loader has a ready slot. Never advance the cursor past a species that isn't ready. Strict rotation, always.

## Shutdown

```cpp
void shutdown();
```

1. Signal the dispatcher thread to exit its loop.
2. Drain the pool (`join_all`).
3. Slots left in `filling` state are safe: their events will fire on GPU; buffers stay alive because `DataLoader` owns them.

## Bounds and invariants

- **Back-pressure**: pool queue depth is bounded by `sum(slots per loader) − ready count`. Because dispatcher blocks on `has_free_slot()`, the pool never accumulates beyond the total slot budget.
- **Fairness**: strict round-robin dispatch and strict round-robin consume mean each species gets exactly its share of batches. No lane is starved.
- **Trainer never waits unnecessarily**: as long as prefetch is faster than consume, `next_ready_batch()` returns immediately.

## Testability

- Register 2 fake loaders with 4 slots each; verify dispatcher never submits when both loaders are full.
- Verify `next_ready_batch()` returns loader-A, loader-B, loader-A, loader-B, … in order.
- Kill-and-restart: `shutdown()` from an arbitrary state leaves no hanging threads and no leaked events.
