# Refactor Design — Multi-Species Trainer

Goal: make the code clearer. Split responsibilities into narrow classes, each doing one thing.

## Class map

| Class | Purpose | Owns |
|---|---|---|
| [Translator](translator.md) | Master model. Holds per-species `Autoencoder`s over a shared layer core. | Shared `Layer`s; per-species `Autoencoder`s |
| Autoencoder (existing) | Forward / backward / update over an ordered layer stack. | Per-batch scratch buffers; refs to `Layer`s (via `shared_ptr`) |
| [Ring](ring.md) | Scheduler. Cycles DataLoaders, submits prepare-tasks to a thread pool, dispenses ready batches to trainer. | **Two** CPU thread pools (`fill_pool_`, shared `decode_pool_`); the set of `DataLoader`s |
| [DataLoader](data_loader.md) | Per-species batch producer. Streams chunks, builds `Batch`es, fills its own `Slot`s. | Its `Slot`s; its chunk-load state (current + optional next chunk); its own seeded RNG; a `cudaStream_t` for H2D (via its Slots) |
| [Slot](slot.md) | One prefetch buffer + state (free/filling/ready). Signals via CUDA event. | Device memory buffers; pinned host buffers; a `cudaEvent_t` |
| [Batch](batch.md) | Transient abstraction: one training batch. Log-norms self, moves self to GPU, self-destructs on consume. | *No memory* — borrows buffers from a `Slot` |

## Design decisions (resolved)

- **Layer layout inside each species' Autoencoder — sandwich**:
  `species-specific encoder → shared encoder → shared decoder → species-specific decoder`.
  Shared layers are literal `shared_ptr` singletons: one instance across all species.
- **Shared-layer weight updates — step per batch**. No cross-species grad accumulation. Adam step runs on every backward, from whichever species' batch just fed forward. This preserves the current per-batch `forward → backward_and_step` shape and requires no new synchronization.
- **Species id — string name** (e.g. `"human"`, `"mouse"`). Every `Batch`, `DataLoader`, and per-species `Autoencoder` carries the same name. `Translator::forward(batch)` dispatches on `batch.species_name()`.
- **Ownership flip vs. current code**: today `Ring` owns slot buffers and `DataLoader` owns threads. In the new design **`DataLoader` owns its `Slot`s** (memory follows the producer) and **`Ring` owns the thread pool** (scheduling follows the scheduler). This is the main structural change.
- **Two thread pools, not one**: Ring owns a small `fill_pool_` (runs `DataLoader::fill()`, usually cheap) and a large shared `decode_pool_` (runs one task per file, only at chunk boundaries). This is how per-file parallelism is achieved without oversubscribing when multiple loaders hit a chunk boundary at once. See [ring.md](ring.md) "Two pools" for the full rationale.
- **`BS::thread_pool`, vendored**: replaces the hand-rolled queue/mutex/condvar pool in current `ring.cu`. Needs to be added under `third_party/` and wired into the Makefile include path before the next build (not done as part of this planning pass).
- **Deterministic slot rotation, not free/ready bookkeeping**: each `DataLoader`'s slots are filled and consumed in the exact same fixed cyclic order. This alone guarantees batch ordering and lets `reserve_free_slot_blocking()` / `take_ready_batch()` be implemented as a single `Slot::await_empty()` / `await_full()` call — no separate mutex/condvar/counters needed. See [data_loader.md](data_loader.md) "Slot rotation".
- **Chunk double-buffering, toggleable**: `DataLoader` can optionally prefetch the next chunk's decode in the background while the current chunk is still being consumed (`double_buffer_chunks_` flag, default on). Trades ~2× chunk memory for hiding decode latency entirely.

## Bug-fix scope

Current code has known bugs. The refactor is not a bug-for-bug rewrite. Each design doc calls out the invariants the new code must uphold; implementers should treat those as authoritative rather than mirroring current behavior.

## Suggested implementation order

1. `Slot` — leaf class, no dependencies.
2. `Batch` — depends on `Slot` (borrows its buffers).
3. `DataLoader` — depends on `Slot`, `Batch`.
4. `Ring` — depends on `DataLoader`, plus a thread pool.
5. `Translator` — depends on existing `Autoencoder` / `Layer`, no loader coupling.
6. New `main_gpu.cpp` wiring.

Each step is independently compilable and unit-testable.
