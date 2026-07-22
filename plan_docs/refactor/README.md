# Refactor Design — Multi-Species Trainer

Goal: make the code clearer. Split responsibilities into narrow classes, each doing one thing.

## Class map

| Class | Purpose | Owns |
|---|---|---|
| [Translator](translator.md) | Master model. Holds per-species `Autoencoder`s over a shared layer core. | Shared `Layer`s; per-species `Autoencoder`s |
| Autoencoder (existing) | Forward / backward / update over an ordered layer stack. | Per-batch scratch buffers; refs to `Layer`s (via `shared_ptr`) |
| [Ring](ring.md) | Scheduler. Cycles DataLoaders, submits prepare-tasks to a thread pool, dispenses ready batches to trainer. | The CPU thread pool; the set of `DataLoader`s |
| [DataLoader](data_loader.md) | Per-species batch producer. Streams chunks, builds `Batch`es, fills its own `Slot`s. | Its `Slot`s; its chunk-load state; a `cudaStream_t` for H2D |
| [Slot](slot.md) | One prefetch buffer + state (free/filling/ready). Signals via CUDA event. | Device memory buffers; pinned host buffers; a `cudaEvent_t` |
| [Batch](batch.md) | Transient abstraction: one training batch. Log-norms self, moves self to GPU, self-destructs on consume. | *No memory* — borrows buffers from a `Slot` |

## Design decisions (resolved)

- **Layer layout inside each species' Autoencoder — sandwich**:
  `species-specific encoder → shared encoder → shared decoder → species-specific decoder`.
  Shared layers are literal `shared_ptr` singletons: one instance across all species.
- **Shared-layer weight updates — step per batch**. No cross-species grad accumulation. Adam step runs on every backward, from whichever species' batch just fed forward. This preserves the current per-batch `forward → backward_and_step` shape and requires no new synchronization.
- **Species id — string name** (e.g. `"human"`, `"mouse"`). Every `Batch`, `DataLoader`, and per-species `Autoencoder` carries the same name. `Translator::forward(batch)` dispatches on `batch.species_name()`.
- **Ownership flip vs. current code**: today `Ring` owns slot buffers and `DataLoader` owns threads. In the new design **`DataLoader` owns its `Slot`s** (memory follows the producer) and **`Ring` owns the thread pool** (scheduling follows the scheduler). This is the main structural change.

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
