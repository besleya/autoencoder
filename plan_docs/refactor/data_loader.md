# DataLoader

Per-species batch producer. Owns its slots. Passive — driven by Ring's dispatcher.

## Responsibilities

- Own a fixed set of `Slot`s (prefetch depth, e.g. 4).
- Own the on-disk file order and current chunk state.
- Produce `Batch`es from chunks, one at a time when Ring asks.
- Report free-slot / ready-slot counts to Ring.

## Non-responsibilities

- Does **not** own threads. Work runs on Ring's pool.
- Does **not** know about the trainer, autoencoder, or loss.
- Does **not** know about other species' loaders.

## Constructor

```cpp
DataLoader(std::string species_name,
           std::vector<std::string> file_paths,
           int chunk_size,
           int batch_size,
           int n_slots,
           std::mt19937& rng);
```

Creates `n_slots` `Slot`s (each pre-allocates its device + pinned buffers to the expected max sizes for this species). Does not touch files until `start()`.

## Lifecycle

```cpp
void start();          // shuffle file order, open first file lazily
int  feature_count();  // peek at first file; used by Translator::species()
```

## Interface for Ring

```cpp
bool  has_free_slot() const;              // any slot in state FREE?
int   free_slot_count() const;
int   ready_slot_count() const;
Slot* reserve_free_slot();                // atomically FREE → FILLING; returns nullptr if none
void  fill(Slot* slot);                   // called on pool worker; runs to completion
Slot* next_ready_slot();                  // blocks until any slot is READY; returns it
```

## `fill(slot)` — the pool-worker entry point

Runs on a Ring pool thread. Steps:

1. Ensure a chunk is loaded (advance to next file if current chunk is exhausted; reshuffle at end of epoch, bump pass counter).
2. Construct a `Batch` object bound to `slot`'s buffers, populated from the next `batch_size` rows of the current chunk.
3. Call `batch.prepare(slot->stream())` which does: type-convert into pinned host buffers → `cudaMemcpyAsync` H2D → launch log-norm kernel on the slot's stream → `cudaEventRecord(slot->ready_event(), slot->stream())`.
4. Transition slot state FILLING → READY (state read happens on GPU via the event; the CPU-visible flag flips here so `ready_slot_count()` is accurate).

The pool worker returns immediately after step 4 — it does not wait for the GPU event. Ring's back-pressure guarantees the slot stays valid until the trainer consumes it.

### Chunk state

DataLoader keeps at most one "current chunk" in memory. Chunk loading (decode from `.1pz`) can be:

- **Option A (simple)**: done inline inside `fill()` when the current chunk is exhausted. The pool worker that triggers this waits on decode; other workers can still fill other loaders' slots.
- **Option B (later)**: submit a separate decode task to the pool, keep a small chunk queue. Only worth doing if profiling shows chunk-load stalls.

Start with A. Note in code where B would slot in.

## Pass counter and epoch boundaries

DataLoader increments a public `int pass()` counter each time its file list wraps. The trainer reads `translator.pass_of("primary_species")` (via loader ref) to decide when to stop. **Batch::chunk_end() is true on the last batch of a chunk**, allowing the trainer to trigger per-chunk loss readouts.

## Ownership summary

| Owned by DataLoader | Borrowed |
|---|---|
| `n_slots` × `Slot` (device + pinned mem, ready_event) | Nothing |
| Current chunk (host-side decoded rows) | RNG (ref from ctor) |
| File paths list, current-file cursor | |
| Pass counter | |

## Testability

- Feed 2 tiny fake files; verify `n_slots` fills, `ready_slot_count()` reaches `n_slots`, no more work happens without a `release`.
- Verify pass counter increments on wrap.
- Verify determinism: same rng seed → same batch order across runs.
