# Refactor Handoff Report

Date: 2026-07-23
Status: Refactor implemented end-to-end. **Not yet built or tested.** Handing off to cleanup agents.

## Cleanup Pass 2 (2026-07-31)

Docs updated this pass: [ring.md](ring.md) (`maybe_rebalance()` rewritten), [batch.md](batch.md) (`prepare()` callback dropped), [data_loader.md](data_loader.md) (kickoff ordering updated to match). Findings from a code survey, in delegation order:

1. **`data_loader.cu` calls methods that don't exist on `Batch`.** `fill()` currently calls `batch->gather()` and `batch->send_to_device()` — neither is a public `Batch` method; only `prepare()` is (per current `batch.h`). Fix: `data_loader.cu`'s `fill()` must call `batch->prepare()` (no arguments — the callback param some docs described was dropped, see [batch.md](batch.md)), then, only if this batch claimed the chunk's last full slice, call `start_next_chunk_decode_async()` — see [data_loader.md](data_loader.md) "Ordering up the next chunk". Also update `batch.h`/`batch.cu`'s `prepare()` signature to drop the `after_gather` callback parameter if it's still present there.
2. **Delete dead code in `data_loader.h`/`data_loader.cu`:** `reserve_free_slot()`, `any_slot_ready()`, `find_ready_slot()` — all superseded by `reserve_slot()`/`take_ready_batch()`'s strict-rotation design, no call sites remain.
3. **Rewrite `Ring::maybe_rebalance()`** in `ring.cu`/`ring.h` per the new averaging-window design in [ring.md](ring.md) (replaces the existing streak-counter implementation). New member state needed: `idle_window_` ring buffer (or equivalent), `idle_window_pos_`, `idle_window_count_`, and the tuning constants `kWindowCycles`, `kShrinkAbove`, `kGrowBelow`, `kIdleMargin` (floors `kMinFillThreads`/`kMinDecodeThreads` already exist).
4. **Deferred to a later pass (do not do yet):** renaming functions/variables across ring/batch/data_loader/slot to the 1-2 word style (e.g. `start_next_chunk_decode_async`, `ensure_chunk_loaded_locked` are candidates). Ordered last since it should follow the design changes above, not precede them.


## What was done

Refactored the trainer into six narrow classes per the specs in this folder:

| Class | File(s) | Status |
|---|---|---|
| [Slot](../../slot.h) | slot.h, slot.cu | New — clean-room |
| [Batch](../../batch.h) | batch.h, batch.cu | New — clean-room |
| [DataLoader](../../data_loader.h) | data_loader.h, data_loader.cu | New — duplicates .1pz decode logic from old loader |
| [Ring](../../ring.h) | ring.h, ring.cu | Rewritten in place; `SparseBatch` POD preserved |
| [Translator](../../translator.h) | translator.h, translator.cpp | New |
| GpuAutoencoder / Layer | unchanged | Reused as-is |

**Deleted:** `gpu_data_loader.h`, `gpu_data_loader.cu`.

**Rewritten:** [main_gpu.cpp](../../main_gpu.cpp) — reduced to ~60 lines via extracted helpers (`parse_args`, `build_loaders`, `training_loop`, `report_chunk_loss`, etc.). New CLI flags: `--species NAME:PATTERN` (repeatable), `--workers`, `--shared-layers`, `--species-layers`, `--shared-dims`, `--species-dims`. Backward-compatible: positional file args become species `"default"`.

**Updated:** [Makefile](../../Makefile) — new sources added, `gpu_data_loader.o` removed, `validate` target removed from `all` and marked TODO.

## Key architectural changes vs. old code

1. **Ownership flip.** Slots (buffers + streams + events) now belong to `DataLoader`, not `Ring`. Thread pool now belongs to `Ring`, not `DataLoader`. `Batch` is a transient, memory-less abstraction that borrows from its `Slot`.
2. **Strict round-robin, never skip.** Both Ring's dispatcher and its consumer walk loaders in registration order and **block** on the current loader rather than skipping ahead. Fixes the acknowledged bug in the old Ring's `ROUND_ROBIN` policy.
3. **Multi-species first-class.** `Translator` holds one `GpuAutoencoder` per species over a shared layer core (sandwich layout: species-enc → shared-enc → shared-dec → species-dec). Shared layers are `shared_ptr<Layer>` singletons; per-batch Adam step updates them interleaved across species.
4. **Batch carries its species.** `translator.forward(batch)` dispatches on `batch.species_name()`. No lane/id juggling in the training loop.

## Known issues and follow-ups

These were surfaced by the implementer agents but left for cleanup — they need attention before or after the first build attempt.

### Will likely surface on first build

- **First remote build has not happened.** Five agents wrote code in isolation; interface mismatches, missing includes, or symbol-name drift are likely. Expect a fix pass on the first `make` output.
- **`GpuAutoencoder` vs `Autoencoder` naming.** The Translator agent noticed the real class name is `GpuAutoencoder` and used that. Double-check callers.
- **`SparseBatch` field order** in the new [ring.h](../../ring.h) — must be byte-identical to the old struct (m, B, nnz, d_col_ptr, d_row_idx, d_values, ready_event, chunk_end). Verify against `gpu_autoencoder.cu`/`layer.cu` before trusting.

### Design gaps

- **Ring dispatcher uses a 50µs poll loop** waiting for a free slot from `DataLoader`, because `DataLoader::reserve_free_slot()` is non-blocking (returns `nullptr` when full). The clean fix is a blocking `reserve_free_slot_blocking()` on `DataLoader` that uses the existing cv. TODO comment left in [ring.cu](../../ring.cu). Low priority — polling works, just wastes a hair of CPU.
- **Translator dim constraint.** The Translator agent's build_shared_layers assumes `shared_dims[0] == species_dims.back()` implicitly. This is automatically true if the shared decoder is defined as the reverse of the shared encoder, which is how [translator.cpp](../../translator.cpp) builds it — but worth a code review to confirm the mirrored dims come out right, especially the boundary between shared-decoder-output and species-decoder-input. Add a targeted unit test if possible (checking `layer(k).in_dim`/`out_dim` at every boundary).
- **Batch's log-norm kernel is duplicated** into [batch.cu](../../batch.cu) with a distinct symbol name (`batch_log_normalize_columns_kernel`) to avoid collisions during transition. The original in `gpu_data_loader.cu` is now deleted, so the duplication is safe but the "distinct name" is no longer necessary. Rename to `log_normalize_columns_kernel` for consistency, and consider replacing with the singlet-lib version per the pre-existing plan in [../singlet_lognorm.md](../singlet_lognorm.md).
- **DataLoader duplicates .1pz decoding helpers.** The new [data_loader.cu](../../data_loader.cu) copied the saturate-cast helpers and OpenMP-parallel decode loop from the old `gpu_data_loader.cu` (which is now deleted). No cleanup needed unless you want the decode helpers pulled into a shared utility.

### Untested paths

- **`validate` target** in the Makefile references the old loader; it was removed from `all` and marked TODO. `tests/validate/validate.cpp` needs a rewrite against the new APIs before validation can run.
- **`kernel_bench/` benchmarks** are untouched — they build independently and don't depend on the main tree.
- **`bench_loader_latency`** (if it still exists) — check whether it references the old DataLoader; needs porting to the new one for perf regression checks.

### Behavior differences to verify vs. old code

- **Chunk decode is inline** (Option A in [data_loader.md](data_loader.md)): the pool worker that hits an exhausted chunk blocks on decode. Old code had a dedicated `T_CL` thread with a 2-deep chunk queue. If profiling shows chunk-load stalls hurting throughput, promote to Option B (separate decode tasks in the pool).
- **Per-slot CUDA streams.** Old code had one `loader_stream` per DataLoader; new code has one stream per Slot. This should improve overlap but adds stream-creation cost at startup. Sanity check that stream counts stay reasonable when many loaders × many slots exist.
- **No warm-up / prefetch primer.** Old code called `loader.start()` and let T_BB race ahead of the trainer. New Ring dispatches immediately on `ring.start()` but the trainer may reach `next_ready_batch()` before any slot is ready on the first call. Expect a slightly larger startup stall — not a correctness issue.

## Files to read for full context

- [README.md](README.md) — class map + design decisions
- [ring.md](ring.md) — **includes the "never skip" invariant** critical for correctness
- [translator.md](translator.md), [data_loader.md](data_loader.md), [slot.md](slot.md), [batch.md](batch.md)

## Recommended cleanup sequence

1. `make` and fix compile errors (probably a handful of trivial issues).
2. Run a smoke test with `--species default:<one small file>` to confirm the single-species path still works end-to-end.
3. Add a second species and confirm strict round-robin (log per-batch species names for a few batches).
4. Verify per-species loss numbers converge similarly to the old single-species baseline on the same data.
5. Port `tests/validate/` to new APIs; re-enable in `all`.
6. Optional: add `DataLoader::reserve_free_slot_blocking()` and drop the Ring poll loop.
7. Optional: fold the duplicated log-norm kernel into a shared header, or swap for singlet-lib version.
