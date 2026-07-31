# DataLoader

Per-species batch producer. Owns its slots. Passive — does nothing until directed by Ring's dispatcher.

## Responsibilities

- Own a fixed set of `Slot`s (prefetch depth, e.g. 4), allocated up front at construction.
- Own the on-disk file order, chunking, and current-chunk state.
- Each epoch: shuffle all files, split into chunks; load a chunk into a contiguous host CSC matrix; shuffle that chunk's column indices; produce `Batch`es from consecutive slices of the shuffled columns.
- Track its own epoch ("pass") counter.

## Non-responsibilities

- Does **not** own any thread pool. Fill work runs on `Ring`'s `fill_pool_`; decode work runs on `Ring`'s `decode_pool_` (passed in).
- Does **not** know about the trainer, autoencoder, or loss.
- Does **not** know about other species' loaders.

## Constructor

```cpp
DataLoader(std::string species_name,
           std::vector<std::string> file_paths,
           int chunk_size,
           int batch_size,
           int n_slots,
           std::mt19937 rng,                 // owned copy, seeded per-loader — see warning below
           BS::thread_pool& decode_pool,      // Ring's shared decode pool; DataLoader does not own it
           bool double_buffer_chunks = true); // see "Chunk double-buffering"
```

Allocates `n_slots` `Slot`s immediately (each pre-allocates device + pinned buffers to the expected max column/nnz capacity for this species — `Slot::grow()` still handles the rare case a batch is larger than expected). Does not touch files until `start()`.

**⚠️ Warning — RNG per loader, not shared.** Passing one `std::mt19937&` shared across all species' `DataLoader`s (as older notes suggested) makes every loader's shuffling depend on call order across *other* loaders — nondeterministic once fill tasks run concurrently on the pool. Give each `DataLoader` its own `std::mt19937`, seeded deterministically from a master seed + species index (e.g. `master_seed + hash(species_name)`), so results are reproducible regardless of scheduling.

## Lifecycle

```cpp
void start();          // shuffle file order for epoch 0, load first chunk synchronously, shuffle its columns
int  feature_count() const;  // valid after start(); used by Translator::add_species()
int  pass() const;           // epoch counter, increments each time the file list wraps
```

`start()` is called once per loader, on the main thread, **before** `ring.start()`. It's allowed to block (it does real I/O) since this happens during init, not during training.

## Interface for Ring

```cpp
Slot* reserve_free_slot_blocking();          // dispatcher-thread only; see "Slot rotation" below
void  fill(Slot* slot);                      // fill_pool_ worker entry point
Batch take_ready_batch();                    // trainer-thread only; blocks until next slot (in rotation) is full
```

## Slot rotation — the key design idea (replaces polling/mutex bookkeeping)

Each `DataLoader` has a **fixed cyclic order** for its own slots: `slots_[0], slots_[1], ..., slots_[n_slots-1], slots_[0], ...`. Batch *i* (in shuffled order) always goes into `slots_[i % n_slots]`. Both production and consumption walk this same sequence, so ordering is automatically preserved as long as each side visits it in lockstep:

```cpp
Slot* DataLoader::reserve_free_slot_blocking() {
  Slot* slot = slots_[fill_idx_].get();
  slot->await_empty();          // CPU-blocks until THIS slot's empty_event has fired
  slot->fill();                 // empty -> filling
  fill_idx_ = (fill_idx_ + 1) % n_slots_;
  return slot;
}

Batch DataLoader::take_ready_batch() {
  Slot* slot = slots_[consume_idx_].get();
  slot->await_full();           // CPU-blocks until THIS slot's ready_event has fired
  consume_idx_ = (consume_idx_ + 1) % n_slots_;
  return Batch(slot, ...);      // consumer takes ownership; slot -> empty happens on Batch destruction
}
```

**Why this is better than "any free slot" + condvar:** with a fixed rotation, "is there a free slot" reduces to "has this *specific* slot's `empty_event` fired", which `Slot` already answers via `await_empty()` (a `cudaEventSynchronize`, no polling, no extra mutex). This removes the need for `slot_mtx_`/`slot_cv_`/free-count/ready-count bookkeeping entirely — the CUDA events plus two plain `int` cursors are sufficient. It also **is** the mechanism that guarantees strict batch ordering: slot *N* always holds batch *N*, never an out-of-order one.

**⚠️ Concurrency rule:** `fill_idx_` must only ever be written by the code path called from Ring's single dispatcher thread; `consume_idx_` must only ever be written by the code path called from Ring's single consumer (trainer) call. Do not add locking "to be safe" — if you find yourself needing a mutex around these ints, it means something is calling `reserve_free_slot_blocking()` or `take_ready_batch()` from more than one thread, which breaks the ordering invariant and should be fixed at the call site, not papered over with a lock.

## `fill(slot)` — the fill_pool_ worker entry point

Runs on a `Ring::fill_pool_` worker. Steps:

1. **Claim next batch's column slice.** Lock `chunk_mtx_` briefly: if the current chunk still has ≥ `batch_size` unconsumed shuffled columns, take the next slice and advance `col_cursor_`; else trigger chunk advance (below). Release lock quickly — this is the only part of `fill()` that touches shared loader state.
2. Construct a `Batch` bound to `slot`'s buffers with that column slice (borrowed reference to the current `Chunk`, no copy).
3. Call `batch.prepare()`: gather + cast + CPU log-norm into `slot`'s pinned buffers, `cudaMemcpyAsync` ×3 on `slot->stream()`, `cudaEventRecord(slot->ready_event(), slot->stream())`.
4. Return. **Does not wait on the GPU event** — the slot is already logically "filling→will become full"; `await_full()` on the consumer side is what actually blocks until the event fires.

### Chunk advance (step 1, when the current chunk is exhausted)

1. If `col_cursor_` reaches the end of the current chunk's columns: if a **double-buffered next chunk** is already decoded (see below), swap it in as `current_chunk_`, reset `col_cursor_ = 0`, shuffle its column order using this loader's RNG.
2. If no next chunk is ready yet, decode now (this call blocks the `fill_pool_` worker — acceptable, see [ring.md](ring.md) "Two pools"):
   - Pick the next `chunk_size` files from `file_order_` (wrapping and reshuffling `file_order_` + incrementing `pass_` if the file list wraps).
   - Submit one task per file to `decode_pool_.submit_task(...)`, each returning a small per-file CSC struct.
   - `.get()` all futures (blocks), then concatenate into one contiguous `Chunk` (rebased `col_ptr`, concatenated `row_idx`/`values`). Concatenation itself is a simple prefix-sum over per-file nnz counts — cheap relative to decode, can stay single-threaded.
   - Shuffle the new chunk's column order with this loader's RNG.

## Chunk double-buffering (configurable — `double_buffer_chunks_`)

When enabled (default), the loader starts decoding the **next** chunk in the background before the current one is exhausted, so chunk-advance in step 1 above almost always finds it already done:

- Trigger point: when `col_cursor_` crosses a threshold (e.g. 50% of current chunk consumed), if no next-chunk decode is already in flight, kick it off: submit the same per-file decode tasks to `decode_pool_`, but store a `std::shared_future<Chunk>` (`next_chunk_future_`) instead of blocking. Whichever `fill()` call reaches the actual chunk boundary later just does `next_chunk_future_.get()`.
- Guard against double-triggering: use a `std::once_flag` or a simple bool checked under `chunk_mtx_` so only one `fill()` call kicks off the prefetch for a given chunk transition.
- **Cost when enabled:** two chunks resident in host memory at once (current + next). Per PLAN.md's numbers (chunk=64 files ≈ 6 GB), that's ~12 GB per loader — confirm this fits your RAM budget for the number of species you run concurrently.
- **When disabled** (`double_buffer_chunks_ = false`): chunk decode only starts inline at the boundary (the simple "Option A" from earlier notes) — lower memory, but the `fill_pool_` worker (and thus that loader's prefetch) stalls for the full decode time at every chunk boundary. Useful for memory-constrained runs or for isolating decode-latency issues while profiling.

## Pass counter and epoch boundaries

DataLoader increments a public `int pass()` counter each time its file list wraps (all chunks of the current epoch consumed). `Batch::chunk_end()` is true on the last batch of a chunk (not necessarily the last of an epoch), letting the trainer trigger per-chunk loss readouts.

## Ownership summary

| Owned by DataLoader | Borrowed |
|---|---|
| `n_slots` × `Slot` (device + pinned mem, events) | `decode_pool_` (ref to Ring's pool) |
| Current chunk + (optionally) next chunk being decoded | |
| File paths list, shuffled `file_order_`, `col_cursor_`, `fill_idx_`, `consume_idx_` | |
| Its own `std::mt19937 rng_` (seeded at construction, not shared) | |
| Pass counter | |

## ⚠️ Other warnings worth flagging now

- **int32 nnz overflow**: a chunk's total nnz (`sum` over `chunk_size` files) can plausibly exceed `2^31` for large chunk sizes on dense-ish data. `col_ptr`/`row_idx` are `int32_t` per `Slot`'s existing layout. Either (a) add an assertion/check in chunk-concat that throws if total nnz would overflow, or (b) budget `chunk_size` conservatively so it can't happen. Decide which before writing the concat code — don't leave it silently wrapping.
- **Per-file decode function must be thread-safe and reentrant** — it will be called concurrently, possibly from multiple `DataLoader`s sharing the same `decode_pool_` at once. No shared mutable state across calls (each call should open/parse/close its own file handle and return a freshly-allocated result).
- **`Chunk` lifetime vs. in-flight `Batch`es**: a `Batch` borrows from `current_chunk_` without copying. Make sure `current_chunk_` isn't freed/swapped out while a `Batch` built from it is still being gathered in `prepare()`. Since chunk swap only happens inside the locked step 1 of `fill()`, and `Batch::prepare()` for a given batch happens strictly after that batch's column slice was claimed, this is safe as long as the chunk isn't mutated in place — only ever replaced by assigning a new `Chunk` object once the old one's last batch has been claimed (not necessarily *consumed*, just claimed — the raw column data it read from is a `shared_ptr<const Chunk>` kept alive by each `Batch` that references it, so freeing is safe once no `Batch` holds a reference). Recommend `current_chunk_` be a `std::shared_ptr<const Chunk>`, and each `Batch` store a `shared_ptr` copy (not a raw pointer) precisely to make this safe automatically instead of by convention.

## Testability

Follow the existing test convention (`tests/slot/test_slot.cu`): plain `bool test_*()` functions, `[PASS]`/`[FAIL]` output, `main()` aggregates. New tests live in `tests/loader/` alongside the existing `bench_loader*` files (which are benchmarks, not correctness tests — keep them separate from the new `test_data_loader.cu`).

- Feed 2 tiny fake files; verify all `n_slots` fill in rotation order and `take_ready_batch()` returns batches whose column slices are in the exact shuffled order generated at chunk start.
- Verify pass counter increments on file-list wrap, and column shuffle changes between epochs (different pass → different order) while being deterministic for a fixed seed.
- Verify determinism: same rng seed → identical sequence of batches (file order, chunk contents, column order) across two full runs.
- Verify double-buffering: with `double_buffer_chunks_ = true`, assert the next chunk's decode is submitted before the current chunk's last batch is claimed (instrument with a counter/mock decode_pool). With it `= false`, assert decode is *not* submitted until the boundary is actually reached.
- Verify chunk-lifetime safety: construct a `Batch` from a chunk, then force a chunk swap, then call `batch.prepare()` — must still read valid data (proves the `shared_ptr<const Chunk>` keeps it alive).
