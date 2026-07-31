# DataLoader

Per-species batch producer. Owns its slots. Passive — does nothing until directed by Ring's dispatcher.

## Responsibilities

- Own a fixed set of `Slot`s (prefetch depth, e.g. 4), allocated up front at construction.
- Own the on-disk file order, chunking, and current-chunk state.
- Each epoch: shuffle all files, split into chunks; load a chunk into a contiguous host CSC matrix; shuffle that chunk's column indices; produce `Batch`es from consecutive slices of the shuffled columns.
- Track its own epoch counter.
- Skip a chunk's trailing partial slice (fewer than `batch_size` columns left) rather than emitting an undersized `Batch` — see "Skipping the trailing partial batch" below.

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
           bool double_buffer_chunks = true); // see "Chunk double-buffering"

void set_ring(Ring* ring);   // called by Ring::add_loader(); gives DataLoader access to decode_pool()
```

Allocates `n_slots` `Slot`s immediately (each pre-allocates device + pinned buffers to the expected max column/nnz capacity for this species — `Slot::grow()` still handles the rare case a batch is larger than expected). Does not touch files until `start()`.

**`set_ring()` replaces a constructor-injected decode-pool reference.** `Ring` is the one place that knows about all pools and all loaders, so it's the natural single source of truth: `Ring::add_loader()` calls `loader->set_ring(this)` right after registering it, and `DataLoader` stores `Ring* ring_` and calls `ring_->decode_pool()` whenever it needs to submit a decode task. **Warning:** `fill()`, `reserve_slot()`, etc. must never be called before `set_ring()` has run — assert `ring_ != nullptr` at the top of any method that touches `ring_->decode_pool()`, since construction order (loader built, *then* registered with Ring) makes it easy to accidentally call something too early.

**⚠️ Warning — RNG per loader, not shared.** Passing one `std::mt19937&` shared across all species' `DataLoader`s (as older notes suggested) makes every loader's shuffling depend on call order across *other* loaders — nondeterministic once fill tasks run concurrently on the pool. Give each `DataLoader` its own `std::mt19937`, seeded deterministically from a master seed + species index (e.g. `master_seed + hash(species_name)`), so results are reproducible regardless of scheduling.

## Lifecycle

```cpp
void start();          // shuffle file order for epoch 0, load first chunk synchronously, shuffle its columns
int  feature_count() const;  // valid after start(); used by Translator::add_species()
int  epoch() const;          // epoch counter, increments each time the file list wraps
```

`start()` is called once per loader, on the main thread, **before** `ring.start()`. It's allowed to block (it does real I/O) since this happens during init, not during training.

## Interface for Ring

```cpp
Slot* reserve_slot();                        // dispatcher-thread only; see "Slot rotation" below
void  fill(Slot* slot);                      // fill_pool_ worker entry point
Batch take_ready_batch();                    // trainer-thread only; blocks until next slot (in rotation) is full
```

## Slot rotation — the key design idea (replaces polling/mutex bookkeeping)

Each `DataLoader` has a **fixed cyclic order** for its own slots: `slots_[0], slots_[1], ..., slots_[n_slots-1], slots_[0], ...`. Batch *i* (in shuffled order) always goes into `slots_[i % n_slots]`. Both production and consumption walk this same sequence, so ordering is automatically preserved as long as each side visits it in lockstep:

```cpp
Slot* DataLoader::reserve_slot() {
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

**⚠️ Concurrency rule:** `fill_idx_` must only ever be written by the code path called from Ring's single dispatcher thread; `consume_idx_` must only ever be written by the code path called from Ring's single consumer (trainer) call. Do not add locking "to be safe" — if you find yourself needing a mutex around these ints, it means something is calling `reserve_slot()` or `take_ready_batch()` from more than one thread, which breaks the ordering invariant and should be fixed at the call site, not papered over with a lock.

## `fill(slot)` — the fill_pool_ worker entry point

Runs on a `Ring::fill_pool_` worker. Steps:

1. **Claim next batch's column slice.** Lock `chunk_mtx_` briefly: if the current chunk still has ≥ `batch_size` unconsumed shuffled columns, take the next full slice of exactly `batch_size` columns and advance `col_cursor_`; if fewer than `batch_size` columns remain, **discard that remainder** (see "Skipping the trailing partial batch" below) and trigger chunk advance instead of building an undersized `Batch`. Note whether the slice just claimed is the **last full slice this chunk has** (i.e. after claiming it, fewer than `batch_size` columns remain, so nothing further can be claimed without a chunk advance). Release the lock quickly — this is the only part of `fill()` that touches shared loader state.
2. Construct a `Batch` bound to `slot`'s buffers with that column slice (a `shared_ptr<const Chunk>` copy, not a raw pointer — see `Chunk` lifetime warning below).
3. Call `batch->prepare()` — this is the **only** `Batch` method `fill()` calls; it does gather+cast+CPU log-norm+H2D+event-record internally (see [batch.md](batch.md)). `DataLoader` never calls `gather_normalize()`/`to_device()` directly.
4. Once `prepare()` returns — and only if step 1 marked this as the chunk's last full slice — call `start_next_chunk_decode_async()`. No callback through `Batch` is needed: `prepare()`'s `to_device()` step is fire-and-forget (`cudaMemcpyAsync`, never blocks on the GPU), so calling the kickoff right after `prepare()` returns carries no timing risk and needs no plumbing. See "Ordering up the next chunk" below.
5. Return. **Does not wait on the GPU event** — the slot is already logically "filling→will become full"; `await_full()` on the consumer side is what actually blocks until the event fires.

### Skipping the trailing partial batch

If a chunk's column count isn't an exact multiple of `batch_size`, the leftover columns (fewer than `batch_size`) are simply **dropped** — no undersized `Batch` is ever constructed. This was an explicit simplification: handling a variable-size final batch (padding, or plumbing a smaller `B` through `Slot`'s pre-sized buffers) added complexity for a rare, low-value case. Concretely: `col_cursor_` still advances to `current_chunk_->n` when the remainder is skipped, so the chunk-advance path in step 1 above triggers immediately once no full slice remains, same as if the columns had been consumed.

### Ordering up the next chunk — exact trigger point (not a time/consumption heuristic)

You asked: order up the next chunk as soon as the last `Batch` from the current chunk no longer needs it, but never make a ready batch wait on that. Here's precisely when a `Batch` stops needing the chunk, and how that's wired up:

- Inside `prepare()` (see [batch.md](batch.md)), the internal `gather_normalize()` step is the only one that reads from the `Chunk` (it copies `row_idx`/`values` for this batch's columns into the slot's own pinned buffers); the internal `to_device()` step only reads from the pinned buffers `gather_normalize()` just wrote, and never blocks on the GPU (`cudaMemcpyAsync` only). So by the time `prepare()` returns — for the batch that claimed the chunk's final full slice — the chunk is already no longer needed, and `send_to_device()`'s async H2D is already queued.
- Because `prepare()` is synchronous CPU work + a non-blocking H2D enqueue, `DataLoader::fill()` can simply call `batch->prepare()` then, only for the last-slice batch, `start_next_chunk_decode_async()` — two plain sequential calls, no callback plumbing through `Batch` is needed. (An earlier draft added an `after_gather` callback parameter to `Batch::prepare()` for this; dropped once it was clear the kickoff can just as well happen after `prepare()` returns entirely, since a callback only saves a few microseconds of `to_device()` queuing time and adds a parameter every other call site has to pass `nullptr` for.)
- `start_next_chunk_decode_async()` submits the per-file decode tasks to `ring_->decode_pool()` and stores the resulting `std::shared_future<Chunk>` in `next_chunk_future_` — it does **not** call `.get()`, so the calling `fill_pool_` worker returns from `fill()` immediately after. That batch becomes ready on the normal schedule; decoding of the next chunk proceeds in the background on `decode_pool_`.
- Guard against double-triggering with a bool under `chunk_mtx_` (only the one `fill()` call that claims the final full slice sees "is last slice" == true, so in practice this is already single-fire without extra locking — but assert it, since a future bug in the claiming logic would otherwise silently double-submit).

### Chunk advance (step 1, when the current chunk is exhausted)

1. If `col_cursor_` reaches the end of the current chunk's columns: `next_chunk_future_` should already be done or nearly done (started as soon as the last batch's gather finished, per above) — `.get()` it, swap it in as `current_chunk_`, reset `col_cursor_ = 0`, shuffle its column order using this loader's RNG.
2. If `double_buffer_chunks_` is disabled, or if for some reason no decode was started ahead of time (e.g. `chunk_size <= batch_size`, so there is no "earlier batch" to trigger it from), decode now inline: submit the same per-file tasks to `ring_->decode_pool()` and `.get()` them directly (blocking the `fill_pool_` worker — acceptable, see [ring.md](ring.md) "Two pools").
   - Pick the next `chunk_size` files from `file_order_` (wrapping and reshuffling `file_order_` + incrementing `pass_` if the file list wraps).
   - Concatenate the per-file results into one contiguous `Chunk` (rebased `col_ptr`, concatenated `row_idx`/`values`) — a simple prefix-sum over per-file nnz counts, cheap relative to decode, can stay single-threaded.

## Chunk double-buffering (configurable — `double_buffer_chunks_`)

When enabled (default), decoding the **next** chunk starts the moment the current chunk's last batch no longer needs the chunk (see "Ordering up the next chunk" above) — not on a consumption-percentage heuristic, and not blocking that batch from becoming ready.

- **Cost when enabled:** two chunks resident in host memory at once (current + next). Per PLAN.md's numbers (chunk=64 files ≈ 6 GB), that's ~12 GB per loader — confirm this fits your RAM budget for the number of species you run concurrently.
- **When disabled** (`double_buffer_chunks_ = false`): chunk decode only starts inline at the boundary (the simple "Option A" from earlier notes) — lower memory, but the `fill_pool_` worker (and thus that loader's prefetch) stalls for the full decode time at every chunk boundary. Useful for memory-constrained runs or for isolating decode-latency issues while profiling.

## Epoch counter and epoch boundaries

DataLoader increments a public `int epoch()` counter each time its file list wraps (all chunks of the current epoch consumed). `Batch::chunk_end()` is true on the last (full) batch of a chunk (not necessarily the last of an epoch), letting the trainer trigger per-chunk loss readouts.

## Ownership summary

| Owned by DataLoader | Borrowed |
|---|---|
| `n_slots` × `Slot` (device + pinned mem, events) | `decode_pool_` (ref to Ring's pool) |
| Current chunk + (optionally) next chunk being decoded | |
| File paths list, shuffled `file_order_`, `col_cursor_`, `fill_idx_`, `consume_idx_` | |
| Its own `std::mt19937 rng_` (seeded at construction, not shared) | |
| Epoch counter | |

## ⚠️ Other warnings worth flagging now

- **int32 nnz overflow**: a chunk's total nnz (`sum` over `chunk_size` files) can plausibly exceed `2^31` for large chunk sizes on dense-ish data. `col_ptr`/`row_idx` are `int32_t` per `Slot`'s existing layout. Either (a) add an assertion/check in chunk-concat that throws if total nnz would overflow, or (b) budget `chunk_size` conservatively so it can't happen. Decide which before writing the concat code — don't leave it silently wrapping.
- **Per-file decode function must be thread-safe and reentrant** — it will be called concurrently, possibly from multiple `DataLoader`s sharing the same `decode_pool_` at once. No shared mutable state across calls (each call should open/parse/close its own file handle and return a freshly-allocated result).
- **`Chunk` lifetime vs. in-flight `Batch`es**: a `Batch` borrows from `current_chunk_` without copying. Make sure `current_chunk_` isn't freed/swapped out while a `Batch` built from it is still being gathered in `prepare()`. Since chunk swap only happens inside the locked step 1 of `fill()`, and `Batch::prepare()` for a given batch happens strictly after that batch's column slice was claimed, this is safe as long as the chunk isn't mutated in place — only ever replaced by assigning a new `Chunk` object once the old one's last batch has been claimed (not necessarily *consumed*, just claimed — the raw column data it read from is a `shared_ptr<const Chunk>` kept alive by each `Batch` that references it, so freeing is safe once no `Batch` holds a reference). Recommend `current_chunk_` be a `std::shared_ptr<const Chunk>`, and each `Batch` store a `shared_ptr` copy (not a raw pointer) precisely to make this safe automatically instead of by convention.

## Testability

Follow the existing test convention (`tests/slot/test_slot.cu`): plain `bool test_*()` functions, `[PASS]`/`[FAIL]` output, `main()` aggregates. New tests live in `tests/loader/` alongside the existing `bench_loader*` files (which are benchmarks, not correctness tests — keep them separate from the new `test_data_loader.cu`).

- Feed 2 tiny fake files; verify all `n_slots` fill in rotation order and `take_ready_batch()` returns batches whose column slices are in the exact shuffled order generated at chunk start.
- Verify epoch counter increments on file-list wrap, and column shuffle changes between epochs (different epoch → different order) while being deterministic for a fixed seed.
- Verify determinism: same rng seed → identical sequence of batches (file order, chunk contents, column order) across two full runs.
- Verify the trailing partial slice (chunk size not a multiple of `batch_size`) is dropped: no undersized `Batch` is ever produced, and the chunk advance still triggers correctly.
- Verify double-buffering: with `double_buffer_chunks_ = true`, assert the next chunk's decode is submitted right after `gather()` returns for the batch that claims the chunk's last full slice — and that this batch still becomes ready without waiting for that decode (instrument with a mock `decode_pool` + a delay). With it `= false`, assert decode is *not* submitted until the boundary is actually reached.
- Verify chunk-lifetime safety: construct a `Batch` from a chunk, then force a chunk swap, then call `batch.prepare()` — must still read valid data (proves the `shared_ptr<const Chunk>` keeps it alive).
- Verify `set_ring()` ordering: calling `fill()`/`reserve_slot()` before `set_ring()` asserts/throws rather than dereferencing a null `Ring*`.
