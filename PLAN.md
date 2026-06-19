# DataLoader Plan

A plan, in plain language, for a sparse-CSC GPU DataLoader that keeps an H100 trainer fed with batches at all times.

## Goal

The GPU is never idle waiting for data. When the trainer asks for the next batch, it is already on the GPU, already log-normalized, ready to use. If the loader can't keep up, the trainer blocks — and we treat that as a bug to be fixed, not a normal mode.

## Inputs and outputs

- Input: thousands of `.1pz` files. Each one holds a sparse CSC matrix of cells (columns) by features (rows). Roughly 50 MB on disk, 500–2000 cells, ~100k features, ~10% non-zero. We'll have 100M cells total across an epoch.
- Output: a stream of GPU-resident sparse CSC mini-batches with shape `m × batch_size`. The values are already log-normalized. Each batch comes with a CUDA event the trainer waits on — and that wait is GPU-side, so the CPU never blocks.

## What the timing run told us (from [bench_read_1pz.slurm](bench_read_1pz.slurm), 100 files)

**Decoded file sizes are very long-tailed:**

| stat   | decoded bytes |
|--------|---------------|
| min    | ~0            |
| p25    | 7.8 MB        |
| median | 42 MB         |
| mean   | 96 MB         |
| p75    | 145 MB        |
| p95    | 399 MB        |
| max    | 849 MB        |

**Single-file decode times (one thread):** median ~300 ms, mean ~745 ms, p95 ~3.05 s. The tail is the problem, not the median.

**Parallel decode throughput (100 files):**

| threads | wall-clock | files/s | MB/s  | speedup vs 1T |
|---------|------------|---------|-------|---------------|
| 1 (est) | ~75 s      | 1.34    | 13    | 1×            |
| 4       | 23.6 s     | 4.24    | 42    | 3.2×          |
| 8       | 15.4 s     | 6.51    | 65    | 4.8×          |
| 16      | 10.1 s     | 9.88    | 98    | 7.4×          |
| 32      | 7.8 s      | 12.84   | 127   | 9.5×          |

Scaling falls off hard past 16 threads — either disk-bound or contention inside `read_1pz`. **16 threads is the sweet spot** for decode; more is wasted.

**The training-vs-decode budget:**

- One file's cells → batches: with mean ~1000 cells/file and `batch=512`, each file is ~2 batches. At 30 ms/step that's ~60 ms of training per file.
- One file's decode at 16 threads: ~100 ms (16 threads · ~750 ms mean / 100 files · 1 file).
- Decode (~100 ms/file) > training (~60 ms/file). **Decode is faster than training? No — the other way around: decode lags by ~40 ms/file at steady state.**

This means **chunk decode cannot fit inside the wall-clock of one chunk's training.** We need to overlap chunk N+1 decode with chunk N training.

**Conclusions baked into the design below:**

1. Decode is the bottleneck and the long tail (p95 ~3 s) is the danger. One slow file in a chunk can starve the loader for seconds.
2. **Double-buffering chunks is mandatory**, not optional. T_CL must keep at least one decoded chunk *ahead* of T_BB at all times.
3. **Default decode parallelism = 16 threads.** Above 16 returns diminish; we save the rest for trainer, lognorm contention, and the second species loader.
4. **Chunk size recommendation: 64–128 files** for a single loader, **32–64 files** if running two species loaders concurrently. The decoded chunk at 96 MB mean × 64 files = ~6 GB, comfortable on 128 GB RAM. With double-buffering, peak host memory is 2 chunks per loader ≈ 12 GB per loader.
5. **The bench script needs a revision** — the sequential per-file loop dominated wall-clock and made SLURM report 3% CPU. Future runs should skip it and just do the parallel sweeps. Not blocking, just noted.

## How a batch flows

```
.1pz files  ──▶  decode (CPU)  ──▶  shuffled per-cell index
                                            │
                                            ▼
                                    pack a batch into pinned host memory (CPU)
                                            │
                                            ▼
                                    cudaMemcpyAsync × 3  (PCIe)
                                            │
                                            ▼
                                    log-normalize kernel  (GPU, loader stream)
                                            │
                                            ▼
                                    cudaEventRecord  →  trainer fence
```

Everything between "decode" and "event record" runs in the background while the trainer is computing on a previous batch. There are 4 batches in flight at any time so even a slow step here or there doesn't starve the GPU.

## Two background threads, one ring buffer

Two long-lived worker threads do all the loader work:

**T_CL — Chunk Loader.** Pulls the next group of files from the shuffled file list, decodes them in parallel with OpenMP (16 threads), and hands the decoded chunk to T_BB through a **2-deep chunk queue** (revised from 1-deep; see the timing analysis above). T_CL is allowed to run one chunk ahead of T_BB so that p95-slow files don't starve the batch builder.

**T_BB — Batch Builder.** Takes the decoded chunk, picks the next `batch_size` columns from the in-chunk shuffle, copies them into a pinned host buffer, fires off three async H2D copies (col_ptr, row_idx, values), launches the lognorm kernel, and records the ready event. Then advances to the next slot in the ring buffer.

The ring buffer has 4 slots by default. Each slot owns its own pinned host buffers, its own device buffers, and its own event. T_BB writes into the next free slot; the trainer reads from the next ready slot. When the trainer is done with a batch, the slot recycles back to free.

## How the threads coordinate

All thread coordination lives inside a dedicated **`Ring` class** (see next section). The DataLoader's threads don't touch mutexes or condvars directly — they only call `ring.acquire_free()`, `ring.publish_ready()`, `ring.acquire_ready()`, `ring.release_consumed()`. Likewise the chunk queue is a tiny standalone class with its own mutex.

The primitives are still plain `std::mutex` + `std::condition_variable` — no lock-free queues, no futures, no atomics-as-flags.

## Why the trainer never waits on the CPU

`next_batch()` returns the moment a slot is in the `READY` state. `READY` means "all H2D copies and the lognorm kernel have been **enqueued** on the loader's CUDA stream and an event has been recorded." It does NOT mean the GPU has actually finished the work. The trainer doesn't care: it calls `cudaStreamWaitEvent(trainer_stream, batch.ready_event, 0)`, which is a pure GPU-side fence. The CPU returns immediately. The GPU's trainer stream just won't start the forward pass until the lognorm finishes — and given a ring depth of 4 and a 30 ms training step, the lognorm has ~120 ms of runway. It's always done long before the trainer reaches that batch.

## Where the work actually happens

| Step                          | Where    | Why                                                 |
|-------------------------------|----------|-----------------------------------------------------|
| File decode (`read_1pz`)      | CPU, parallel | Likely the slowest step; needs OpenMP. |
| uint32 → int32 cast (indices) | CPU      | We need int32 in pinned memory before the H2D copy. |
| uint32 → float32 cast (values)| CPU      | Same reason.                                        |
| Per-chunk column shuffle      | CPU      | Cheap; just permutes indices into the chunk.        |
| Pack batch into pinned host   | CPU (T_BB) | Has to be host-side before the async copy.        |
| H2D copies (3 of them)        | PCIe (loader stream) | Async; overlaps with compute.            |
| Log-normalization             | GPU (loader stream) | Memory-bound; H100 doesn't notice it because we have prefetch slack. CPU has better things to do (decode). |

## Log-normalization on GPU: the contention question

Putting lognorm on the GPU does compete with the trainer for HBM bandwidth and SMs. But because we prefetch 4 batches ahead, the lognorm of batch N+3 only has to finish before the trainer reaches batch N+3 — about 90 ms of slack on a 30 ms step. A lognorm of one ~512-cell batch is a sub-millisecond kernel. So in practice it runs in the gaps and is effectively free. The only scenario where this bites is if every batch's lognorm gets bigger than the trainer step; we'll measure, but that's not a realistic regime for this data.

The alternative — lognorm on the CPU during the batch build — would steal cores from decode, which IS the bottleneck. So GPU lognorm wins.

## Concat vs pointer-list

Two ways to organize a decoded chunk:

- **CONCAT_HOST**: glue all decoded files into one big CSC matrix, shuffle a single column-permutation, batch-build is a tight contiguous gather.
- **POINTER_LIST**: keep each decoded file separate, the column permutation is `(file_idx, local_col)` pairs, batch-build does scattered reads.

CONCAT_HOST is simpler and likely faster for the inner loop (better cache behavior on the gather). POINTER_LIST avoids the concat step (a one-time cost per chunk). Which wins depends on chunk size. **Open question** — we'll time it. Default to CONCAT_HOST until proven otherwise.

## Sizing the buffers

Each slot has pinned host buffers and device buffers sized to "the largest batch we've ever seen." First time a batch exceeds the current size, we reallocate (free + alloc) all four slots' buffers. Initial guess based on `batch_size × max_nnz_per_cell × 1.2` so the first epoch typically doesn't reallocate. Subsequent epochs never do.

We do NOT allocate per-batch. `cudaMallocHost` is too slow for a hot path.

## Memory budget

For 100k features, batch=512, ~10% nnz → ~5M nnz per batch worst case → ~40 MB pinned + ~40 MB device per slot → ~160 MB pinned + ~160 MB device for a full ring. Trivial on 128 GB / 80 GB H100. Two simultaneous loaders (future multi-species) double these and still fit easily.

The decoded chunk is the bigger memory user. Using the measured mean of 96 MB decoded per file:

| config                                 | host memory |
|----------------------------------------|-------------|
| 1 loader, chunk=64, 2-deep chunk queue | 2 · 64 · 96 MB ≈ 12 GB |
| 1 loader, chunk=128                    | 2 · 128 · 96 MB ≈ 25 GB |
| 2 loaders, chunk=64 each               | 2 · 12 GB = 24 GB |

All comfortable inside 128 GB. We will default to `chunk_size = 64`, leaving plenty of room for two-species or larger chunks later.

## Shuffling

Two levels, both seeded:
1. At `begin_epoch()`: shuffle the file path list with the loader's `mt19937`. Single-threaded, on the main thread, deterministic.
2. Inside each chunk: shuffle the column permutation with a sub-seed drawn from the same RNG. T_BB does this on its own thread so the main thread RNG isn't touched after `begin_epoch()` returns.

This gives reproducibility (same seed → same epoch order) without the threads racing on RNG state.

## Public API (sketch)

```cpp
DataLoader(paths, chunk_size, batch_size, rng, policy, ring_depth=4, n_concurrent_loaders=1);
void begin_epoch();        // shuffle file order, reset, unblock workers; returns immediately
bool next_batch(SparseBatch* out);   // blocks if no batch ready; returns false at epoch end
int  m() const;
cudaStream_t loader_stream() const;
```

`SparseBatch` is `m`, `B`, `nnz`, three device pointers, a `cudaEvent_t`, and an `eof_after` flag. The pointers stay valid until the 4th subsequent `next_batch()` call.

## Lifecycle

- One DataLoader, lives for all training. `begin_epoch()` called once per epoch.
- Drop-last per chunk: any leftover columns at the end of a chunk that don't fill a full batch are skipped. (If we ever need them, we re-include them in the next epoch since file order reshuffles.)
- EOF: T_BB marks `eof_after = true` on the last batch; the next `next_batch()` after that returns false.
- Destruction: set a stop flag, notify all condvars, join both worker threads, destroy events, free buffers. Safe even mid-epoch.
- Errors: if `read_1pz` throws, print the path and the error to stderr and `std::terminate()`. Partial-epoch corruption is silent and worse than a crash.

## Future-proofing for multi-species

Two DataLoader instances on the same GPU (one per species, trainer alternates) need:
- Independent CUDA streams. Each loader makes its own. ✓
- Independent worker thread pools. Each loader spawns its own T_CL/T_BB. ✓
- Shared RNG safety: each loader holds its own `mt19937&`; the caller is responsible for not constructing/begin_epoch'ing two loaders concurrently from the same RNG. ✓ (just a doc requirement)
- **Alternation between species is handled by the `Ring` class** (see below), which can multiplex over multiple loaders so the trainer asks one place for the next batch and the Ring picks the right species according to a policy. This means the trainer code does NOT change when we add species — only the Ring's policy does.

---

# The `Ring` class

One class owns all batch-slot bookkeeping, all trainer-wait statistics, and (when configured for it) the species alternation policy. Each species' DataLoader has its own producer-side state inside the Ring; the trainer pulls from the Ring as a single source.

## Conceptual layout

```
   Ring
   │
   ├─ Lane 0 (species "human")        ├─ slots: [FREE|READY|BUILDING|CONSUMED] × ring_depth
   │                                  ├─ producer: DataLoader_0's T_BB
   │                                  └─ free_count / ready_count
   │
   ├─ Lane 1 (species "mouse")        └─ … same shape
   │
   ├─ alternation policy             ─→ picks which lane the next pull comes from
   ├─ stall counters per lane         ─→ incremented when trainer waited
   └─ single shared mutex + condvar
```

With one species (today), there's just one lane and no policy logic to run. With multiple species (future), the Ring multiplexes.

## Public API (sketch)

```cpp
enum class AlternationPolicy {
    SINGLE,            // exactly one lane, no choice (default)
    ROUND_ROBIN,       // cycle through lanes in order
    WEIGHTED,          // by configured weight (e.g. proportional to dataset size)
    CALLER_CHOOSES     // trainer passes a lane id to next_batch()
};

class Ring {
public:
    Ring(int n_lanes, int ring_depth, AlternationPolicy policy);
    ~Ring();

    // ---- producer side (called by T_BB of each loader) ----
    // Block until a FREE slot is available in this lane. Returns slot index.
    // Updates SlotState FREE → BUILDING.
    int  acquire_free(int lane);

    // Mark slot ready for consumption. BUILDING → READY. Notifies trainer.
    void publish_ready(int lane, int slot_idx, bool eof_after);

    // ---- consumer side (called by trainer) ----
    // Block until *some* lane chosen by policy has a READY slot.
    // Returns (lane, slot_idx). READY → CONSUMED. Returns false on epoch end
    // for all lanes.
    bool acquire_ready(int* out_lane, int* out_slot);

    // For CALLER_CHOOSES policy only: pull from a specific lane.
    bool acquire_ready_from(int lane, int* out_slot);

    // Mark slot recyclable. CONSUMED → FREE. Notifies producer.
    void release_consumed(int lane, int slot_idx);

    // ---- statistics ----
    struct Stats {
        uint64_t trainer_waits;         // # times trainer blocked in acquire_ready
        uint64_t trainer_wait_ns_total; // total time trainer spent blocked
        uint64_t producer_waits;        // # times producer blocked on FREE
        uint64_t producer_wait_ns_total;
        uint64_t batches_published;
        uint64_t batches_consumed;
    };
    Stats stats(int lane) const;        // per-lane
    Stats stats_total() const;          // summed across lanes
    void  reset_stats();

    // ---- epoch lifecycle ----
    // Called by Loader.begin_epoch() to reset per-lane EOF state.
    void begin_epoch(int lane);

    // Set per-lane weight for WEIGHTED policy.
    void set_weight(int lane, double w);
};
```

## Implementation sketch

- **One mutex, one condvar pair** shared across all lanes (the lane count is small — 1 or 2 today, never more than handful). This keeps the implementation simple and avoids cross-lane deadlock concerns.
- Each lane holds a `vector<Slot>` of size `ring_depth`, plus its own `free_count` and `ready_count`.
- The trainer wait in `acquire_ready` checks `any-lane-has-ready OR all-lanes-EOF`. The policy picks among lanes with ready slots.
- **Stall measurement.** Inside `acquire_ready`, we record a wall-clock timestamp on entry and again on exit. If the condvar's predicate was already true on entry (no wait), `trainer_waits` is *not* incremented. If the condvar actually slept, `trainer_waits++` and the elapsed time is added to `trainer_wait_ns_total`. This is the single most important loader-health metric: in steady state it should be **zero**.
- **Alternation policies.**
  - `SINGLE`: only lane 0 exists. `acquire_ready` is the same as the original ring.
  - `ROUND_ROBIN`: keeps a `next_lane_` counter; if that lane has no READY slot but another does, falls through to the next lane (we do NOT block waiting for a specific lane when another could serve us). This prevents one slow loader from idling the trainer.
  - `WEIGHTED`: like round-robin, but the cycle is drawn from a precomputed weighted schedule (e.g. lane 0 appears 3× for every 1× lane 1).
  - `CALLER_CHOOSES`: the trainer explicitly names a lane; useful for evaluation or for a curriculum schedule. Will block on just that lane.
- **Per-lane EOF handling.** A lane is "done" once its loader has published a slot with `eof_after = true` *and* the trainer has consumed it. When all lanes are done, `acquire_ready` returns false.
- **Shutdown.** Setting `stopped_ = true` and broadcasting wakes every thread; both `acquire_free` and `acquire_ready` check `stopped_` in their predicates and exit gracefully.

## Why a single Ring instead of one per loader

With one Ring per loader plus a wrapper on top to alternate, every trainer pull is two condvar checks (the wrapper checks each loader). With one Ring multiplexing, it's one. More importantly, the **stall counter has to live somewhere global** — it's a trainer-side metric ("did the trainer wait?"), not a loader-side metric. The Ring is the right home for it.

The DataLoader stays the same code-wise: it just gets handed a `Ring*` and a `lane_id` at construction, and its T_BB calls `acquire_free(lane_id)` / `publish_ready(lane_id, …)` instead of using internal counters.

## What we'll measure before finalizing

1. ~~**`read_1pz` cold/warm decode time**~~ — **done**, see [bench_read_1pz.cpp](bench_read_1pz.cpp) and [bench_read_1pz.slurm](bench_read_1pz.slurm). Conclusions baked into this plan.
2. **CONCAT_HOST vs POINTER_LIST** batch-build cost — small standalone bench, after the DataLoader is wired up.
3. **Trainer-wait latency benchmark** — the main loader-health benchmark; see below.
4. **Stall counter** in production: built into the Ring (`stats().trainer_waits`). If this is ever > 0 in steady-state training, the loader is the bottleneck and we go back to optimization.

---

# The unit test / benchmark: `bench_loader_latency`

A standalone executable that measures **how long the trainer waits for the next batch** — the headline metric for whether we've succeeded.

## What it measures

Three numbers per run:

1. **Time between batches.** Wall-clock from when `next_batch()` returns for batch N to when it returns for batch N+1, with a **simulated trainer step** of configurable duration (`--step-ms`) in between. The simulation is just `std::this_thread::sleep_for(step_ms)` plus a `cudaStreamWaitEvent` on the batch's event so we exercise the GPU fence path too.
2. **Trainer wait time per batch.** Wall-clock spent *inside* `next_batch()`, i.e. the time the trainer was blocked. This is the number we want at zero.
3. **Steady-state vs. warmup.** Reports both. The first 4 batches always have some warmup; we discard them from steady-state stats but report them separately.

## How it works

```
DataLoader(paths=<chosen set>, chunk_size, batch_size=512)
begin_epoch();

loop N times:
    t0 = now()
    next_batch(&b)
    t1 = now()                          // wait_time = t1 - t0
    cudaStreamWaitEvent(stream, b.ready_event)
    cudaStreamSynchronize(stream)        // force the lognorm to truly complete
    sleep_for(step_ms)                   // simulated trainer compute
    t2 = now()                          // batch_to_batch = t2 - prev_t2
```

## CLI

```
bench_loader_latency \
    --paths <dir-or-listfile> \
    --chunk-size 64 \
    --batch-size 512 \
    --step-ms 30 \
    --n-batches 500 \
    --decode-threads 16 \
    --ring-depth 4
```

## What it prints

```
# steady-state (excluding first 4 batches)
batches              : 496
mean batch-to-batch  : 30.12 ms   (target: == step-ms == 30.0)
p50 / p95 / max      : 30.05 / 30.4 / 31.1 ms
mean trainer-wait    : 0.005 ms   (target: ~0)
p50 / p95 / max wait : 0.0 / 0.02 / 1.4 ms
trainer_waits count  : 3 / 496    (target: 0)
producer_waits count : 491 / 500  (expected: high — loader pacing itself)

# warmup (first 4 batches)
mean batch-to-batch  : 215 ms
mean trainer-wait    : 185 ms     (expected — cold decode)
```

The pass/fail criteria:

- **`trainer_waits == 0` in steady state** — the GPU never had to wait. This is the hard requirement.
- **`mean batch-to-batch ≈ step-ms`** — the loader is keeping pace with the simulated trainer.
- `producer_waits` being high is **good** — it means the loader is throttled by the ring being full (i.e. faster than the trainer needs).

## Sweeps

The script form (`bench_loader_latency.slurm`) runs the binary across:

- `step-ms ∈ {10, 30, 100}` — simulates fast/normal/slow trainers.
- `chunk-size ∈ {32, 64, 128}` — finds the sweet spot.
- `ring-depth ∈ {2, 4, 8}` — confirms 4 is enough (it should be).
- `decode-threads ∈ {8, 16, 32}` — confirms 16 is enough.

For each combo, prints a one-line summary: `step=30 chunk=64 ring=4 dec=16 trainer_waits=0 mean_wait=0.005ms`. Anything with `trainer_waits>0` at `step-ms=30` is a failing config.

## Why this test matters

It's the **only** test that proves the headline goal. Unit tests on the Ring class (state transitions, EOF, stats accuracy) are necessary but not sufficient. This benchmark is what we run after every loader change to know whether we broke the latency budget. It also gives us a tunable knob: when we add a new feature (lognorm variant, larger features, two species), we re-run and check the stall count is still zero.

## Open questions

- ~~Exact `read_1pz` decode time and parallel scaling~~ — **resolved.** 16 threads, scaling falls off past that. Mean 745 ms/file (median 300 ms, p95 3050 ms).
- ~~Whether to double-buffer chunks~~ — **resolved.** Required, because chunk decode > chunk training. 2-deep chunk queue is now the default.
- CONCAT_HOST vs POINTER_LIST — pick based on a follow-up benchmark once the loader is wired up.
- Whether the long tail (p95 = 3 s, max = 3 s on 800 MB files) needs a per-file size cap or pre-sort. **Tentatively no** — double-buffering plus 16-thread parallel decode should hide individual outliers. We'll watch `bench_loader_latency` for stalls at chunk boundaries and revisit.
- Whether to use `cuFile`/GDS to skip CPU decode entirely. Likely not — decode is the format-defining step, not a copy. Noted but not pursued.
