# Implementation Progress

Crash-recovery checkpoint. If wifi drops and the agent restarts, resume from the first incomplete step.

Source of truth: [PLAN.md](PLAN.md). Every implementation prompt to Haiku must reference PLAN.md sections.

---

## Status legend

- [ ] not started
- [~] in progress (dispatched, awaiting Haiku return)
- [x] done — code written to disk

## Build status (separate tracker)

The user has stated **NEVER build, NEVER run.** All compile/test work is the user's job. We only write source.

---

## Step 1 — Ring class

- [x] Write `ring.h` + `ring.cu` (or `ring.cpp` — Haiku picks)
- Files created: [ring.h](ring.h), [ring.cu](ring.cu)
- get_errors clean on both files (compile errors not yet known — user builds)

## Step 2 — DataLoader rewrite

- [x] Rewrite `gpu_data_loader.h` + `gpu_data_loader.cu`
- Files: [gpu_data_loader.h](gpu_data_loader.h), [gpu_data_loader.cu](gpu_data_loader.cu)
- Constructor takes `Ring* ring, int lane_id`. `next_batch()` removed (trainer talks to Ring directly).
- 2-deep chunk queue. Default `omp_threads=16`.
- get_errors clean.
- Reference: PLAN.md § "Two background threads", § "Where the work actually happens", § "Sizing the buffers"
- Constructor now takes `Ring* ring`, `int lane_id` instead of internal ring state
- Two threads: T_CL (chunk loader, 16 OpenMP threads default), T_BB (batch builder)
- **2-deep chunk queue** (mandatory double-buffering — see PLAN.md timing analysis)
- T_BB writes into ring via `ring->acquire_free(lane_id)` / `publish_ready(lane_id, slot, eof_after)`
- The ring OWNS the slots (pinned/device buffers, event) — Loader does NOT
  - so the H2D+lognorm enqueue logic in T_BB writes into pointers it gets from the Ring slot
- Lognorm on GPU on loader_stream (record event after enqueue)
- uint32→int32/float32 saturate cast on CPU
- CONCAT_HOST default policy
- begin_epoch resets cursors, shuffles file order, draws per-chunk sub-seed, calls `ring->begin_epoch(lane_id)`
- EOF: T_BB sets eof_after on last batch's slot
- Errors: print + std::terminate
- Stall counter NOT in DataLoader (lives in Ring)

## Step 3 — bench_loader_latency

- [x] Write `bench_loader_latency.cpp`
- File: [bench_loader_latency.cpp](bench_loader_latency.cpp)
- get_errors clean.
- Reference: PLAN.md § "The unit test / benchmark"
- **No `--step-ms`** — measure the floor, don't accept one
- CLI: `--paths`, `--chunk-size`, `--batch-size`, `--n-batches`, `--decode-threads`, `--ring-depth`
- Loop: `next_batch` → `cudaStreamWaitEvent` → `cudaStreamSynchronize` → record timestamps → repeat
- Reports: steady-state mean/p50/p95/max batch interval, min trainer step ms (=max interval), trainer-wait stats, producer-wait stats, warmup separately
- Warmup = first 4 batches (excluded from steady-state)

## Step 4 — bench_loader_latency.slurm

- [x] Write `bench_loader_latency.slurm`
- File: [bench_loader_latency.slurm](bench_loader_latency.slurm)
- 27-combo sweep, summary CSV at end.
- Pattern after `bench_read_1pz.slurm`
- Sweep: chunk-size {32, 64, 128} × ring-depth {2, 4, 8} × decode-threads {8, 16, 32}
- One-line summary per combo (parse from bench output)
- Pick 200 files via `find ~/quant -mindepth 3 -maxdepth 3 -name '*.1pz' | shuf -n 200`

## Step 5 — Makefile updates

- [x] Add targets for `ring.o`, updated `gpu_data_loader.o`, `bench_loader_latency`
- Added: `ring.o`, `bench_loader_latency{,o,.dlink.o}` targets
- Updated: `gpu_data_loader.o`, `main_gpu`, `main_gpu.dlink.o`, `bench_loader`, `bench_loader.dlink.o`, `tests/validate/validate{,.dlink.o}`, `clean`
- Reference: existing patterns in Makefile (lines for `bench_loader`, `gpu_data_loader.o`)

## Step 6 — Sanity review (agent, not Haiku)

- [x] Read each generated file briefly, confirm it matches PLAN.md
- [x] Report file paths + one-line description back to user
- Found 1 mismatch: `Ring::SlotView` declared `int32_t**` (pointer-to-pointer) but DataLoader used `slot_view.h_col_ptr` directly. Root cause: my Ring prompt had self-contradictory guidance. Dispatched a fix to Haiku; ring.h/ring.cu now use flat `int32_t*` with snapshot semantics + call-slot_view-again-after-ensure_capacity contract. Confirmed.

---

## Notes / decisions made during implementation

- Ring SlotView is a **snapshot** of slot pointers + capacities. After `ensure_capacity()`, the caller must call `slot_view()` again to get fresh pointers. (Documented in ring.h.)
- Chunk queue depth = 2, hardcoded constant `kChunkQueueMaxSize` in `gpu_data_loader.cu`.
- Default `omp_threads = 16` for decode (PLAN.md analysis: 16 threads is the sweet spot).
- Workers stay alive across epochs; `begin_epoch()` resets cursors and notifies via condvar.
- `Ring::shutdown()` is NOT called from DataLoader destructor (Ring is externally owned, may serve multiple loaders).
- Saturate casts for uint→int32 / uint→float32 preserved from original gpu_data_loader.cu.

## Open follow-ups for user

- **Build is the user's job.** Run `make ring.o gpu_data_loader.o bench_loader_latency` on the remote machine. Fix any compile errors that come back — these are most likely missing includes or signature mismatches at this stage.
- **Trainer code update.** Whatever currently calls `loader.next_batch(&b)` must be updated to talk to the Ring directly: `ring.acquire_ready(&b, &lane, &slot)` → use → `ring.release_consumed(lane, slot)`. Affected files likely include `main_gpu.cpp` and `tests/validate/validate.cpp`. Not done in this round.
- **Run `bench_loader_latency.slurm`** to actually measure the floor and pick optimal chunk/ring/decode params.
- Once trainer is updated, decide whether `Ring` is constructed by `main_gpu.cpp` (which then passes it to one or more DataLoaders) — currently DataLoader takes `Ring*` so the caller owns it.
