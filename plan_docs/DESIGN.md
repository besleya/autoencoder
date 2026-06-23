# GPU DataLoader Redesign — scratch doc, delete after implementation

---

## 1. Public API

```cpp
enum class ConcatPolicy { CONCAT_HOST, POINTER_LIST };

// Lifetime: pointers valid until the ring_depth-th subsequent next_batch()
// call on the same loader (default K=4). Trainer holds at most 1 live batch
// per loader at a time, so K=4 is always safe.
struct SparseBatch {
    int m, B, nnz;
    const int32_t* d_col_ptr;   // device, length B+1
    const int32_t* d_row_idx;   // device, length nnz
    const float*   d_values;    // device, length nnz
    cudaEvent_t    ready_event; // signaled after H2D + lognorm; owned by slot
    bool           eof_after;   // true on the last batch of epoch
};

class DataLoader {
public:
    // rng consumed only in begin_epoch() (file shuffle + per-chunk sub-seed);
    // never touched by worker threads.
    DataLoader(const std::vector<std::string>& paths,
               int chunk_size, int batch_size,
               std::mt19937& rng,
               ConcatPolicy policy = ConcatPolicy::CONCAT_HOST,
               int ring_depth = 4,
               int n_concurrent_loaders = 1);
    ~DataLoader();

    // Shuffle paths, reset cursors, seed per-chunk sub-RNG, unblock T_CL.
    // Returns immediately — does NOT wait for first batch.
    void begin_epoch();

    // Returns false when epoch exhausted. Trainer pattern:
    //   next_batch(&b);
    //   cudaStreamWaitEvent(trainer_stream, b.ready_event, 0); // GPU fence only
    //   net.forward(b, trainer_stream);
    bool next_batch(SparseBatch* out);

    int          m()             const;
    cudaStream_t loader_stream() const;
};
```

---

## 2. SparseBatch lifecycle & event ownership

`cudaEvent_t` created per slot at construction with `cudaEventDisableTiming`.
Batch-builder (T_BB) calls `cudaEventRecord(slot.ready_event, loader_stream_)`
after the 3 H2D copies + lognorm kernel are enqueued. GPU signals the event
when all three complete. Trainer calls `cudaStreamWaitEvent(trainer_stream,
b.ready_event, 0)` — pure GPU-side fence, zero CPU stall. On slot recycle,
T_BB simply re-records the event on the same object; CUDA overwrites the prior
signal. No `cudaEventSynchronize` needed before re-recording.

---

## 3. Internal architecture

```
  .1pz files ──▶ [ T_CL: Chunk-loader ]
                      │ decoded ChunkData (1-deep ChunkQueue)
                      ▼
                 [ T_BB: Batch-builder ]
                      │ fills pinned host bufs per shuffled permutation
                      │ on LAST col of chunk N: signals T_CL → start N+1  ← CPU trigger
                      │ cudaMemcpyAsync ×3 + lognorm + EventRecord
                      ▼
          ┌─────────────────────────────────────────┐
          │  Ring buffer (depth=4)                  │
          │  slot[0..3]: h_col_ptr/row_idx/values   │
          │              d_col_ptr/row_idx/values   │
          │              ready_event  state  nnz/B  │
          └──────────────────┬──────────────────────┘
                             ▼  next_batch()
                      Trainer (main thread)
              cudaStreamWaitEvent(trainer_stream, b.ready_event, 0)
```

Slot state machine: `FREE → BUILDING → READY → CONSUMED → FREE`
(`BUILDING` = T_BB owns slot, packing pinned bufs + enqueuing H2D;
`READY` = event recorded, GPU work may still be in flight;
`CONSUMED` = trainer has taken pointer; recycled K=ring_depth calls later.)

T_CL runs 1 chunk ahead of T_BB via the 1-deep ChunkQueue. T_CL blocks when
the queue is full; T_BB blocks when the queue is empty. T_BB sends the
"chunk consumed" signal to T_CL the instant it finishes copying the last
column of chunk N into a ring slot — before the slot's H2D is dispatched.

---

## 4. Synchronization primitives

**mutex + condition_variable** (C++17, matches existing codebase).

Ring: `free_count_` (init=ring_depth) and `ready_count_` (init=0) under one
`ring_mtx_`. T_BB waits on `free_count_ > 0`; `next_batch()` waits on
`ready_count_ > 0`. Both call `ring_cv_.notify_all()` after updating counts.

ChunkQueue: separate mutex+condvar pair, 1-slot. T_CL waits if full; T_BB
waits if empty; T_BB signals T_CL (`chunk_consumed_cv_`) after last column of
chunk N is assembled into a ring slot (not after H2D).

---

## 5. CUDA stream layout per loader

One non-blocking stream `loader_stream_` at **default priority**. (High
priority would steal SMs from the trainer during lognorm — wrong tradeoff.)

```
loader_stream_:  [H2D col_ptr][H2D row_idx][H2D values][lognorm<<<B,256>>>]
                 [EventRecord(ready_event)]
                 [H2D col_ptr][H2D row_idx][H2D values][lognorm] ...  (next batch)
```

All H2D + lognorm for every batch in this loader are serialized on
`loader_stream_`. Two concurrent loaders have independent streams; their PCIe
transfers and lognorm kernels overlap naturally. The trainer stream is external;
the only coupling is `cudaStreamWaitEvent`.

---

## 6. ConcatPolicy implementations

### CONCAT_HOST (T_CL thread)
Decode each file with `singlet::pz::read_1pz()`. Verify `r.m` matches.
Build unified col_ptr via prefix sum: `col_ptr[col_off[i]+c+1] =
r.indptr[c+1] + nnz_off[i]` (uint32→int32 cast). Concatenate row_idx/values
(with saturate cast) into one contiguous pageable buffer.
T_BB assembles each batch by gathering `batch_size` contiguous columns from
the shuffled permutation: compute per-column nnz from col_ptr, memcpy ranges
into pinned host bufs, fix up col_ptr to be 0-based.

### POINTER_LIST (T_CL thread)
Decode each file into a `DecodedFile` struct (separate int32/float arrays,
pageable). Build permutation as `vector<pair<int,int>>` (file_idx, local_col)
covering all columns, then shuffle. T_BB iterates the batch's permutation
slice: for each `(fi, ci)`, copies `decoded[fi].row_idx[start..end]` and
`decoded[fi].values[start..end]` into pinned host bufs with running nnz
offset. More pointer indirection than CONCAT_HOST but avoids the concat step.
Both policies produce identical pinned CSC layout; H2D path is identical.

---

## 7. Pinned buffer sizing strategy

**Pre-allocated, grow-on-exceed (option a).** Each slot's pinned and device
buffers are sized to `max_batch_nnz_seen_` (updated under `ring_mtx_`). When a
new max is observed, reallocate all slot buffers (`cudaFreeHost` + `cudaMallocHost`
+ `cudaFree` + `cudaMalloc`). This is rare after epoch 1. Hot path has zero
allocation overhead. Per-batch `cudaMallocHost` (option b) costs ~100 µs/call
and serializes pinned VA space — unacceptable in the ring pump.

---

## 8. Lifecycle: begin_epoch and EOF

`begin_epoch()`: shuffle `file_paths_` into `epoch_order_`; draw `uint64_t
chunk_sub_seed_` from rng (T_BB uses this to seed its own `mt19937` for column
permutation); reset all ring counters and cursors; signal T_CL to start chunk 0.
Returns before any batch is ready.

EOF: T_CL sets `last_chunk_flag_` after pushing the final chunk. T_BB, after
building the last batch from the last chunk, sets `slot.eof_after = true` before
marking READY. `next_batch()` after consuming that slot returns false on the
next call (`ready_count_ == 0` + `eof_epoch_ == true`).

---

## 9. Error handling

If `read_1pz` throws in T_CL: print `[DataLoader] FATAL: <path>: <what>` to
stderr and call `std::terminate()`. Partial epoch data silently corrupts
gradients; there is no useful recovery. Do not forward exceptions across threads.

---

## 10. Two-loader concurrency notes

- `PinnedPool::acquire` wraps `cudaMallocHost`, which is thread-safe per CUDA docs.
- Each loader has its own `loader_stream_`; streams are independent.
- `read_1pz` thread-safety: presumed safe (independent file handles) — verify.
- RNG (`mt19937&` ref): only touched in `begin_epoch()` on the main thread;
  caller serializes the two `begin_epoch()` calls.
- Disk I/O: two T_CL threads reading different files — bandwidth contention
  only, no correctness issue.

OMP plan TBD by separate audit.

---

## 11. Files touched

- **Modify**: `gpu_data_loader.h`, `gpu_data_loader.cu`
- **Modify `Makefile`**: remove `load_pz.o data.o data-alt.o test_loader.o`
  from all link lines; drop `test_loader` and `alt_obj` targets; update
  `gpu_data_loader.o` deps; add `-I$(SINGLET_INCLUDE)` to its nvcc rule.
- **Delete**: `load_pz.h`, `load_pz.cpp`, `data.h`, `data.cpp`, `data-alt.cpp`,
  `test_loader.cpp`
- **`main_gpu.cpp`**: drop `#include "data.h"` / `"load_pz.h"`; replace
  `validate_1pz` peek with `DataLoader::peek_m(path)` static or inline read;
  replace `batch.stream` with `batch.ready_event`; add trainer stream; add
  `cudaStreamWaitEvent` per batch; update `ConcatPolicy` constructor arg.

---

## 12. Open questions

1. **RNG determinism**: column permutation is now generated in T_BB from a
   sub-seed drawn in `begin_epoch`. This breaks the old single-threaded RNG
   call sequence. If bit-for-bit reproducibility with the old loader matters,
   all permutations must be pre-generated in `begin_epoch` on the main thread
   (cost: `total_cols_per_epoch * 4` bytes, potentially hundreds of MB).

2. **`read_1pz` thread-safety**: need to confirm two T_CL threads (one per
   loader) can call it concurrently on different paths without internal races
   (global state, static buffers, etc.).

3. **int32 overflow**: `total_nnz` per chunk can exceed 2^31 with large
   chunk_size. Keep int32 + terminate-on-overflow check, or widen to int64
   internally and cast at H2D time?

4. **POINTER_LIST memory lifetime**: `ChunkData` must stay alive until T_BB
   finishes the last batch of that chunk. The 1-deep queue enforces this.
   Confirm the CONCAT_HOST unified buffers have the same lifetime guarantee
   (they do, by the same mechanism).

---

## 13. OpenMP plan

Makefile already has `-fopenmp` in `CXXFLAGS`, `LDFLAGS`, and `NVCCFLAGS`
(via `-Xcompiler`). No build changes needed.

### Per-loader thread budget

In the `DataLoader` constructor:
```cpp
int total = omp_get_max_threads();                       // respects OMP_NUM_THREADS
omp_threads_ = std::max(1, total / n_concurrent_loaders); // user tells us how many coexist
```
Inside any OMP region in T_CL, set explicitly via the `num_threads(omp_threads_)`
clause on each `parallel` directive (do not call `omp_set_num_threads` globally —
it's process-wide). With K loaders all budgeted to N/K, the total active OMP
worker count stays at N. If the user runs only one loader they pass
`n_concurrent_loaders = 1` (the default) and the loader uses all
`OMP_NUM_THREADS` threads. **No `omp_set_nested`** — we never nest.

### Step 1 — `read_1pz` over `chunk_size` files (T_CL)

Fuse with steps 2/3: one parallel region per chunk, each thread reads
its assigned files and immediately performs the cast/copy phase for those
files. Avoids a barrier between decode and cast.

```cpp
#pragma omp parallel for schedule(dynamic) num_threads(omp_threads_)
for (int i = 0; i < n_files; ++i) {
    files[i] = singlet::pz::read_1pz(paths[chunk_start + i]);
    // then: per-file cast into unified buffer (CONCAT_HOST)
    // or:   per-file cast into decoded[i] arrays (POINTER_LIST)
}
```
- `schedule(dynamic)`: file sizes vary.
- Exception handling: per-iteration `try { ... } catch (...) { atomic_flag fail; }`;
  T_CL checks the flag after the region and terminates per section 9.
- Expected speedup: 3–6× on 8 threads (zstd decompress is CPU-bound;
  disk caches usually absorb the I/O).

### Step 2 — CONCAT_HOST: prefix-sum then parallel copy

Two-phase. Phase 1 sequential (cheap, O(n_files)). Phase 2 fuses into the
Step 1 parallel region above once `nnz_off[]` and `col_ptr_off[]` are known.
Because Phase 1 needs each file's `r.nnz` (only known after decode), the
flow is:

```
parallel-for: read_1pz(i) → files[i]                  (Step 1, fused decode)
[implicit barrier]
sequential:   compute nnz_off[], col_ptr_off[] from files[*].nnz   (Phase 1)
parallel-for: copy+cast files[i] into unified buffer  (Phase 2)
```

So with CONCAT_HOST there are **two** parallel-for regions in T_CL per chunk,
separated by a tiny sequential prefix-sum. POINTER_LIST is **one** region
(decode and cast fused).

### Step 3 — POINTER_LIST: per-file cast (outer loop)

Parallelize over files, not over nnz within a file:
```cpp
#pragma omp parallel for schedule(dynamic) num_threads(omp_threads_)
for (int i = 0; i < n_files; ++i) {
    files[i] = singlet::pz::read_1pz(paths[chunk_start + i]);
    decoded[i].row_idx.resize(files[i].nnz);
    decoded[i].values.resize(files[i].nnz);
    for (size_t j = 0; j < files[i].nnz; ++j) {
        decoded[i].row_idx[j] = saturate_cast_u32_to_i32(files[i].indices[j]);
        decoded[i].values[j]  = saturate_cast_u32_to_f32(files[i].data[j]);
    }
    // similarly cast indptr → int32
}
```

### Step 4 — Batch-builder column gather

**Not parallelized.** `batch_size ~ 256`, per-column work ~1 µs;
OMP fork/join overhead exceeds the benefit. Keep sequential in T_BB.

### Step 5 — H2D + lognorm

CPU-async; nothing for OMP to do.

### Saturate-cast warning flag

Old static `bool warned` is a race. Replace with module-scope atomic:
```cpp
static std::atomic<bool> saturate_warned{false};
inline float saturate_cast_u32_to_f32(uint32_t v) {
    if (v > (1u << 24)) {
        bool expected = false;
        if (saturate_warned.compare_exchange_strong(expected, true)) {
            std::fprintf(stderr, "[DataLoader] WARNING: uint32 value %u "
                "exceeds 2^24; fp32 cast loses precision.\n", v);
        }
    }
    return static_cast<float>(v);
}
```
First thread to see overflow prints once; others fast-path the load and skip.

### `read_1pz` thread safety

Open question #2 still stands. Mitigation if not safe: serialize the
`read_1pz` call under a mutex but do the cast outside the mutex (still a net
win — decompression is most of the cost). Decide after checking the singlet
source.

### OMP regions inside `std::thread`

Well-defined since OMP 4.0. The OMP thread pool is process-wide; both T_CL
threads share it, which is exactly why the N/2 budget per loader is needed.
No `OMP_PROC_BIND` configuration assumed; default `false` is fine.
