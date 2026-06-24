# DataLoader Unit-Test Plan

Goal: build a focused unit-test suite that catches *fundamental* bugs in
`DataLoader` (file `gpu_data_loader.cu`) — the kinds of issues that would
silently corrupt training even when the pipeline looks healthy.

Scope is the **producer side** only: from a list of `.1pz` files to batches
visible on the consumer side of the `Ring`. The trainer / autoencoder is out
of scope; we drive the loader from a test consumer that pulls batches off the
ring, copies them back to host, and inspects them.

---

## 1. Fundamentals a data loader must guarantee

Below is the checklist of "things every data loader ought to be doing".
Each item is followed by the test that exercises it. Tests are grouped by
category and ordered roughly from cheapest to most expensive.

### A. Shape & layout correctness

A1. **Reported batch dimensions match the contract.** Every published
    `SparseBatch` has `m == loader.m()`, `B == batch_size`, and `nnz ==
    h_col_ptr[B] - h_col_ptr[0]`. Negative or out-of-range fields are bugs.

A2. **CSC structural invariants.**
    - `col_ptr[0] == 0`
    - `col_ptr` is monotonically non-decreasing
    - `col_ptr[B] == nnz`
    - Every `row_idx[k]` satisfies `0 <= row_idx[k] < m`
    - Within each column, `row_idx` values are unique (no duplicates) and
      sorted ascending — this is what `read_1pz` is expected to produce and
      what downstream sparse kernels assume.

A3. **No NaN / Inf and no negatives in values.** After log-normalization,
    all values must be finite and `>= 0` (log1p of non-negative input is
    non-negative).

### B. Numerical correctness (log-normalization)

B1. **Empty columns survive.** A column with `col_ptr[i] == col_ptr[i+1]`
    contributes nothing and must not crash. Test by constructing a synthetic
    `.1pz` (or substituting at the host level) with at least one empty
    column.

B2. **Lognorm formula matches the reference.** Pull a batch, copy
    `values` back to host, recompute the expected result in CPU code:

    ```
    s = sum(values_orig[col])
    if s > 0: values[k] = log1p(values_orig[k] * 10000 / s)
    else:     values[k] = 0
    ```

    Compare against the GPU output to within `1e-5` relative tolerance.
    Bug-class caught: wrong scaler, missing `log1p`, normalization across
    wrong axis.

B3. **Scaler is exactly `10000.0f`.** Implicit in B2 — the comparison must
    fail if the scaler is changed by even 1%. Keep this as an asserted
    constant in the test so a refactor that drifts the scaler is caught.

### C. Data coverage & shuffling

C1. **Every column from every file is delivered exactly once per pass.**
    Build a synthetic dataset with $N$ files where every file has a unique
    "signature" column (e.g. set value at row 0 to a per-file id). Run one
    pass. Collect signatures of all delivered batches. The multiset of
    signatures must equal the multiset of source signatures, modulo the
    tail-drop rule (see C2). Catches: lost files, duplicated columns,
    pointer-list/concat mismatches, off-by-one in `col_perm`.

C2. **Tail drop is bounded.** The loader drops any partial batch at the end
    of a chunk. Assert that for each chunk, `cols_delivered ==
    floor(chunk_cols / batch_size) * batch_size`, and that the dropped
    count is `< batch_size`. Catches: silent over-drop, accidental tail
    inclusion that produces a short batch the consumer assumes is full.

C3. **`chunk_end` flag fires exactly once per chunk.** Track `chunk_end`
    across all batches in a pass. Count of `true` values must equal number
    of chunks. Each `true` must coincide with the last batch of its chunk.

C4. **Shuffling actually shuffles.** With two different RNG seeds, the
    *sequence* of column signatures differs (with overwhelming probability),
    but the multiset of signatures is identical. With the same seed, the
    sequence is identical (determinism — see E1).

C5. **Multi-pass auto-rolling.** Consume enough batches to wrap the file
    list 2–3 times. Check that `ring->lane_pass(lane)` increments correctly,
    that the second pass is a different shuffle than the first, and that
    coverage (C1) holds on each pass independently.

### D. Concurrency / Ring contract

D1. **No corruption under producer pressure.** Run with `ring_depth=2`,
    small batches, fast consumer. Verify A1–A3 still hold on every batch.
    Catches: producer writing to a slot the consumer still holds (state-
    machine bug), missing `cudaEventRecord` causing stale device reads.

D2. **Slot lifecycle is respected.** The test consumer copies device
    buffers to host using the slot's `ready_event` (`cudaStreamWaitEvent`).
    Then call `release_consumed`. Assert that immediately re-acquiring
    the same slot does not show stale data — overwrite host copies with a
    sentinel and re-read after `release_consumed` to confirm.

D3. **Capacity growth is sound.** Construct a sequence where successive
    batches in a single slot have monotonically increasing `nnz` (forces
    `ensure_capacity` reallocation). Confirm A1–A3 hold across the
    realloc. Catches: snapshot-pointer staleness after `ensure_capacity`
    (the loader is supposed to refresh `slot_view`).

D4. **Clean shutdown.** Construct loader → `start()` → consume a handful
    → destructor. Must return within a bounded wall time (e.g. 2 s) and
    not leak threads or CUDA resources (check `cudaMemGetInfo` before and
    after; allow slack for one-shot allocations).

D5. **Construction-before-start is safe.** After the constructor returns
    but before `start()` is called, the worker threads must not deadlock
    or push anything to the ring. Test by sleeping 100 ms before `start()`
    and then verifying batches still flow.

### E. Determinism & RNG

E1. **Bit-identical replay.** Two `DataLoader` instances with the same
    seed, same file list, same chunk/batch size, same policy must produce
    bit-identical batch sequences (same `col_ptr`, `row_idx`, `values`).
    Catches: nondeterministic shuffles, OpenMP-induced ordering bugs in
    the concat path.

E2. **Seed sensitivity.** Different seeds must produce different per-batch
    column orderings; otherwise the shuffle is broken.

### F. Policy parity

F1. **`CONCAT_HOST` and `POINTER_LIST` are equivalent.** With the same
    seed, both policies must deliver the same multiset of columns (the
    per-batch *order* may differ because permutation generation is shaped
    differently, but the columns themselves and their CSC content must
    match). Sort-then-compare on a per-column signature.

### G. Input validation

G1. **Bad-arg rejection.** Constructor must throw on: empty file list,
    `chunk_size <= 0`, `batch_size <= 0`, null ring, `lane_id` out of
    range. One assertion per case.

G2. **Feature-count mismatch is fatal, not silent.** Build a 2-file
    dataset where file 2 has a different `m`. Expect the loader to abort
    (we already `std::terminate` on this); the test uses a death-test
    wrapper (fork + assert non-zero exit).

### H. Edge cases

H1. **Single file, single batch.** `files=[f]`, `chunk_size=1`,
    `batch_size = f.n_cols`. Exactly one batch, `chunk_end=true`.

H2. **Batch larger than chunk.** `batch_size > chunk_cols` ⇒ zero batches
    per chunk (everything is tail-dropped). Pass should complete with no
    batches delivered but workers must not hang. (This is degenerate but
    the loader should not crash.)

H3. **All-zero column.** A column whose values all sum to zero must come
    out with values left at zero (the `sum > 0` guard in the kernel).
    Build a synthetic file or patch via the POINTER_LIST path.

H4. **Saturating casts trigger only when they should.** Feed values
    `> 2^24` and indices `> INT32_MAX/2`; check that warnings fire (via
    captured stderr) but the loader continues.

---

## 2. Test harness design

```
tests/loader/
    test_loader.cpp         # gtest or simple assert-based main
    fixtures.h / fixtures.cpp
    synthetic_pz.cpp        # write a tiny .1pz with controllable content
    Makefile                # links against gpu_data_loader.o, ring.o, singlet
```

### 2.1 Synthetic dataset generator

A helper that writes a `.1pz` file with caller-specified `m`, `n`, and a
callable `value_at(file_id, col, row) -> uint32_t`. Used to plant
signatures (C1), all-zero columns (H3), and varying nnz (D3).

If writing `.1pz` is too invasive, an alternative is to bypass the file
reader entirely with a thin test seam: a `friend`-or-`#ifdef TESTING`
hook that lets the test inject a pre-decoded `ReadResult` per file path.
The plan below assumes the synthetic-file approach because it tests the
full pipeline including decode.

### 2.2 Test consumer

A small function that owns a `Ring`, calls `acquire_ready`, copies the
slot's device buffers back to host via a dedicated stream that
`cudaStreamWaitEvent`s on the slot's `ready_event`, runs the assertions
for the current test, and calls `release_consumed`.

### 2.3 Death-test for fatal paths (G2)

The loader uses `std::terminate` on decode failure. Use `fork()`; the
child constructs the loader and `start()`s it; the parent `waitpid()`s
and asserts a non-zero exit status within a timeout.

### 2.4 Resource accounting (D4)

Wrap each test in a fixture that snapshots `cudaMemGetInfo` and
`/proc/self/status` (VmRSS) before construction and after destruction.
Allow a fixed per-test budget (e.g. ≤ 16 MB leak tolerance).

---

## 3. What we are NOT testing (and why)

- **Throughput / latency.** Already covered by `bench_loader.cpp` and
  `bench_loader_latency.cpp`. Unit tests should be fast and deterministic;
  performance regressions belong in the existing bench suite.
- **Multi-lane scheduling policies.** That belongs to a `ring` test plan.
  Here we use a 1-lane SINGLE ring.
- **The `read_1pz` decoder itself.** Tested upstream in `singlet`.

---

## 4. Suggested implementation order

1. Synthetic `.1pz` writer + minimal one-batch smoke test (covers A1, A2).
2. Coverage test (C1) — biggest bug-class for the least effort.
3. Determinism + seed sensitivity (E1, E2).
4. Lognorm reference comparison (B1, B2, B3, H3).
5. `chunk_end` and tail-drop (C2, C3).
6. Policy parity (F1).
7. Bad-arg + feature-mismatch (G1, G2).
8. Concurrency / capacity / shutdown (D1–D5).
9. Edge cases (H1, H2, H4) and saturate-cast warnings.

Stop after step 5 if pressed for time — those five items alone catch
the majority of silent-corruption bugs.
