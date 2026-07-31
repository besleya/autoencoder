# Batch

Transient abstraction of one training batch. Owns no memory of its own — it borrows pinned/device buffers from a `Slot` and a read-only view into a `Chunk`. Its whole job is: pick its columns out of the chunk, materialize them in contiguous host memory, CPU log-normalize them, then ship them to the GPU.

## Purpose

- Give downstream code (Translator, Autoencoder) a clean, self-describing handle.
- Own the "select columns from chunk → CPU log-norm → ship to GPU" pipeline as one call (`prepare()`).
- Carry species identity so `Translator::forward(batch)` can dispatch to the right per-species model.

## Supporting types

```cpp
// Chunk — a concatenation of one or more singlet::pz::ReadResult objects along
// columns (cells). DataLoader owns/produces this; Batch only ever reads from
// it and assumes it remains valid for Batch's entire lifetime (no defensive
// copying — DataLoader guarantees the chunk outlives every Batch drawn from it).
struct Chunk {
    int m = 0;                       // rows/features, same for the whole chunk
    int n = 0;                       // total columns across concatenated files
    std::vector<uint32_t> col_ptr;   // length n+1, cumulative nnz offsets (rebased across files)
    std::vector<uint32_t> row_idx;   // length nnz, row/gene index per nonzero
    std::vector<uint32_t> values;    // length nnz, raw widened counts (straight from read_1pz — NOT yet float, NOT yet log-normed)
};

// SparseView — plain aggregate handed to the numeric/compute layer (Layer,
// cuSPARSE calls). Deliberately knows nothing about species, Slot, or events,
// so it can be used from tests/benchmarks with no Batch/Slot machinery at all.
struct SparseView {
    int m = 0, B = 0, nnz = 0;
    const int32_t* d_col_ptr = nullptr;
    const int32_t* d_row_idx = nullptr;
    const float*   d_values  = nullptr;
};
```

## Interface

```cpp
class Batch {
public:
    Batch(Slot* slot,
          std::string species_name,
          const Chunk* chunk,
          std::vector<int> column_indices,   // this batch's column positions into the chunk, length B
          bool chunk_end,
          float scale = 10000.0f);           // log-norm target scale

    ~Batch();                                // marks slot FREE if still bound

    Batch(Batch&&) noexcept;                 // move-only: copy would double-free the slot
    Batch& operator=(Batch&&) noexcept;
    Batch(const Batch&) = delete;
    Batch& operator=(const Batch&) = delete;

    void prepare();  // gather+cast+CPU lognorm -> H2D -> record ready_event

    const std::string& species_name() const;
    bool               chunk_end() const;
    int                m()      const;       // features
    int                B()      const;       // batch size (== column_indices.size())
    int                nnz()    const;       // valid only after prepare() has run
    SparseView         sparse_view() const;
    cudaEvent_t        ready_event() const;
};
```

**Constraints / contracts:**
- Constructor is cheap: it only validates (`slot`/`chunk` non-null, throws `std::runtime_error` otherwise), stores fields, and calls `slot->mark_filling()`. It does **not** touch chunk data, allocate, or copy anything.
- `prepare()` must be called exactly once, after construction, before any accessor other than `species_name()`/`chunk_end()`/`m()`/`B()` is meaningful (`nnz()` and `sparse_view()` are only valid after `prepare()` returns).
- `column_indices` must all be valid positions in `[0, chunk->n)`; `Batch` does no bounds-checking on them (fail-fast is preferred over masking a `DataLoader` bug).
- `Batch` never blocks on the GPU. `prepare()` returns as soon as the async H2D copies are queued and the event is recorded — it does not synchronize.
- Move-only. After a move, the moved-from `Batch`'s destructor is a no-op (its `slot_`/`chunk_` are nulled).

## Usage

`Batch` is constructed and driven entirely by `DataLoader`, on one of `Ring`'s pool worker threads — never directly by the trainer.

```cpp
// DataLoader::fill(Slot* slot) — runs on a Ring pool worker thread
void DataLoader::fill(Slot* slot) {
    ensure_chunk_loaded_locked();

    std::vector<int> column_indices = next_batch_columns(batch_size_);   // slice of the shuffled index list
    bool is_chunk_end = advance_cursor_and_check_chunk_end(column_indices.size());

    auto batch = std::make_unique<Batch>(
        slot, species_name_, &current_chunk_, std::move(column_indices), is_chunk_end);

    batch->prepare();                 // does ALL the work: gather, cast, CPU lognorm, H2D, event
    if (is_chunk_end) start_next_chunk_decode_async();   // ordered up AFTER prepare() returns, see data_loader.md
    mark_slot_ready(slot, std::move(batch));
}
```

On the trainer side, `Batch` is only ever consumed through `Ring`, and only its accessors are used — the trainer never constructs or calls `prepare()` itself:

```cpp
std::unique_ptr<Batch> batch = ring.next_ready_batch();
cudaStreamWaitEvent(trainer_stream, batch->ready_event(), 0);

translator.forward(*batch, trainer_stream);            // dispatches on batch->species_name()
translator.backward_and_step(*batch, lr, trainer_stream);

// batch (unique_ptr) goes out of scope / is reset here -> ~Batch() -> slot marked FREE
```

Inside `Translator`/`GpuAutoencoder`, once the right per-species model is chosen, the numeric layers only ever see `batch.sparse_view()` — never the `Batch` itself — so `Layer` stays decoupled from species/Slot/event concerns:

```cpp
SparseView x = batch.sparse_view();
layer.forward(x, stream);
```

## Implementation

### `prepare()` — top-level order

```cpp
void Batch::prepare() {
    nnz_ = layout();                  // 1. size + build this batch's col_ptr
    gather_normalize(...);            // 2. gather + cast + CPU log-norm (fused, per column)
    to_device();                      // 3. H2D copy + record ready_event
}
```
CPU log-normalization **fully completes before any GPU copy is issued** — there is no GPU log-norm kernel. This is a deliberate departure from earlier versions of this design (and from the aspirational GPU-side plan in [`../singlet_lognorm.md`](../singlet_lognorm.md)): log-norm must run on CPU.

**No callback parameter (v2 — simplification).** An earlier draft threaded an `after_gather` callback through `prepare()` so `DataLoader` could kick off the next chunk's decode partway through. That's been dropped: `to_device()` is fire-and-forget (`cudaMemcpyAsync`, never blocks on the GPU), so `DataLoader` can simply call `batch->prepare()` and, once it returns, call its own next-chunk kickoff — no callback plumbing needed, and `prepare()` stays a plain no-argument call for every caller. See [data_loader.md](data_loader.md) "Ordering up the next chunk" for exactly when `DataLoader` does this.

### `layout()` — sizing pass

Two small `O(B)` passes (not performance-critical — the hot path is `gather_normalize`, not this):
1. **Size pass**: for each selected column `column_indices_[j]`, look up `chunk_->col_ptr[col+1] - chunk_->col_ptr[col]` and sum to get the batch's total `nnz`. No writes yet.
2. `slot_->ensure_capacity(B_+1, nnz, nnz)` — grows the slot's pinned+device buffers if this batch is bigger than anything seen before.
3. **Write pass**: prefix-sums the same per-column lengths into `slot_->pinned_col_ptr()` (`int32_t`, length `B_+1`) — this *is* the batch's own CSC `col_ptr`, independent of the chunk's.

Returns the total `nnz`.

### `gather_normalize()` — the hot path (fused per-column gather + cast + log-norm)

A free function (anonymous namespace in `batch.cu`) called once from `prepare()`, parallelized over the `B` columns with `#pragma omp parallel for`. For each column, **all** of its work happens before moving to the next column, so that column's small working set stays resident in L1 cache instead of the whole batch being swept multiple times:

1. Look up the column's source range in the chunk (`chunk_->col_ptr[col] .. chunk_->col_ptr[col+1]`) and its destination range in the batch (`dst_col_ptr[j] .. dst_col_ptr[j+1]`, from `layout()`).
2. **First inner loop** over that one column: cast+copy `row_idx` (`uint32_t → int32_t`) into the destination pinned buffer; cast `values` (`uint32_t → float`, raw count → float) into the destination pinned buffer (temporarily un-normalized); accumulate a running `float` sum for the column.
3. If the column's sum is `0`, skip normalization for that column (guards divide-by-zero / NaN — degenerate all-zero column is left as zeros).
4. Otherwise compute one multiplier per column, `mult = scale / sum` (a single division per column, not per element).
5. **Second inner loop** over the same small destination range: `dst_values[k] = log1p(dst_values[k] * mult)`. This loop reads data the first loop just wrote, which is why it stays L1-hot.

This is the only place type conversion happens (`uint32_t → int32_t` for indices, `uint32_t → float` for values) — everywhere else in the pipeline already deals in `int32_t`/`float`.

### `to_device()` — H2D + event

Three `cudaMemcpyAsync` calls, all `cudaMemcpyHostToDevice`, all on `slot_->stream()`:
- `col_ptr`: `sizeof(int32_t) * (B_+1)` bytes, pinned → device.
- `row_idx`: `sizeof(int32_t) * nnz_` bytes, pinned → device.
- `values`: `sizeof(float) * nnz_` bytes, pinned → device.

Then `cudaEventRecord(slot_->ready_event(), slot_->stream())` on the same stream, and `slot_->mark_ready()`. No synchronization — `prepare()` returns immediately after the copies/event are queued.

### Construction / destruction / move

- **Constructor**: stores `slot_`, `species_name_`, `chunk_`, `column_indices_`, `chunk_end_`, `scale_`; sets `m_ = chunk_->m`, `B_ = column_indices.size()`; calls `slot_->mark_filling()` last. No chunk data is touched.
- **Destructor**: if `slot_` is non-null, calls `slot_->mark_free()`. Buffers stay allocated in the slot for reuse; only the `Batch` object goes away.
- **Move ctor/assignment**: transfers all fields (including `slot_`, `chunk_`); nulls out the moved-from object's `slot_`/`chunk_` so its destructor becomes a no-op. Move-assignment calls `mark_free()` on the *current* slot (if any) before adopting the other's.

### Slot methods `Batch` relies on

`mark_filling()`, `mark_ready()`, `mark_free()` (CPU-visible state transitions), `pinned_col_ptr()`/`pinned_row_idx()`/`pinned_values()`, `device_col_ptr()`/`device_row_idx()`/`device_values()`, `stream()`, `ready_event()`, `ensure_capacity(int, int, int)`. `Batch` never allocates memory itself — all buffers come from `Slot`.

## Ownership summary

| Batch owns | Batch borrows | Batch does not touch |
|---|---|---|
| Its own metadata (species name, shape, chunk_end flag, column_indices, scale) | All buffers (pinned + device) from its `Slot` | Threads, streams (uses slot's stream) |
| The right to release the slot on destruction | The slot's `cudaEvent_t` | The `Chunk` itself — read-only, owned by `DataLoader` |

## Invariants

- A `Batch` and a `Slot` are bound 1:1 for the Batch's lifetime; the slot is marked `FILLING` at construction, `READY` at the end of `prepare()`, `FREE` at destruction.
- `species_name()` matches the owning `DataLoader`'s species.
- After `~Batch`, the slot is `FREE` and available for the next fill.
- `ready_event()` is valid to wait on immediately after `prepare()` returns; the trainer stream must `cudaStreamWaitEvent` on it before reading device buffers via `sparse_view()`.
- `Batch` assumes `chunk_` outlives it — no defensive copying of chunk data; `DataLoader` guarantees the chunk stays stable until every `Batch` drawn from it has finished `prepare()`.

## Testability

- Construct a `Batch` on a fake `Slot` + a small hand-built `Chunk` with known raw counts; call `prepare()`; verify pinned/device values match a CPU reference `log1p` computation after event sync.
- Verify a column with all-zero counts doesn't produce NaN/Inf.
- Verify destructor returns slot to `FREE`.
- Verify move-construction transfers slot/chunk ownership and the moved-from `Batch` does not release on destruction.
