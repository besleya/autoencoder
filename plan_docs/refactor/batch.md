# Batch

Transient abstraction of one training batch. Owns no memory. Borrows buffers from a `Slot`.

## Purpose

- Give downstream code (Translator, Autoencoder) a clean, self-describing handle.
- Encapsulate the "prepare on host, ship to GPU, log-norm" pipeline as one call.
- Carry species identity so `Translator::forward(batch)` can dispatch.

## Lifecycle

1. **Init**: `DataLoader::fill(slot)` constructs a `Batch` bound to that `slot`, populated with the next `batch_size` rows drawn from the current chunk. At construction, the Batch exists only conceptually in CPU memory (indices into the chunk).
2. **Materialize on host**: `Batch::prepare()` type-converts and packs the rows into the slot's **pinned** buffers, in CSC form.
3. **Ship to GPU**: `prepare()` issues three `cudaMemcpyAsync` calls (col_ptr, row_idx, values) on the slot's stream.
4. **Log-norm on GPU**: `prepare()` launches the log-normalize kernel on the slot's stream.
5. **Signal ready**: `prepare()` records the slot's `ready_event` and returns.
6. **Consume**: Trainer receives the `Batch` from `Ring::next_ready_batch()`, waits on the ready event from its trainer stream, runs forward/backward/update.
7. **Self-destruct**: `Batch`'s destructor marks the slot FREE. Buffers stay allocated in the slot for reuse; only the transient `Batch` object goes away.

## Interface

```cpp
class Batch {
public:
    // Constructed only by DataLoader; not user-facing.
    Batch(Slot* slot,
          std::string species_name,
          /* chunk rows to include */,
          bool chunk_end);

    ~Batch();   // marks slot FREE if still bound

    // Move-only. Copy would double-free the slot.
    Batch(Batch&&) noexcept;
    Batch& operator=(Batch&&) noexcept;
    Batch(const Batch&) = delete;

    void prepare();   // host-pack → H2D → lognorm → record ready_event. Idempotent-safe: called once by DataLoader.

    // Accessors used by trainer / Translator / Autoencoder
    const std::string& species_name() const;
    bool               chunk_end() const;
    int                m()      const;   // features
    int                B()      const;   // batch size
    int                nnz()    const;
    SparseView         sparse_view() const;   // { d_col_ptr, d_row_idx, d_values, m, B, nnz }
    cudaEvent_t        ready_event() const;   // trainer waits on this
};
```

`SparseView` is a plain aggregate (device pointers + shape). It replaces today's `SparseBatch` struct at the trainer boundary; the existing `Autoencoder::forward(const SparseBatch&, ...)` signature can keep the same shape (rename or typedef as convenient).

## Log-normalization

Runs **inside `prepare()`**, on GPU, on the slot's stream, before `ready_event` is recorded. Formula unchanged from current implementation (per-column sum → `log(1 + v · 10000 / sum)`). The plan in [`../singlet_lognorm.md`](../singlet_lognorm.md) still applies: swap the kernel body for the singlet-lib version.

## Ownership summary

| Batch owns | Batch borrows | Batch does not touch |
|---|---|---|
| Its own metadata (species name, shape, chunk_end flag) | All buffers (pinned + device) from its `Slot` | Threads, streams (uses slot's stream) |
| The right to release the slot on destruction | The slot's `cudaEvent_t` | Files, chunks (that's DataLoader's job) |

## Invariants

- A `Batch` and a `Slot` are bound 1:1 for the Batch's lifetime.
- `species_name()` matches the owning DataLoader's species.
- After `~Batch`, the slot is FREE and available for the next `reserve_free_slot()` call.
- `ready_event()` is valid to wait on immediately after `prepare()` returns; the trainer stream must `cudaStreamWaitEvent` on it before reading device buffers.

## Testability

- Construct a Batch on a fake slot with known data; call `prepare()`; verify device buffers match a CPU reference after event sync.
- Verify destructor returns slot to FREE.
- Verify move-construction transfers slot ownership and the moved-from Batch does not release on destruction.
