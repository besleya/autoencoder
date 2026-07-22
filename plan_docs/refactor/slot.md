# Slot

One prefetch buffer. Tracks its own state. Owns GPU/pinned memory.

## States

```
FREE → FILLING → READY → (consumed) → FREE
```

- **FREE**: no active batch. Ready to be assigned. Slots start here.
- **FILLING**: DataLoader has reserved the slot (Ring dispatcher called `reserve_free_slot()`). Enters this state **immediately** on reservation, even before any thread starts real work. Stays here until `Batch::prepare()` finishes queuing GPU work and records the ready event.
- **READY**: `Batch::prepare()` has returned. The `ready_event` has been recorded; the GPU may or may not have reached it yet. Trainer consumes by waiting on the event on its trainer stream.

State transitions are driven by DataLoader (CPU-visible flag) and observed on GPU via `ready_event`.

## Owned resources

- Pinned host buffers: `col_ptr`, `row_idx`, `values` — capacities set to the species' max batch nnz / size at construction. Grown (realloc) if a batch exceeds capacity; rare.
- Device buffers: matching triple on GPU.
- `cudaStream_t` — dedicated stream for this slot's H2D + log-norm work. Owning a stream per slot keeps prep work parallel across slots.
- `cudaEvent_t ready_event` — recorded at end of `Batch::prepare()`.

## Interface

```cpp
class Slot {
public:
    enum class State { FREE, FILLING, READY };

    Slot(int m, int batch_size, int max_nnz_estimate);
    ~Slot();                                     // frees pinned + device + stream + event

    State state() const;                         // atomic load
    void  mark_filling();                        // FREE → FILLING (asserts prior state)
    void  mark_ready();                          // FILLING → READY (called from fill())
    void  mark_free();                           // READY → FREE (called on consume)

    // Buffer access — used by Batch::prepare()
    void ensure_capacity(int cap_col, int cap_row, int cap_val);
    int32_t* pinned_col_ptr();  int32_t* pinned_row_idx();  float* pinned_values();
    int32_t* device_col_ptr();  int32_t* device_row_idx();  float* device_values();

    cudaStream_t stream() const;
    cudaEvent_t  ready_event() const;
};
```

## Invariants

- State transitions are strictly ordered; any illegal transition asserts.
- Only one `Batch` is bound to a slot at a time.
- Buffers are never freed while state ≠ FREE.
- `ready_event` is only recorded from `Batch::prepare()`; the trainer waits on it before touching device buffers.

## Testability

- Construct a slot; assert initial state FREE.
- Cycle through FREE→FILLING→READY→FREE without errors.
- Assert illegal cycles trip (e.g. FREE→READY directly).
- Verify `ensure_capacity` grows buffers when asked and is a no-op when already big enough.
