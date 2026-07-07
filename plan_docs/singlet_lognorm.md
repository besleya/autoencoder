# Plan: Replace Local Log-Norm Kernel with `singlet::gpu::preprocess::log_normalize`

## Motivation

`gpu_data_loader.cu` contains a file-local `__global__` kernel
`log_normalize_columns_kernel` that duplicates logic already provided by the
singlet library at `singlet/gpu/preprocess/lognorm.h`. Replacing the local
kernel with the library implementation:

- eliminates ~30 lines of duplicated CUDA kernel code
- gains Kahan-compensated column-sum accuracy (vs. bare `atomicAdd`)
- aligns with the maintained singlet API for future improvements (e.g.,
  `ScranDeconvolution`, streaming OOC)

**Constraint**: The resulting `DataLoader` must be fully backwards-compatible:
same public C++ API, same output tensor format, numerically equivalent
normalized values.

---

## Source File Inventory

| File | Role |
|---|---|
| `gpu_data_loader.cu` | Primary change target |
| `gpu_data_loader.h` | Public header — **signature unchanged** |
| `singlet/include/singlet/gpu/preprocess/lognorm.h` | New dependency |
| `singlet/include/singlet/gpu/core/types.h` | Provides `DeviceCSC`, `DeviceMemory<T>` |

---

## API Surface Being Replaced

### What is removed

```cuda
// gpu_data_loader.cu — lines ~56–80
__global__ void log_normalize_columns_kernel(int n_cols,
                                             float scaler,
                                             const int32_t* __restrict__ col_ptr,
                                             float* __restrict__ values) {
    int col = blockIdx.x;
    if (col >= n_cols) return;
    int start = col_ptr[col];
    int end   = col_ptr[col + 1];
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    float thread_sum = 0.0f;
    for (int k = start + threadIdx.x; k < end; k += blockDim.x)
        thread_sum += values[k];
    if (thread_sum != 0.0f) atomicAdd(&s_sum, thread_sum);
    __syncthreads();
    float sum = s_sum;
    if (sum > 0.0f) {
        float inv = scaler / sum;
        for (int k = start + threadIdx.x; k < end; k += blockDim.x)
            values[k] = log1pf(values[k] * inv);
    }
}
```

### What replaces it

```cpp
// singlet/include/singlet/gpu/preprocess/lognorm.h
namespace singlet::gpu::preprocess {

inline LogNormResult log_normalize(
    singlet::gpu::core::DeviceCSC& mat,
    const LogNormConfig&           cfg    = {},
    cudaStream_t                   stream = nullptr);

}
```

`DeviceCSC` (from `singlet/gpu/core/types.h`) has a non-owning factory:

```cpp
static DeviceCSC from_device_ptrs(int m, int n, int nz,
                                  int32_t* d_col_ptr,
                                  int32_t* d_row_indices,
                                  float*   d_values);
```

`LogNormConfig`:
```cpp
struct LogNormConfig {
    LogNormMethod method       = LogNormMethod::TotalCount;
    float         target_count = 0.0f;  // 0 = use median of positive col-sums
    bool          approximate_median = false;
    uint64_t      seed         = 0;
};
```

Setting `target_count = 10000.0f` makes `T = 10000.0f` (used directly, not
as a median seed). See §Numerical Equivalence for proof of equivalence.

`LogNormResult` is RAII — both `size_factors` and `qc_mask` device buffers
free automatically when the object is destroyed. It must not outlive its
stream's sync point if the buffers are accessed afterwards, but since
`DataLoader` discards the result, this is safe.

---

## Detailed Change Specification

### 1. `gpu_data_loader.cu` — Add include

**Location**: include block at the top of the file, after existing singlet
includes.

```cpp
// ADD:
#include <singlet/gpu/preprocess/lognorm.h>
```

No changes to `gpu_data_loader.h` — `lognorm.h` types are implementation
details and must not leak into the public header.

---

### 2. `gpu_data_loader.cu` — Remove local kernel

**Action**: Delete the entire `log_normalize_columns_kernel` `__global__`
function definition (~30 lines).

`kSparseLogNormScaler = 10000.0f` is retained; it is reused as
`LogNormConfig::target_count` in both call sites below.

---

### 3. `gpu_data_loader.cu` — Replace call in `batch_builder_thread`

**Location**: Inside the `while (col_idx < chunk->n_cols)` loop, after the
H2D `cudaMemcpyAsync` calls for `d_values`.

**Before**:
```cpp
// Launch lognorm kernel
log_normalize_csc_columns(B, kSparseLogNormScaler,
                          slot_view.d_col_ptr, slot_view.d_values,
                          impl->loader_stream);
```

**After**:
```cpp
// Launch lognorm via singlet (TotalCount, target=10000)
{
    namespace sp = singlet::gpu::preprocess;
    auto mat = singlet::gpu::core::DeviceCSC::from_device_ptrs(
        impl->m, B, batch_nnz,
        slot_view.d_col_ptr,
        slot_view.d_row_idx,
        slot_view.d_values);
    sp::LogNormConfig cfg;
    cfg.target_count = kSparseLogNormScaler;
    sp::log_normalize(mat, cfg, impl->loader_stream);
    // LogNormResult (size_factors, qc_mask) discarded here — not needed
    // downstream; RAII destructor frees device buffers automatically.
}
```

All three values needed by `DeviceCSC::from_device_ptrs` are already in
scope at this call site:
- `impl->m` — number of features (rows), set at construction time
- `B` — `impl->batch_size`, the column count for this batch
- `batch_nnz` — computed earlier in the same loop iteration

---

### 4. `gpu_data_loader.cu` — Replace `log_normalize_csc_columns` wrapper

The public wrapper function (declared in `gpu_data_loader.h`, signature
**must not change**) currently launches the local kernel. It must be
reimplemented using the singlet API.

**Current signature** (do not change):
```cpp
void log_normalize_csc_columns(int n_cols,
                                float scaler,
                                const int32_t* d_col_ptr,
                                float* d_values,
                                cudaStream_t stream);
```

**Challenge**: The signature does not include `m` (number of rows) or `nnz`.
`DeviceCSC::from_device_ptrs` requires these. However:

- All singlet lognorm kernels (`compute_col_sums_kernel`,
  `compute_size_factors_and_apply_kernel`) derive their iteration bounds from
  `indptr` alone; they do not use `mat.rows` or `mat.nnz` for any
  computation.
- Intermediate workspace reported by singlet is "9n bytes" where n = `n_cols`
  — all allocations are col-count-sized, not nnz-sized.
- Row indices are not accessed during normalization (kernels only read
  `values` and `indptr`).

**Therefore**: `from_device_ptrs(m=0, n=n_cols, nz=0, d_col_ptr, /*row_indices=*/nullptr, d_values)`
is expected to be safe. The `rows=0` and `nnz=0` fields are metadata that
singlet does not use during the normalization computation.

> ⚠️ **Verification required before implementation**: Confirm that
> `singlet::gpu::preprocess::log_normalize` does not dereference `mat.nnz`
> for any allocation or does not assert `nnz > 0`. Read the body of
> `log_normalize` (not just the kernels) to verify this assumption before
> writing the code.

**After** (if assumption confirmed):
```cpp
void log_normalize_csc_columns(int n_cols,
                                float scaler,
                                const int32_t* d_col_ptr,
                                float* d_values,
                                cudaStream_t stream) {
    if (n_cols <= 0) return;
    namespace sp = singlet::gpu::preprocess;
    // Row indices not used in normalization; nnz/m not used by singlet kernels.
    auto mat = singlet::gpu::core::DeviceCSC::from_device_ptrs(
        /*m=*/0, /*n=*/n_cols, /*nz=*/0,
        const_cast<int32_t*>(d_col_ptr),
        /*d_row_indices=*/nullptr,
        d_values);
    sp::LogNormConfig cfg;
    cfg.target_count = scaler;
    sp::log_normalize(mat, cfg, stream);
}
```

**Fallback** (if assumption is not confirmed): Keep the local kernel only for
this wrapper. In that case, only changes 1–3 are applied, and the kernel is
kept (renamed to `log_normalize_columns_kernel_legacy` or similar with a
comment noting it exists only for the public wrapper).

---

## Numerical Equivalence

**Old algorithm** (per column `j`):
```
sum_j  = Σ values[k]  for k in [col_ptr[j], col_ptr[j+1])
inv    = scaler / sum_j        (scaler = 10000.0f)
values[k] ← log1pf(values[k] * inv)
         = log1pf(values[k] * 10000 / sum_j)
```

**New algorithm** (`LogNormMethod::TotalCount`, `target_count = 10000.0f`):
```
T      = target_count = 10000.0f   (used directly, not median)
sum_j  = Σ values[k]  for k in [col_ptr[j], col_ptr[j+1])   (Kahan-compensated)
s_j    = sum_j / T
values[k] ← log1pf(values[k] / s_j)
           = log1pf(values[k] * T / sum_j)
           = log1pf(values[k] * 10000 / sum_j)
```

The expressions are **mathematically identical**. Minor floating-point
differences are expected and acceptable: the new implementation uses
Kahan-compensated accumulation (more accurate), whereas the old kernel uses
`atomicAdd` into shared memory (susceptible to round-off). This is a
numerical improvement, not a regression.

---

## Backwards Compatibility Matrix

| API surface | Before | After | Compatible? |
|---|---|---|---|
| `DataLoader` constructor signature | unchanged | unchanged | ✓ |
| `DataLoader::start()` | unchanged | unchanged | ✓ |
| `DataLoader::m()` | unchanged | unchanged | ✓ |
| `DataLoader::loader_stream()` | unchanged | unchanged | ✓ |
| `DataLoader::lane_id()` | unchanged | unchanged | ✓ |
| `log_normalize_csc_columns` signature | 5 params | 5 params | ✓ |
| Output batch format (CSC on GPU) | unchanged | unchanged | ✓ |
| Normalized values (math) | `log1p(x·10000/sum)` | `log1p(x·10000/sum)` | ✓ |
| `gpu_data_loader.h` contents | unchanged | unchanged | ✓ |

---

## New Build Dependencies

| Dependency | Already linked? | Action needed |
|---|---|---|
| `singlet/gpu/preprocess/lognorm.h` | header-only | Add `#include` |
| `cooperative_groups.h` | pulled in transitively | None |
| `singlet/gpu/core/types.h` | already included transitively | None (also add explicit include for clarity) |

If singlet is already on the include path (it is — other singlet headers are
already used in `gpu_data_loader.cu`), no Makefile changes are required.

---

## Implementation Order

1. Verify the `log_normalize` function body does not use `mat.nnz` or
   `mat.rows` for any allocation or guard (read the body in `lognorm.h`).
2. Add `#include <singlet/gpu/preprocess/lognorm.h>` to `gpu_data_loader.cu`.
3. Replace the call in `batch_builder_thread` (Change 3 above).
4. Replace `log_normalize_csc_columns` wrapper (Change 4 above), or fall back
   to keeping the local kernel if the verification in step 1 fails.
5. Delete `log_normalize_columns_kernel` (Change 2 above).
6. Build and run the existing test suite. Any numerical differences in
   normalized values should be within FP32 epsilon.

---

## Open Questions / Assumptions to Resolve

1. **Critical**: Does `singlet::gpu::preprocess::log_normalize` access
   `mat.nnz` or `mat.rows` in the function body (not just the kernels)?
   If it does, the `log_normalize_csc_columns` wrapper must keep the local
   kernel (fallback path above) or its signature must be extended.

2. **Minor**: `LogNormResult` contains `size_factors` and `qc_mask` device
   buffers that are freed by RAII. When destroyed inside the `{ }` block in
   `batch_builder_thread`, the `cudaFree` calls happen on the CPU thread, not
   the CUDA stream. This is always safe (cudaFree is synchronous for the
   device), but confirm there is no performance concern for the tight
   batch-building loop.

3. **Minor**: The singlet `log_normalize` function (if not inlined) may call
   `singlet::gpu::core::default_context()` when `stream = nullptr`. Since we
   always pass `impl->loader_stream` (never `nullptr`), this path is not
   taken. Confirm that passing a non-null stream bypasses the singleton
   initialization entirely.
