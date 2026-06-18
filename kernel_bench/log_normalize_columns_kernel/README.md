# log_normalize_columns_kernel — CUB+Thrust Replacement

## Overview

This directory contains a CUDA-library replacement for the `log_normalize_columns_kernel` from `gpu_data_loader.cu`.

### Original Kernel Behavior

For a sparse CSC matrix with column pointers (`col_ptr`) and values (`values`):
- For each column j: compute `s = sum(values[col_ptr[j]:col_ptr[j+1]])`
- If s > 0: replace each value v with `log1pf(scaler * v / s)`
- If s ≤ 0: leave values unchanged

### Replacement Strategy

**CUB Segmented Reduce + Thrust Transform (library-only, zero hand-written kernels)**

1. **Compute per-column sums**:
   - Use `cub::DeviceSegmentedReduce::Sum` with `col_ptr` as segment offsets
   - Output: `d_col_sums` (length n_cols)

2. **Transform values**:
   - Use `thrust::for_each` with a counting iterator over nnz indices
   - Custom functor `LogNormalizeTransform`:
     - Binary-search each nnz index in `col_ptr` to find its column
     - Apply `log1pf(scaler * v / col_sums[col])` if col_sums[col] > 0

**Workspace Management**: CUB workspace is allocated internally within `log_normalize_columns_lib()` via `cudaMalloc`, then freed at function end. No external workspace management required for testing.

## Files

- **`original.cu`**: Verbatim copy of the original kernel from `gpu_data_loader.cu:41` with launcher wrapper
- **`replacement.cu`**: Library-only implementation using CUB + Thrust
- **`reference.py`**: Generate CSC test matrix on CPU (torch/numpy)
  - n_cols=256, nnz ~5000 (uniform [0,40] per column)
  - Values float32 in [0, 100]
  - Outputs: `col_ptr.bin`, `values_init.bin`, `expected_values.bin`, `scaler.bin`, `meta.txt`
- **`test_accuracy.cu`**: Verify correctness against reference (rel ≤ 1e-5, abs ≤ 1e-6)
- **`bench.cu`**: Benchmark both implementations (10 warmup + 100 timed iterations, re-init each time)
- **`Makefile`**: Build test_accuracy, bench, and compile-only check targets
- **`run_tests.sh`**: SLURM submission script (GPU required)
- **`README.md`**: This file

## Building

```bash
cd /mnt/home/besleya/autoencoder/kernel_bench/log_normalize_columns_kernel

# Compile-only check (no GPU required)
make check

# Full build (GPU required for linking)
make
```

## Testing

### On SLURM-enabled GPU node (recommended):
```bash
sbatch run_tests.sh
```

### Locally (GPU required):
```bash
bash run_tests.sh
```

### Step-by-step:
```bash
~/s3/bin/python reference.py    # Generate test data
make                            # Build
./test_accuracy                 # Verify correctness
./bench                         # Benchmark
```

## Compile-Only Validation

Verify that both implementations compile without errors (no GPU required):
```bash
/usr/local/cuda/bin/nvcc -c -O3 -std=c++17 --gpu-architecture=sm_90 --extended-lambda \
  -I/usr/local/cuda/include original.cu -o /tmp/orig_klnc.o

/usr/local/cuda/bin/nvcc -c -O3 -std=c++17 --gpu-architecture=sm_90 --extended-lambda \
  -I/usr/local/cuda/include replacement.cu -o /tmp/repl_klnc.o
```

Or use the Makefile:
```bash
make check
```

## Performance Notes

- **Original**: One CUDA thread-block per column, shared-memory reduction for per-column sum
- **Replacement**: CUB handles the reduction more efficiently, thrust::for_each parallelizes the transform
- Expected speedup: Depends on column distribution and GPU saturation

## Expected Results

- **Accuracy**: Both implementations produce identical results (within 1e-5 relative / 1e-6 absolute tolerance)
- **Benchmark output**:
  ```
  === Speedup ===
  Original:  X.XXXX ms
  Library:   X.XXXX ms
  Speedup:   Y.YY×
  ```

## Dependencies

- CUDA 12.8+ with CUB and Thrust
- Python 3.8+ with torch 2.8+ (CPU mode for reference.py)
- sm_90 GPU (H100) for execution; compile-only checks work on any system with nvcc

## Tolerance Rationale

- `log1pf` has ~6-7 significant digits of precision in float32
- Relative tolerance 1e-5 (~5 digits) + absolute tolerance 1e-6 (captures near-zero values)
- These tolerances are standard for GPU double-reduction & transform operations

## Notes

- All files remain in this subdirectory; no modifications to parent directory or original source files
- GPU execution is via SLURM (`sbatch run_tests.sh`); no direct GPU access on login node
- Compile checks use `make check` and do not require GPU
