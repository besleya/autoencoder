# kernel_add_bias Replacement Benchmark

## Overview
This directory contains a cuBLAS-based replacement for the `kernel_add_bias` CUDA kernel from `/mnt/home/besleya/autoencoder/layer.cu` (line 62).

## Target Kernel
- **Name**: `kernel_add_bias`
- **Location**: `/mnt/home/besleya/autoencoder/layer.cu:62`
- **Signature**: `__global__ void kernel_add_bias(int out, int B, float* z, const float* b)`
- **Operation**: Broadcasts bias vector `b` (length `out`) onto a column-major matrix `z` (shape out × B)
  - `z[i, j] += b[i]` for all i ∈ [0, out), j ∈ [0, B)
  - z is stored column-major with leading dimension `out`

## Replacement Strategy
Uses cuBLAS `cublasSger` to implement a rank-1 outer-product update:
```
z += 1.0 * b * ones^T
```
Where:
- `b` is the bias vector of length `out`
- `ones` is a vector of length `B` containing all 1.0
- `z` is the output matrix (out × B, column-major)

This leverages the optimized cuBLAS library rather than a custom CUDA kernel, allowing:
- Better GPU memory bandwidth utilization
- Automatic tuning for different architectures
- Reduced custom kernel maintenance

## Files

### Core Implementation
- **`original.cu`**: Verbatim copy of the original kernel from layer.cu with citation
  - Contains `kernel_add_bias` and `add_bias_kernel` launcher
  
- **`replacement.cu`**: cuBLAS replacement implementation
  - `add_bias_cublas(handle, out, B, d_z, d_b, d_ones_B, stream)`: Main variant expecting caller-managed ones buffer
  - `add_bias_cublas_standalone(...)`: Standalone variant that allocates its own ones buffer

### Testing & Benchmarking
- **`reference.py`**: Generates reference data using PyTorch
  - Parameters: out=512, B=256
  - Outputs: `z_init.bin`, `b.bin`, `expected_z.bin`, `meta.txt`
  
- **`test_accuracy.cu`**: Verifies correctness
  - Loads reference inputs
  - Runs original kernel on copy of z
  - Runs cuBLAS replacement on copy of z
  - Compares both against expected output (rel_err ≤ 1e-5, abs_err ≤ 1e-6)
  - Reports pass/fail
  
- **`bench.cu`**: Performance benchmark
  - 10 warmup iterations
  - 100 timed iterations per implementation
  - Reinitializes z from z_init before each iteration (ensures values don't accumulate)
  - Reports median time (ms) and speedup ratio

### Build & Execution
- **`Makefile`**: Builds `test_accuracy` and `bench` executables
  - Includes `compile-check` target for compilation verification without execution
  
- **`run_tests.sh`**: SLURM submission script
  - Generates reference data
  - Builds
  - Runs accuracy and benchmark tests
  - Capture to slurm-%j.out

## Building & Testing

### Compile-only check (no execution):
```bash
cd /mnt/home/besleya/autoencoder/kernel_bench/kernel_add_bias
make compile-check
```

### Local build:
```bash
cd /mnt/home/besleya/autoencoder/kernel_bench/kernel_add_bias
python reference.py
make
./test_accuracy
./bench
```

### SLURM submission:
```bash
sbatch /mnt/home/besleya/autoencoder/kernel_bench/kernel_add_bias/run_tests.sh
```

## Parameters
- **out**: Output dimension (hidden layer size) = 512
- **B**: Batch size = 256
- **Total elements in z**: 512 × 256 = 131,072

## Expected Outcomes
- **Accuracy**: Both implementations should match reference within tolerance (rel_err ≤ 1e-5)
- **Performance**: cuBLAS is expected to be faster due to:
  - Better memory bandwidth utilization (gemv-like operations)
  - Optimized for batch operations
  - Reduced kernel launch overhead when amortized

## Notes
- Bias replication (ones vector) can be cached across multiple add_bias calls in real applications
- The benchmark reinitializes z each iteration to prevent value accumulation and ensure consistent numerical properties
- All files use column-major storage (CUDA/cuBLAS convention)

## Constraints
- GPU compilation only; no CPU fallback
- Uses sm_90 architecture (adjust CXXFLAGS if targeting different GPU)
- Requires CUDA toolkit with cuBLAS support
