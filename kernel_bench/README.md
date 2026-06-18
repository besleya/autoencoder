# Kernel Benchmark and Replacement Test Suite

This directory contains automated tests and benchmarks for custom CUDA kernel replacements using CUDA libraries (Thrust, cuBLAS, CUB).

## Overview

Each subdirectory contains a self-contained SLURM job script that:
1. Generates reference test data using Python
2. Builds the test and benchmark binaries
3. Runs accuracy verification against original kernel
4. Benchmarks performance of the replacement vs. original

## Kernel Benchmarks

### 1. **kernel_add_bias**
- **Location**: `/mnt/home/besleya/autoencoder/layer.cu:62`
- **Replacement Strategy**: cuBLAS `cublasSger` (rank-1 outer-product update)
- **Operation**: Broadcasts bias vector onto matrix: `z[i,j] += b[i]`

### 2. **kernel_relu_forward**
- **Location**: `/mnt/home/besleya/autoencoder/layer.cu:73`
- **Replacement Strategy**: Thrust `thrust::transform` with device lambda
- **Operation**: Forward pass ReLU activation: `a = max(z, 0)`

### 3. **kernel_relu_backward**
- **Location**: `/mnt/home/besleya/autoencoder/layer.cu` (backward)
- **Replacement Strategy**: Thrust `thrust::transform` with device lambda
- **Operation**: Backward pass ReLU gating: `dz *= (z > 0) ? 1 : 0`

### 4. **kernel_fill_ones**
- **Location**: `/mnt/home/besleya/autoencoder/layer.cu:89`
- **Replacement Strategy**: Thrust `thrust::fill_n` library call
- **Operation**: Fill vector with ones: `v[i] = 1.0`

### 5. **kernel_adam_update**
- **Location**: `/mnt/home/besleya/autoencoder/layer.cu:97-107`
- **Replacement Strategy**: Thrust single-pass transform (vs. multi-call cuBLAS)
- **Operation**: Adam optimizer per-element update:
  - `m[i] = beta1*m[i] + (1-beta1)*g[i]`
  - `v[i] = beta2*v[i] + (1-beta2)*g[i]²`
  - `p[i] -= lr_t * m[i] / (sqrt(v[i]) + eps)`

### 6. **kernel_sparse_loss_and_grad**
- **Location**: `gpu_autoencoder.cu:59-79` (kernel), line 334 (launcher)
- **Replacement Strategy**: Thrust + CUB + cuBLAS for sparse CSC operations
- **Operation**: Compute sparse loss and gradients for sparse targets

### 7. **kernel_accumulate_loss**
- **Location**: `/mnt/home/besleya/autoencoder/gpu_autoencoder.cu:87-97`
- **Replacement Strategy**: Library-based replacement for serial scalar accumulation
- **Operation**: Accumulate batch loss into epoch sum: `*d_epoch_sum += (*d_dot + *d_loss) / (d0*B)`

### 8. **log_normalize_columns_kernel**
- **Location**: `gpu_data_loader.cu`
- **Replacement Strategy**: CUB Segmented Reduce + Thrust Transform
- **Operation**: Normalize sparse matrix columns with log scaling

## Running the Tests

### Submit all kernel jobs at once:

```bash
bash ~/autoencoder/kernel_bench/submit_all.sh
```

This script will:
- Create `slurm_logs/` directories in each kernel subdirectory
- Submit all 8 jobs to SLURM
- Print job IDs and status check commands
- Provide instructions for viewing results

### Monitor job status:

```bash
squeue -u $USER
```

### View specific job output (replace JOBID with actual ID):

```bash
tail -f ~/autoencoder/kernel_bench/<kernel>/slurm_logs/kbench_<kernel>_JOBID.out
```

### Check all results once complete:

```bash
ls -lh ~/autoencoder/kernel_bench/*/slurm_logs/
grep -h "PASS\|FAIL" ~/autoencoder/kernel_bench/*/slurm_logs/*.out
```

## Test Output

Each job produces `.out` and `.err` files in the kernel's `slurm_logs/` subdirectory with:
- **Timing Information**: Warmup and timed iteration results
- **Accuracy Status**: PASS/FAIL message from `./test_accuracy`
- **Performance Metrics**: Throughput/latency from `./bench`
- **Build Logs**: Compilation output from `make clean && make`

## Important Notes

- The in-place replacements in the top-level `.cu` files (e.g., `layer.cu`, `gpu_autoencoder.cu`) have **NOT** been applied yet
- These benchmarks serve as **verification** that the library-based replacements are correct and performant
- Only after SLURM tests pass will replacements be integrated into the main codebase
- Each kernel test is independent and can run in parallel on the GPU partition

## Accuracy Test Tolerance

Each kernel's `test_accuracy` binary uses relative and absolute error tolerances:
- Typical values: `relative_tol = 1e-5`, `absolute_tol = 1e-6`
- Check individual kernel README.md files for specific tolerances

## Next Steps (After Tests Pass)

1. Review benchmark results in `slurm_logs/` directories
2. If all tests PASS, integrate replacements into top-level `.cu` files
3. Re-benchmark the full autoencoder training pipeline
4. Measure end-to-end speedup from kernel replacements
