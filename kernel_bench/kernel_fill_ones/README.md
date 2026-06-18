# kernel_fill_ones: CUDA Library Replacement

## Summary

Replacement of custom CUDA kernel `kernel_fill_ones` with `thrust::fill_n` library call.

### Original Kernel

```cuda
__global__ void kernel_fill_ones(int n, float* v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        v[idx] = 1.0f;
    }
}
```

**Source**: `/mnt/home/besleya/autoencoder/layer.cu`, line ~89  
**Launcher**: `/mnt/home/besleya/autoencoder/layer.cu`, line ~330 in `_ensure_batch_buffers`

### Replacement Strategy

Use `thrust::fill_n(thrust::cuda::par.on(stream), v, n, 1.0f)`:
- **Why thrust?** `cudaMemset` cannot directly set `float 1.0f` (it works at byte level)
- **Why fill_n?** Cleanest library mapping for filling a device buffer with a value
- **Stream support?** Yes, via `thrust::cuda::par.on(stream)`

### Implementation

```cuda
void fill_ones_thrust(int n, float* v, cudaStream_t stream) {
    thrust::fill_n(thrust::cuda::par.on(stream), v, n, 1.0f);
}
```

## Files

- **original.cu**: Verbatim kernel definition + launcher with source citation
- **replacement.cu**: Thrust-based implementation
- **reference.py**: CPU reference (PyTorch) generating expected_v.bin and meta.txt
- **test_accuracy.cu**: Correctness test (alloc, fill with garbage, run impl, compare)
- **bench.cu**: Benchmark (10 warmup, 100 timed iters, median + speedup)
- **Makefile**: Build configuration with `--extended-lambda` flag
- **run_tests.sh**: SLURM submission script
- **README.md**: This file

## Build & Test

### Local Compile-Only Check

```bash
cd /mnt/home/besleya/autoencoder/kernel_bench/kernel_fill_ones

# Compile original
/usr/local/cuda/bin/nvcc -c -O3 -std=c++17 --gpu-architecture=sm_90 \
  --extended-lambda -I/usr/local/cuda/include original.cu -o /tmp/orig_kfo.o

# Compile replacement
/usr/local/cuda/bin/nvcc -c -O3 -std=c++17 --gpu-architecture=sm_90 \
  --extended-lambda -I/usr/local/cuda/include replacement.cu -o /tmp/repl_kfo.o

# Link test
/usr/local/cuda/bin/nvcc -O3 -std=c++17 --gpu-architecture=sm_90 \
  --extended-lambda -I/usr/local/cuda/include test_accuracy.cu \
  original.cu replacement.cu -o /tmp/link_kfo_test \
  -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusparse -lnvToolsExt -lpthread

# Link bench
/usr/local/cuda/bin/nvcc -O3 -std=c++17 --gpu-architecture=sm_90 \
  --extended-lambda -I/usr/local/cuda/include bench.cu \
  original.cu replacement.cu -o /tmp/link_kfo_bench \
  -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusparse -lnvToolsExt -lpthread
```

### SLURM Job

```bash
sbatch run_tests.sh
```

This will:
1. Generate reference data (expected_v.bin, meta.txt)
2. Build test_accuracy and bench
3. Run accuracy test
4. Run benchmark

## Test Parameters

- **n**: 4096 floats (12 KB)
- **Expected value**: 1.0f (bit-exact)
- **Accuracy tolerance**: rel <= 1e-5, abs <= 1e-6 (or bit-exact check)
- **Benchmark**: 10 warmup, 100 timed iterations, report median

## Expected Results

- **Correctness**: All output values should be 1.0f
- **Performance**: Thrust may be comparable or slightly faster due to library optimizations
- **Compilation**: Both original and replacement compile cleanly with nvcc

## Notes

- No modifications outside this directory
- No GPU execution (compile-only checks only)
- No cuDNN dependency
- Uses CUDA 12.x (cudaStream_t, thrust on device)
