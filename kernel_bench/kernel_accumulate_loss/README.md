# kernel_accumulate_loss Replacement Benchmark

## Kernel Overview

**Name**: `kernel_accumulate_loss`  
**Source**: `/mnt/home/besleya/autoencoder/gpu_autoencoder.cu` (lines 87–97)  
**Launch Site**: `/mnt/home/besleya/autoencoder/gpu_autoencoder.cu` (line 358)  
**Launch Config**: `<<<1, 1>>>`

### What It Does

Single-thread CUDA kernel that accumulates per-batch loss into a device-side epoch sum:

```cpp
*d_epoch_sum += (*d_dot + *d_loss) / (d0 * B)
```

- Called **once per batch** during backward pass
- Reads two scalars (`d_dot`, `d_loss`), updates one scalar (`d_epoch_sum`)
- No thread cooperation; pure serial computation

## Replacement Strategy

This kernel is **trivial but has high launch overhead** due to:
- Single-thread serialization
- Minimal computational work (2 FLOPs)
- Kernel launch/sync overhead dominates

### Approach A: cuBLAS saxpy (saxpy_v2 × 2)

```cpp
scale = 1/(d0*B)
cublasSaxpy(n=1): d_epoch_sum += scale * d_dot
cublasSaxpy(n=1): d_epoch_sum += scale * d_loss
```

**Pros**: GPU-based, avoids host-side ops  
**Cons**: Two cuBLAS function calls per batch

### Approach B: Host-based accumulation (cudaMemcpyAsync)

```cpp
D2H: d_dot, d_loss, d_epoch_sum → host (3 floats)
Host: epoch_sum += (dot + loss) / (d0*B)
H2D: epoch_sum → device
```

**Pros**: Minimal device overhead, single-threaded host logic eliminates kernel launch  
**Cons**: Host ↔ device memory transfers, but only 3 floats = negligible

### Recommended: Approach B (Host Accumulation)

**Rationale**: For single-element scalars, D2H transfer overhead is minimal (~microseconds) and eliminates kernel launch overhead. Host accumulation is simpler and typically **faster than original or saxpy**.

## File Structure

- **original.cu**: Verbatim kernel + launcher
- **replacement.cu**: Approach B (primary) and Approach A (comparison alternative)
- **reference.py**: Generate test reference data (d_dot, d_loss, expected epoch_sum)
- **test_accuracy.cu**: Verify original vs replacement correctness
- **bench.cu**: Timing comparison (10 warmup, 1000 timed iterations)
- **Makefile**: Build both executables
- **run_tests.sh**: SLURM submission script

## Test Parameters

- **d0**: 64
- **B**: 32
- **denom**: d0 × B = 2048
- **Initial epoch_sum**: 0.5
- **Test inputs**: Random floats (seed=42 for reproducibility)
- **Tolerance**: rel_err ≤ 1e-5 or abs_err ≤ 1e-6

## How to Run

### Interactive (Compile-Only Check)

From the kernel directory:

```bash
cd /mnt/home/besleya/autoencoder/kernel_bench/kernel_accumulate_loss

# Compile-only checks
/usr/local/cuda/bin/nvcc -c -O3 -std=c++17 --gpu-architecture=sm_90 \
  -I/usr/local/cuda/include original.cu -o /tmp/orig_kal.o

/usr/local/cuda/bin/nvcc -c -O3 -std=c++17 --gpu-architecture=sm_90 \
  -I/usr/local/cuda/include replacement.cu -o /tmp/repl_kal.o

# Link test
/usr/local/cuda/bin/nvcc -O3 -std=c++17 --gpu-architecture=sm_90 \
  -I/usr/local/cuda/include test_accuracy.cu original.cu replacement.cu \
  -o /tmp/link_kal_test -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusparse -lnvToolsExt -lpthread
```

### Full Benchmark (GPU Required)

Generate reference data and build:

```bash
~/s3/bin/python reference.py
make
```

Run tests:

```bash
./test_accuracy
./bench
```

Or submit as SLURM job:

```bash
chmod +x run_tests.sh
sbatch run_tests.sh
```

## Expected Results

### Test Accuracy Output

```
=== Test Accuracy: kernel_accumulate_loss ===
Parameters: d0=64, B=32, denom=2048
Inputs: d_dot=..., d_loss=..., epoch_sum_init=0.500000
Expected: ...

Test 1: Original kernel
  Result: ...
  Status: PASS

Test 2: Replacement (host accumulate)
  Result: ...
  Status: PASS

=== All tests completed ===
```

### Benchmark Output

Both kernels should complete within microseconds. Host-based replacement typically shows **1.5–3× speedup** due to:
- Elimination of kernel launch overhead (~2–5 µs)
- Minimal D2H/H2D transfer cost (~1 µs for 12 bytes)
- Simpler cuBLAS overhead for saxpy approach

## Build Flags

```
-O3 -std=c++17 --gpu-architecture=sm_90
-I/usr/local/cuda/include -L/usr/local/cuda/lib64
-lcudart -lcublas -lcusparse -lnvToolsExt -lpthread
```

- **-O3**: Maximum optimization
- **--gpu-architecture=sm_90**: Target H100 GPU
- **-lcublas**: cuBLAS library (for saxpy alternative)
- **-lcusparse**: Included for consistency (not used)
- **-lnvToolsExt**: NVIDIA Tools Extension
- **-lpthread**: Pthreads (for cudaMemcpyAsync synchronization)

## Notes

- Kernel is **memory-bound** for host accumulation (no compute-to-memory ratio improvement)
- **Real speedup** depends on host ↔ device transfer latency and kernel launch cost
- For production, integrate replacement into `gpu_autoencoder.cu` near backward pass
- Consider **batching** multiple loss accumulations if feasible
