// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <random>
#include <vector>
#include "gpu_data_loader.h"

// Autoencoder on GPU with cuBLAS (dense GEMM), cuSPARSE (sparse layer 0),
// and custom kernels (ReLU, Adam, loss reduction).
// API mirrors the CPU Autoencoder exactly.
class GpuAutoencoder {
public:
    GpuAutoencoder();
    ~GpuAutoencoder();

    // Initialize parameters (He init), Adam state (zeros), and allocate device
    // buffers. Must consume RNG state in EXACTLY the same order as the CPU
    // autoencoder to allow bit-equivalent single-file training.
    // 
    // Implementation: draw on HOST using std::normal_distribution<float>,
    // then upload to device.
    void init(const std::vector<int>& layer_dims, std::mt19937& rng);

    // Forward pass. Stores activations on device. Layer 0 input is the sparse
    // mini-batch; deeper layers are dense (cuBLAS GEMM). On return, a[L]
    // holds the reconstruction.
    void forward(const SparseBatch& x);

    // Backprop MSE loss against target `x` (sparse), run one Adam step, and
    // return the mean-squared-error (averaged over batch and features) for
    // monitoring. Mirrors CPU semantics including the intentional asymmetry
    // between loss (divided by d0*B) and dz[L-1] (divided by B only).
    float backward_and_step(const SparseBatch& x, float lr);

    int num_layers() const;

private:
    // ---- layer dimensions ----
    std::vector<int> dims;
    int num_l;  // L = dims.size() - 1
    int batch_size;  // B, fixed after first forward (assumed constant)
    bool initialized_buffers;

    // ---- device parameters ----
    std::vector<float*> d_W;   // W[l]: (dims[l+1] × dims[l]) col-major
    std::vector<float*> d_b;   // b[l]: length dims[l+1]
    std::vector<float*> d_mW;  // mW[l]: same shape as W[l]
    std::vector<float*> d_vW;  // vW[l]: same shape as W[l]
    std::vector<float*> d_mb;  // mb[l]: same shape as b[l]
    std::vector<float*> d_vb;  // vb[l]: same shape as b[l]

    // ---- device activations and gradients ----
    std::vector<float*> d_a;   // a[l]: (dims[l] × B) col-major, l = 0..L
    std::vector<float*> d_z;   // z[l]: (dims[l+1] × B) col-major, l = 0..L-1
    std::vector<float*> d_dz;  // dz[l]: same shape as z[l]
    std::vector<float*> d_dW;  // dW[l]: same shape as W[l]
    std::vector<float*> d_db;  // db[l]: same shape as b[l]

    // ---- scalar buffers ----
    float* d_loss;  // device scalar for loss reduction
    float* d_ones;  // device vector (length max_B) of ones for bias sum

    // ---- Adam state ----
    int t;
    static constexpr float beta1 = 0.9f;
    static constexpr float beta2 = 0.999f;
    static constexpr float eps   = 1e-8f;

    // ---- cuBLAS / cuSPARSE handles ----
    void* cublas_handle;
    void* cusparse_handle;

    // Allocate device buffers on first forward (after batch_size is known).
    void allocate_buffers();

    // Deallocate all device buffers.
    void deallocate_buffers();
};
