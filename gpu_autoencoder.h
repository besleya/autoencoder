// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>
#include "gpu_data_loader.h"
#include "layer.h"

// Autoencoder on GPU with per-layer math delegated to the Layer class.
// The autoencoder is responsible for:
// - Managing the layer stack (construction, validation, lifetime via shared_ptr)
// - Computing reconstruction loss and loss gradient
// - Orchestrating forward/backward/update across all layers
//
// SHARED-LAYER TRAINING INVARIANT:
// When a Layer is shared between two GpuAutoencoder instances during training,
// the caller MUST complete autoencoder A's full forward → backward → update
// sequence before starting autoencoder B's forward pass. The per-batch buffers
// inside each shared Layer (d_z_, d_y_, d_dz_, etc.) are sized once per layer
// and overwritten by each forward() call. Interleaving forward/backward calls
// between autoencoders corrupts these buffers and produces incorrect gradients.
//
// OWNERSHIP RATIONALE:
// Layers are held as std::shared_ptr<Layer> so that the same Layer object can
// be shared across multiple GpuAutoencoder instances with correct lifetime
// management and zero per-autoencoder ownership bookkeeping. Each Layer frees
// its own device memory when its reference count drops to zero.
//
// LAYER VALIDATION (caller-supplied constructor):
// When the caller supplies pre-constructed layers, the autoencoder validates
// that each layer's in_dim, out_dim, and sparse_input flag match the expected
// topology. Mismatches cause an error message to stderr and exit().
class GpuAutoencoder {
public:
    GpuAutoencoder();
    ~GpuAutoencoder();

    // Initialize with all-layers-owned: constructs every layer from the given
    // topology. First layer is constructed with sparse_input=true; all others
    // with sparse_input=false. Activation policy: hidden layers use ReLU, the
    // final (output) layer uses None. Parameters are initialized using He init
    // (for ReLU) or Xavier (for None) via the provided RNG. RNG state is
    // consumed in the same order as the CPU autoencoder to allow bit-equivalent
    // training.
    void init(const std::vector<int>& layer_dims, std::mt19937& rng);

    // Initialize with caller-supplied layers: takes a topology and a vector of
    // pre-constructed layers. Any null entry is constructed by the autoencoder;
    // any non-null entry is validated (in_dim, out_dim, sparse_input must match
    // the topology) and taken as-is. This allows the caller to share layers
    // across multiple autoencoders or provide custom layer configurations.
    void init(const std::vector<int>& layer_dims,
              const std::vector<std::shared_ptr<Layer>>& layers_in,
              std::mt19937& rng);

    // Forward pass. Stores the last batch size internally. Layer 0 input is the
    // sparse mini-batch; deeper layers receive the previous layer's output.
    // On return, the reconstruction is available via the last layer's output().
    // Launches kernels and H2D ops on the provided stream.
    void forward(const SparseBatch& x, cudaStream_t stream);

    // Backprop MSE loss against target `x` (sparse), run one Adam step.
    // Per-batch loss is accumulated into a device-side epoch sum and read via
    // read_epoch_loss(). Mirrors the original autoencoder's semantics including
    // the intentional asymmetry between loss (divided by d0*B) and loss gradient
    // (divided by B only). Launches kernels and ops on the provided stream.
    void backward_and_step(const SparseBatch& x, float lr, cudaStream_t stream);

    // Reset the device-side epoch loss accumulator to 0. Call once at the
    // start of each epoch, BEFORE the batch loop. Synchronous on the default
    // stream (cheap; called once per epoch).
    void reset_epoch_loss();

    // Read back the accumulated epoch loss from device, divide by num_batches,
    // and return the per-epoch mean reconstruction loss. Synchronizes the
    // default stream. Call ONCE per epoch after the batch loop.
    float read_epoch_loss(int num_batches);

    // --- Public accessors ---
    int num_layers() const noexcept;
    std::shared_ptr<Layer> layer(int i) const;
    const std::vector<std::shared_ptr<Layer>>& layers() const noexcept;

private:
    // Layer topology and stack
    std::vector<int> dims_;
    std::vector<std::shared_ptr<Layer>> layers_;
    int num_l_;  // L = dims.size() - 1 = layers_.size()

    // Batch state
    int batch_size_;
    bool initialized_buffers_;

    // Autoencoder-level device buffers (owned)
    float* d_grad_loss_;  // Loss gradient, (dims_[L] × batch_size_) col-major
    float* d_loss_;       // Scalar loss accumulator on device
    float* d_dot_;              // Scalar device buffer for cublasSdot result (||a_L||^2)
    float* d_epoch_loss_sum_;   // Device scalar: sum over batches of per-batch mean loss

    // cuBLAS / cuSPARSE handles (owned)
    cublasHandle_t cublas_handle_;
    cusparseHandle_t cusparse_handle_;

    // Allocate autoencoder-level buffers (d_grad_loss_, d_loss_)
    void allocate_buffers_();

    // Deallocate autoencoder-level buffers
    void deallocate_buffers_();

    // Helper: construct a single layer given topology, activation, and seed
    std::shared_ptr<Layer> construct_layer_(int in_dim, int out_dim,
                                             Layer::Activation activation,
                                             bool sparse_input,
                                             unsigned long seed);
};
