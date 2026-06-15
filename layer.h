// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include "gpu_data_loader.h"

// Forward declaration
struct SparseBatch;

// Layer: a composable unit in a neural network stack.
//
// OWNERSHIP RATIONALE:
// This class is designed to be held via std::shared_ptr<Layer> and shared
// across multiple GpuAutoencoder instances. The destructor releases all CUDA
// device buffers (weights, biases, activations, gradients) via cudaFree().
// std::shared_ptr ensures the destructor runs exactly once when the last
// reference is released, preventing double-free and resource leaks even if
// the layer is shared and accessed from multiple contexts.
//
// SHARED-LAYER TRAINING INVARIANT:
// When a Layer is shared between two autoencoders during training, the caller
// MUST complete autoencoder A's full forward → backward → update sequence
// before starting autoencoder B's forward pass. The per-batch activation and
// gradient buffers (d_z_, d_y_, d_dz_, d_dW_, d_db_) are sized once and
// reused. Interleaving forward/backward calls between autoencoders corrupts
// these buffers and produces incorrect gradients.
class Layer {
public:
    enum class Activation { None, ReLU };

    // Constructor: allocates weights, biases, and Adam state on device.
    // Per-batch buffers (activations, gradients) are allocated lazily on first forward().
    // sparse_input is IMMUTABLE after construction; determines which forward/backward
    // overload must be called.
    // cublas_handle / cusparse_handle are non-owning (borrowed references; must
    // remain valid for the Layer's lifetime or until no CUDA kernels referencing
    // them are in flight).
    Layer(int in_dim,
          int out_dim,
          Activation activation,
          bool sparse_input,
          unsigned long seed,
          cublasHandle_t cublas_handle,
          cusparseHandle_t cusparse_handle);

    ~Layer();

    // Non-copyable
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;

    // --- Sparse-input overloads (require sparse_input() == true) ---

    // Forward with sparse input. Returns device pointer to output activations
    // (out_dim × batch_size, column-major). Batch size is inferred from x.B.
    const float* forward(const SparseBatch& x);

    // Backward with sparse input. d_grad_output is device ptr to d L / d y
    // (out_dim × batch_size, column-major). Sparse layers cannot produce
    // gradient w.r.t. the input data, so no grad_input is returned.
    void backward(const SparseBatch& x, const float* d_grad_output);

    // --- Dense-input overloads (require sparse_input() == false) ---

    // Forward with dense input. d_in is device ptr to input (in_dim × batch_size,
    // column-major). Returns device pointer to output activations (out_dim × batch_size,
    // column-major).
    const float* forward(const float* d_in, int batch_size, cudaStream_t stream = 0);

    // Backward with dense input. d_in, d_grad_output are device ptrs
    // (same shapes as forward). If compute_grad_input is true, returns device ptr
    // to d L / d x (in_dim × batch_size, column-major); else returns nullptr.
    const float* backward(const float* d_in,
                          const float* d_grad_output,
                          bool compute_grad_input,
                          cudaStream_t stream = 0);

    // Adam optimization step. Uses gradients accumulated by the last backward() call.
    // Updates weights and biases in-place, increments timestep counter.
    void update(float lr, cudaStream_t stream = 0);

    // --- Accessors ---

    int in_dim() const noexcept;
    int out_dim() const noexcept;
    Activation activation() const noexcept;
    bool sparse_input() const noexcept;
    const float* output() const noexcept;      // d_y_, valid after forward()
    int last_batch_size() const noexcept;
    const float* weights() const noexcept;     // d_W_, (out, in) col-major
    const float* bias() const noexcept;        // d_b_, length out
    int timestep() const noexcept;             // Adam t

private:
    // Configuration
    int in_dim_;
    int out_dim_;
    Activation activation_;
    bool sparse_input_;

    // Handles (non-owning)
    cublasHandle_t cublas_handle_;
    cusparseHandle_t cusparse_handle_;

    // Device parameters (owned)
    float* d_W_;    // (out, in) col-major
    float* d_b_;    // length out
    float* d_mW_;   // Adam moment, same shape as W
    float* d_vW_;   // Adam second moment, same shape as W
    float* d_mb_;   // Adam moment, length out
    float* d_vb_;   // Adam second moment, length out

    // Per-batch buffers (owned, allocated lazily)
    float* d_z_;           // (out, batch_size) col-major
    float* d_y_;           // (out, batch_size) col-major
    float* d_dz_;          // (out, batch_size) col-major
    float* d_dW_;          // (out, in) col-major
    float* d_db_;          // length out
    float* d_grad_input_;  // (in, batch_size) col-major, lazily allocated
    float* d_ones_;        // length batch_size, for bias broadcast

    // State
    int last_batch_size_;
    int t_;  // Adam timestep counter

    // Private helpers
    void _ensure_batch_buffers(int batch_size, bool need_grad_input);
    void _sp_forward(const SparseBatch& x);
    void _dn_forward(const float* d_in, int batch_size, cudaStream_t stream);
    void _sp_backward(const SparseBatch& x, const float* d_grad_output);
    void _dn_backward(const float* d_in, const float* d_grad_output,
                      bool compute_grad_input, cudaStream_t stream);
    void _apply_activation_forward(int batch_size, cudaStream_t stream);
    void _apply_activation_backward(int batch_size, cudaStream_t stream, const float* d_grad_output);
};
