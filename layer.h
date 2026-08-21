// SPDX-License-Identifier: MIT
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "data_loader.h"

// Forward declaration
struct SparseBatch;

// Layer: a composable unit in a neural network stack.
//
// OWNERSHIP RATIONALE:
// This class is designed to be held via std::shared_ptr<Layer> and shared
// across multiple Autoencoder instances. The destructor releases all CUDA
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
    // Launches kernels and ops on the provided stream.
    const float* forward(const SparseView& x, cudaStream_t stream = 0);

    // Backward with sparse input. d_grad_output is device ptr to d L / d y
    // (out_dim × batch_size, column-major). Sparse layers cannot produce
    // gradient w.r.t. the input data, so no grad_input is returned.
    // Launches kernels and ops on the provided stream.
    void backward(const SparseView& x, const float* d_grad_output, cudaStream_t stream = 0);

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

    // --- Weight / bias initialization from host (for deterministic tests) ---

    // Copy row-major host weight matrix (size = out_dim * in_dim) to device d_W_.
    // Resets Adam moment buffers and timestep to 0.
    void set_weights_from_host(const float* host_W);  // size = out_dim * in_dim, row-major

    // Copy host bias vector (size = out_dim) to device d_b_.
    // Resets Adam moment buffers and timestep to 0.
    void set_biases_from_host(const float* host_b);   // size = out_dim

    // Reset Adam optimizer state (moment buffers m/v, timestep t) to zero.
    void reset_optimizer_state();

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

    // Read-only device pointer accessors (for validation/debugging)
    const float* d_W() const noexcept { return d_W_; }    // weights
    const float* d_dW() const noexcept { return d_dW_; }  // weight gradients (after backward)
    const float* d_b() const noexcept { return d_b_; }    // biases
    const float* d_db() const noexcept { return d_db_; }  // bias gradients (after backward)

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
    void _sp_forward(const SparseView& x, cudaStream_t stream);
    void _dn_forward(const float* d_in, int batch_size, cudaStream_t stream);
    void _sp_backward(const SparseView& x, const float* d_grad_output, cudaStream_t stream);
    void _dn_backward(const float* d_in, const float* d_grad_output,
                      bool compute_grad_input, cudaStream_t stream);
    void _apply_activation_forward(int batch_size, cudaStream_t stream);
    void _apply_activation_backward(int batch_size, cudaStream_t stream, const float* d_grad_output);
};
