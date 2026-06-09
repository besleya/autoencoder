// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Dense>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Autoencoder with Adam optimizer and MSE reconstruction loss.
// Hidden layers use ReLU; the final decoder output layer is linear.
//
// Layer convention: `dims` is the full layer dimension list,
//   e.g. {m, 512, 128, 32, 128, 512, m}.
// W[l] has shape (dims[l+1], dims[l]); b[l] has length dims[l+1].
// ---------------------------------------------------------------------------
struct Autoencoder {
    // ---- parameters ----
    std::vector<int> dims;
    std::vector<Eigen::MatrixXf> W;
    std::vector<Eigen::VectorXf> b;

    // ---- Adam state ----
    // First/second moment estimates, same shapes as W and b.
    std::vector<Eigen::MatrixXf> mW, vW;
    std::vector<Eigen::VectorXf> mb, vb;
    int   t     = 0;        // timestep, incremented per optimizer step
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps   = 1e-8f;

    // ---- reusable forward/backward buffers (sized for the current batch) ----
    std::vector<Eigen::MatrixXf> z;   // pre-activations,  z[l] = W[l]*a[l] + b[l]
    std::vector<Eigen::MatrixXf> a;   // activations,      a[0] = x, a[L] = output
    std::vector<Eigen::MatrixXf> dW;  // dL/dW[l]
    std::vector<Eigen::VectorXf> db;  // dL/db[l]
    std::vector<Eigen::MatrixXf> dz;  // dL/dz[l]

    int num_layers() const { return static_cast<int>(W.size()); }

    // Initialize parameters (He init), Adam state (zeros), and buffer slots.
    void init(const std::vector<int>& layer_dims, std::mt19937& rng);

    // Forward pass. `x` has shape (dims[0], B). On return, a[L] holds the
    // reconstruction.
    void forward(const Eigen::MatrixXf& x);

    // Backprop MSE loss against target `x`, run one Adam step, and return the
    // mean-squared-error (averaged over batch and features) for monitoring.
    //
    // NOTE: Each call OVERWRITES dW[l] and db[l] (assignment, not +=), so
    // gradients do NOT accumulate across mini-batches.
    float backward_and_step(const Eigen::MatrixXf& x, float lr);
};
