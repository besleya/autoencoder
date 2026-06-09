// SPDX-License-Identifier: MIT
#include "autoencoder.h"

#include <cmath>

void Autoencoder::init(const std::vector<int>& layer_dims, std::mt19937& rng) {
    dims = layer_dims;
    const int L = static_cast<int>(dims.size()) - 1;

    W.resize(L);
    b.resize(L);
    mW.resize(L);
    vW.resize(L);
    mb.resize(L);
    vb.resize(L);

    for (int l = 0; l < L; ++l) {
        const int in  = dims[l];
        const int out = dims[l + 1];

        // He init: N(0, sqrt(2/in))
        const float stddev = std::sqrt(2.0f / static_cast<float>(in));
        std::normal_distribution<float> nd(0.0f, stddev);
        W[l].resize(out, in);
        for (int j = 0; j < in; ++j)
            for (int i = 0; i < out; ++i)
                W[l](i, j) = nd(rng);
        b[l] = Eigen::VectorXf::Zero(out);

        // Adam moments start at zero.
        mW[l] = Eigen::MatrixXf::Zero(out, in);
        vW[l] = Eigen::MatrixXf::Zero(out, in);
        mb[l] = Eigen::VectorXf::Zero(out);
        vb[l] = Eigen::VectorXf::Zero(out);
    }

    // Pre-size per-layer buffer slots; their actual matrices are (re)sized on
    // the first forward/backward.
    z.resize(L);
    a.resize(L + 1);
    dW.resize(L);
    db.resize(L);
    dz.resize(L);

    t = 0;
}

void Autoencoder::forward(const Eigen::MatrixXf& x) {
    a[0] = x;
    const int L = num_layers();
    for (int l = 0; l < L; ++l) {
        z[l].noalias() = W[l] * a[l];
        z[l].colwise() += b[l];
        if (l == L - 1) {
            a[l + 1] = z[l];                   // linear output
        } else {
            a[l + 1] = z[l].cwiseMax(0.0f);    // ReLU
        }
    }
}

float Autoencoder::backward_and_step(const Eigen::MatrixXf& x, float lr) {
    const int   L = num_layers();
    const float B = static_cast<float>(x.cols());

    // Scalar loss for reporting (mean over all entries).
    const float loss = (a[L] - x).squaredNorm() /
                       (static_cast<float>(x.rows()) * B);

    // ---- backward pass: gradients are ASSIGNED (overwritten), not accumulated ----
    for (int l = L - 1; l >= 0; --l) {
        if (l == L - 1) {
            // Output layer (linear): dz[l] = (a[L] - x) / B
            dz[l] = (a[L] - x) / B;
        } else {
            // Hidden layer (ReLU): dz[l] currently holds da[l+1] from the previous
            // iteration's tail. Apply ReLU mask in place.
            dz[l].array() *= (z[l].array() > 0.0f).cast<float>();
        }
        dW[l].noalias() = dz[l] * a[l].transpose();             // overwrite
        db[l]            = dz[l].rowwise().sum();               // overwrite
        // Propagate gradient to previous layer by writing into dz[l-1]
        if (l > 0) {
            dz[l - 1].noalias() = W[l].transpose() * dz[l];
        }
    }

    // ---- Adam update ----
    ++t;
    const float bc1   = 1.0f - std::pow(beta1, static_cast<float>(t));
    const float bc2   = 1.0f - std::pow(beta2, static_cast<float>(t));
    // Efficient form: lr_t * m / (sqrt(v) + eps_hat)
    const float lr_t  = lr * std::sqrt(bc2) / bc1;
    const float eps_h = eps;  // eps applied to sqrt(v) directly

    for (int l = 0; l < L; ++l) {
        // Weights
        mW[l] = beta1 * mW[l] + (1.0f - beta1) * dW[l];
        vW[l].array() = beta2 * vW[l].array() +
                        (1.0f - beta2) * dW[l].array().square();
        W[l].array() -= lr_t * mW[l].array() /
                        (vW[l].array().sqrt() + eps_h);

        // Biases
        mb[l] = beta1 * mb[l] + (1.0f - beta1) * db[l];
        vb[l].array() = beta2 * vb[l].array() +
                        (1.0f - beta2) * db[l].array().square();
        b[l].array() -= lr_t * mb[l].array() /
                        (vb[l].array().sqrt() + eps_h);
    }

    return loss;
}
