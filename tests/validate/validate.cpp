// SPDX-License-Identifier: MIT
// validate.cpp — deterministic validation of the C++ GPU autoencoder.
//
// Produces:
//   validate_loss.csv      — header "epoch,mse", 3 rows (post-update MSE per epoch)
//   validate_embedding.csv — 128×256 bottleneck activations (row=dim, col=cell)
//
// Build:  make tests/validate/validate   (from repo root)
// Run:    ./tests/validate/validate      (requires GPU)

#include "gpu_autoencoder.h"
#include "gpu_data_loader.h"
#include "layer.h"

#include <singlet/pileup/pz_reader.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

// ============================================================================
// Constants
// ============================================================================

static constexpr const char* INPUT_FILE =
    "/mnt/home/besleya/quant/GSE260931/GSM8128195/counts.1pz";
static constexpr const char* OUT_DIR =
    "/mnt/home/besleya/autoencoder/tests/validate/";

static constexpr int   N_COLS          = 256;
static constexpr int   HIDDEN          = 128;
static constexpr float LR              = 1e-3f;
static constexpr int   N_EPOCHS        = 3;
static constexpr float LOGNORM_SCALER  = 10000.0f;

// ============================================================================
// CUDA error check
// ============================================================================

#define CUDA_CHECK(call) do {                                             \
    cudaError_t _err = (call);                                            \
    if (_err != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_err));            \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

// ============================================================================
// main
// ============================================================================

int main() {
    // Set GPU device before any CUDA calls
    CUDA_CHECK(cudaSetDevice(0));

    // -----------------------------------------------------------------------
    // 1. Read .1pz file via singlet::pz::read_1pz
    // -----------------------------------------------------------------------
    fprintf(stderr, "[validate] reading %s\n", INPUT_FILE);
    singlet::pz::ReadResult r = singlet::pz::read_1pz(INPUT_FILE);

    const int m = (int)r.m;
    const int n = (int)r.n;
    const int B = N_COLS;

    fprintf(stderr, "[validate] m=%d  n=%d  nnz=%llu\n",
            m, n, (unsigned long long)r.nnz);

    if (n < B) {
        fprintf(stderr, "ERROR: file has only %d columns, need %d\n", n, B);
        return 1;
    }

    // -----------------------------------------------------------------------
    // 2. Slice first B columns (columns 0..255)
    //    indptr is uint32_t[n+1]; values are uint32_t
    // -----------------------------------------------------------------------
    const uint32_t nz_start = r.indptr[0];   // should be 0 for well-formed CSC
    const uint32_t nz_end   = r.indptr[B];
    const int      nnz      = (int)(nz_end - nz_start);

    std::vector<int32_t> h_col_ptr(B + 1);
    for (int c = 0; c <= B; ++c)
        h_col_ptr[c] = (int32_t)(r.indptr[c] - nz_start);

    std::vector<int32_t> h_row_idx(nnz);
    std::vector<float>   h_values(nnz);
    for (int k = 0; k < nnz; ++k) {
        h_row_idx[k] = (int32_t)r.indices[nz_start + k];
        h_values[k]  = (float)r.data[nz_start + k];   // uint32_t → float32
    }

    fprintf(stderr, "[validate] sliced B=%d cols, nnz=%d\n", B, nnz);

    // -----------------------------------------------------------------------
    // 3. Allocate device buffers, H2D copy
    // -----------------------------------------------------------------------
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int32_t* d_col_ptr = nullptr;
    int32_t* d_row_idx = nullptr;
    float*   d_values  = nullptr;

    CUDA_CHECK(cudaMalloc(&d_col_ptr, (size_t)(B + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_row_idx, (size_t)nnz      * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_values,  (size_t)nnz      * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_col_ptr, h_col_ptr.data(),
                          (size_t)(B + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_idx, h_row_idx.data(),
                          (size_t)nnz * sizeof(int32_t),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values,  h_values.data(),
                          (size_t)nnz * sizeof(float),       cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // 4. Log-normalize in place — same function as DataLoader's batch builder
    //    formula: values[k] = log1p(values[k] * 10000 / col_sum)
    // -----------------------------------------------------------------------
    log_normalize_csc_columns(B, LOGNORM_SCALER, d_col_ptr, d_values, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy log-normalised values back to host for the MSE computation later.
    std::vector<float> h_values_lognorm(nnz);
    CUDA_CHECK(cudaMemcpy(h_values_lognorm.data(), d_values,
                          (size_t)nnz * sizeof(float), cudaMemcpyDeviceToHost));

    // -----------------------------------------------------------------------
    // 5. Wrap in SparseBatch (we sync manually, so no ready_event needed)
    // -----------------------------------------------------------------------
    SparseBatch batch;
    batch.m           = m;
    batch.B           = B;
    batch.nnz         = nnz;
    batch.d_col_ptr   = d_col_ptr;
    batch.d_row_idx   = d_row_idx;
    batch.d_values    = d_values;
    batch.ready_event = nullptr;
    batch.eof_after   = true;

    // -----------------------------------------------------------------------
    // 6. Build autoencoder: [m, 128, m]
    // -----------------------------------------------------------------------
    std::mt19937 rng(0);   // consumed by init() for random weight init;
                            // we override immediately below so seed is irrelevant
    GpuAutoencoder ae;
    ae.init({m, HIDDEN, m}, rng);

    // -----------------------------------------------------------------------
    // 7. Override weights with deterministic formula (row-major):
    //      W[i,j] = (((i * in_dim + j) % 7) - 3) * 0.01f
    //    Biases: all zero.
    //    set_weights_from_host() also resets Adam moments to 0.
    // -----------------------------------------------------------------------

    // Encoder: layer 0  out=HIDDEN=128, in=m
    {
        const int out_dim = HIDDEN, in_dim = m;
        std::vector<float> W((size_t)out_dim * in_dim);
        for (int i = 0; i < out_dim; ++i)
            for (int j = 0; j < in_dim; ++j)
                W[(size_t)i * in_dim + j] =
                    (float)(((i * in_dim + j) % 7) - 3) * 0.01f;
        ae.layer(0)->set_weights_from_host(W.data());

        std::vector<float> b(out_dim, 0.0f);
        ae.layer(0)->set_biases_from_host(b.data());
    }

    // Decoder: layer 1  out=m, in=HIDDEN=128
    {
        const int out_dim = m, in_dim = HIDDEN;
        std::vector<float> W((size_t)out_dim * in_dim);
        for (int i = 0; i < out_dim; ++i)
            for (int j = 0; j < in_dim; ++j)
                W[(size_t)i * in_dim + j] =
                    (float)(((i * in_dim + j) % 7) - 3) * 0.01f;
        ae.layer(1)->set_weights_from_host(W.data());

        std::vector<float> b(out_dim, 0.0f);
        ae.layer(1)->set_biases_from_host(b.data());
    }

    // -----------------------------------------------------------------------
    // 7a. Write log-normalized data: first 10 columns
    //     (for comparing log-normalization between C++ and Python)
    // -----------------------------------------------------------------------
    {
        const int N_LOGNORM_COLS = 10;
        const std::string lognorm_path = std::string(OUT_DIR) + "validate_lognorm_10cols.csv";
        std::ofstream lognorm_file(lognorm_path);
        if (!lognorm_file) {
            fprintf(stderr, "ERROR: cannot open %s\n", lognorm_path.c_str());
            return 1;
        }

        // Densify first N_LOGNORM_COLS columns from CSC sparse format
        std::vector<float> h_dense_cols((size_t)m * N_LOGNORM_COLS, 0.0f);
        for (int j = 0; j < N_LOGNORM_COLS; ++j) {
            int start = h_col_ptr[j];
            int end   = h_col_ptr[j + 1];
            for (int k = start; k < end; ++k) {
                h_dense_cols[(size_t)h_row_idx[k] + (size_t)j * m] = h_values_lognorm[k];
            }
        }

        // Write row-by-row: m rows × 10 cols
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < N_LOGNORM_COLS; ++j) {
                if (j > 0) lognorm_file << ',';
                lognorm_file << std::setprecision(9) << h_dense_cols[(size_t)i + (size_t)j * m];
            }
            lognorm_file << '\n';
        }
        lognorm_file.close();
        fprintf(stderr, "[validate] wrote log-normalized 10 cols to %s\n", lognorm_path.c_str());
    }

    // -----------------------------------------------------------------------
    // 8. Open output files
    // -----------------------------------------------------------------------
    const std::string loss_path = std::string(OUT_DIR) + "validate_loss.csv";
    const std::string emb_path  = std::string(OUT_DIR) + "validate_embedding.csv";
    const std::string emb_epoch0_path = std::string(OUT_DIR) + "validate_embedding_epoch0.csv";

    std::ofstream loss_file(loss_path);
    if (!loss_file) {
        fprintf(stderr, "ERROR: cannot open %s\n", loss_path.c_str());
        return 1;
    }
    loss_file << "epoch,mse\n";

    // -----------------------------------------------------------------------
    // 8a. Pre-training forward pass: capture bottleneck embedding (epoch 0)
    //     This tests the forward pass before any training updates
    // -----------------------------------------------------------------------
    {
        ae.forward(batch, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<float> h_emb_epoch0((size_t)HIDDEN * B);
        CUDA_CHECK(cudaMemcpy(h_emb_epoch0.data(),
                              ae.layer(0)->output(),
                              (size_t)HIDDEN * B * sizeof(float),
                              cudaMemcpyDeviceToHost));

        std::ofstream emb_epoch0_file(emb_epoch0_path);
        if (!emb_epoch0_file) {
            fprintf(stderr, "ERROR: cannot open %s\n", emb_epoch0_path.c_str());
            return 1;
        }
        // h_emb_epoch0 is (HIDDEN × B) column-major: h_emb[i + j*HIDDEN] = element[i, j]
        // Output row i = embedding dimension i across all B cells
        for (int i = 0; i < HIDDEN; ++i) {
            for (int j = 0; j < B; ++j) {
                if (j > 0) emb_epoch0_file << ',';
                emb_epoch0_file << std::setprecision(9) << h_emb_epoch0[i + (size_t)j * HIDDEN];
            }
            emb_epoch0_file << '\n';
        }
        emb_epoch0_file.close();
        fprintf(stderr, "[validate] wrote pre-training embedding to %s\n", emb_epoch0_path.c_str());
    }

    // -----------------------------------------------------------------------
    // 9. Training loop — 3 epochs, full batch (all 256 columns), no shuffling
    // -----------------------------------------------------------------------
    for (int epoch = 1; epoch <= N_EPOCHS; ++epoch) {
        // --- Reset epoch loss accumulator ---
        ae.reset_epoch_loss();

        // --- Forward pass ---
        ae.forward(batch, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // --- Backward + Adam step ---
        ae.backward_and_step(batch, LR, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // --- Read epoch loss ---
        const float mse = ae.read_epoch_loss(1);  // num_batches=1

        loss_file << epoch << "," << std::setprecision(9) << mse << "\n";
        fprintf(stderr, "[validate] epoch %d  MSE=%.9f\n", epoch, mse);
    }
    loss_file.close();

    // -----------------------------------------------------------------------
    // 10. Write embedding CSV:
    //     layer(0)->output() = bottleneck post-ReLU, (HIDDEN × B) col-major
    //     CSV: 128 rows × 256 cols, row-major (row = embedding dimension,
    //                                          col = cell index)
    // -----------------------------------------------------------------------
    std::vector<float> h_emb((size_t)HIDDEN * B);
    CUDA_CHECK(cudaMemcpy(h_emb.data(),
                          ae.layer(0)->output(),
                          (size_t)HIDDEN * B * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::ofstream emb_file(emb_path);
    if (!emb_file) {
        fprintf(stderr, "ERROR: cannot open %s\n", emb_path.c_str());
        return 1;
    }
    // h_emb is (HIDDEN × B) column-major: h_emb[i + j*HIDDEN] = element[i, j]
    // Output row i = embedding dimension i across all B cells
    for (int i = 0; i < HIDDEN; ++i) {
        for (int j = 0; j < B; ++j) {
            if (j > 0) emb_file << ',';
            emb_file << std::setprecision(9) << h_emb[i + (size_t)j * HIDDEN];
        }
        emb_file << '\n';
    }
    emb_file.close();

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_col_ptr));
    CUDA_CHECK(cudaFree(d_row_idx));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaStreamDestroy(stream));

    fprintf(stderr, "[validate] done. Output written to %s\n", OUT_DIR);
    return 0;
}
