// SPDX-License-Identifier: MIT

// STRATEGY B (alternative: host-side concat then single H2D):
//
// This is an alternative implementation of load_dataset, kept for comparison.
// See data.cpp for Strategy A (device-side concat with per-file streams).
//
// Approach B loads all files with keep_host_pinned=true, so after each
// per-file stream syncs, host_indptr/host_indices/host_values are populated.
// Then the entire concat happens on host via OpenMP-parallel copy with offset
// addition. Finally, one bulk H2D to GPU with a single stream.
// Expected total PCIe bytes through GPU same as Strategy A, but no D2D step,
// and serial H2D after CPU concat instead of overlapped with decompression.

#include "data.h"

#include <singlet/singlet.h>
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/core/types.h>

#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// CUDA error checking macro
// ============================================================================

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t err = (expr);                                                \
    if (err != cudaSuccess) {                                                \
      std::ostringstream oss;                                                \
      oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "         \
          << cudaGetErrorString(err);                                        \
      throw std::runtime_error(oss.str());                                   \
    }                                                                         \
  } while (0)

// ============================================================================

Dataset load_dataset(const std::vector<std::string>& paths) {
    using PzDeviceMatrix = singlet::gpu::io::PzDeviceMatrix;

    const int F = static_cast<int>(paths.size());
    if (F == 0) {
        throw std::runtime_error("load_dataset: no files provided");
    }

    std::vector<PzDeviceMatrix> file_matrices(F);
    std::vector<std::string> errors;
    std::mutex err_mu;
    std::mutex log_mu;

    // ---- Step 1: OpenMP-parallel load with keep_host_pinned=true ----
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < F; ++i) {
        try {
            // load_pz with keep_host_pinned=true populates host_indptr,
            // host_indices, host_values after stream sync.
            file_matrices[i] = singlet::gpu::io::load_pz(
                paths[i], /*stream=*/nullptr, /*keep_host_pinned=*/true);

            std::lock_guard<std::mutex> lk(log_mu);
            std::cout << "  loaded " << paths[i]
                      << "  (" << file_matrices[i].n_genes << " x "
                      << file_matrices[i].n_cells
                      << ", nnz=" << file_matrices[i].mat.nnz << ')'
                      << std::endl;
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(err_mu);
            std::ostringstream oss;
            oss << "failed to load " << paths[i] << ": " << e.what();
            errors.push_back(oss.str());
        }
    }
    if (!errors.empty()) {
        throw std::runtime_error(errors.front());
    }

    // ---- Step 2: consistency check + size accounting ----
    Dataset ds;
    ds.m = file_matrices[0].n_genes;
    int64_t total_cols = 0;
    int64_t total_nnz = 0;

    std::vector<int> col_off(F + 1, 0);
    std::vector<int64_t> nnz_off(F + 1, 0);

    for (int i = 0; i < F; ++i) {
        if (file_matrices[i].n_genes != ds.m) {
            std::ostringstream oss;
            oss << "feature-count mismatch in " << paths[i] << ": got "
                << file_matrices[i].n_genes << ", expected " << ds.m;
            throw std::runtime_error(oss.str());
        }
        col_off[i + 1] = col_off[i] + file_matrices[i].n_cells;
        nnz_off[i + 1] = nnz_off[i] + file_matrices[i].mat.nnz;

        total_cols += file_matrices[i].n_cells;
        total_nnz += file_matrices[i].mat.nnz;
    }

    if (total_cols > std::numeric_limits<int>::max() ||
        total_nnz > std::numeric_limits<int>::max()) {
        throw std::runtime_error("dataset too large for 32-bit indices");
    }

    ds.n = static_cast<int>(total_cols);
    ds.nnz = static_cast<int64_t>(total_nnz);

    // ---- Step 3: sync per-file streams to ensure host buffers are ready ----
    for (int i = 0; i < F; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(file_matrices[i].producer_stream));
    }

    // ---- Step 4: allocate three PinnedBuffer for host-side concat ----
    size_t col_ptr_bytes = (ds.n + 1) * sizeof(int);
    size_t indices_bytes = ds.nnz * sizeof(int);
    size_t values_bytes = ds.nnz * sizeof(float);

    singlet::gpu::core::PinnedBuffer pinned_col_ptr =
        singlet::gpu::core::PinnedPool::acquire(col_ptr_bytes);
    singlet::gpu::core::PinnedBuffer pinned_indices =
        singlet::gpu::core::PinnedPool::acquire(indices_bytes);
    singlet::gpu::core::PinnedBuffer pinned_values =
        singlet::gpu::core::PinnedPool::acquire(values_bytes);

    int* col_ptr_host = pinned_col_ptr.as<int>();
    int* indices_host = pinned_indices.as<int>();
    float* values_host = pinned_values.as<float>();

    // ---- Step 5: OpenMP-parallel concat from per-file host buffers ----
    col_ptr_host[0] = 0;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < F; ++i) {
        const int* file_col_ptr = file_matrices[i].pinned_indptr.as<int>();
        const int* file_indices = file_matrices[i].pinned_indices.as<int>();
        const float* file_values =
            file_matrices[i].pinned_values.as<float>();

        const int file_n = file_matrices[i].n_cells;
        const int file_nnz = file_matrices[i].mat.nnz;
        const int col_o = col_off[i];
        const int64_t nnz_o = nnz_off[i];

        // col_ptr: copy with offset added
        for (int c = 0; c < file_n; ++c) {
            col_ptr_host[col_o + c + 1] =
                static_cast<int>(file_col_ptr[c + 1] + nnz_o);
        }

        // indices and values: direct copy
        for (int k = 0; k < file_nnz; ++k) {
            indices_host[nnz_o + k] = file_indices[k];
            values_host[nnz_o + k] = file_values[k];
        }
    }

    // ---- Step 6: allocate DeviceCSC and transfer host → device ----
    ds.X = singlet::gpu::core::DeviceCSC(ds.m, ds.n,
                                         static_cast<int>(total_nnz));

    cudaStream_t transfer_stream;
    CUDA_CHECK(
        cudaStreamCreateWithPriority(&transfer_stream, cudaStreamNonBlocking,
                                     -1));

    // Three H2D copies
    CUDA_CHECK(cudaMemcpyAsync(ds.X.col_ptr.get(), col_ptr_host,
                               col_ptr_bytes, cudaMemcpyHostToDevice,
                               transfer_stream));

    CUDA_CHECK(cudaMemcpyAsync(ds.X.row_indices.get(), indices_host,
                               indices_bytes, cudaMemcpyHostToDevice,
                               transfer_stream));

    CUDA_CHECK(cudaMemcpyAsync(ds.X.values.get(), values_host, values_bytes,
                               cudaMemcpyHostToDevice, transfer_stream));

    // Sync transfer
    CUDA_CHECK(cudaStreamSynchronize(transfer_stream));

    // ---- Step 7: cleanup ----
    CUDA_CHECK(cudaStreamDestroy(transfer_stream));

    for (int i = 0; i < F; ++i) {
        CUDA_CHECK(cudaStreamDestroy(file_matrices[i].producer_stream));
    }

    file_matrices.clear();

    return ds;
}
