// SPDX-License-Identifier: MIT

// STRATEGY A (device-side concat with per-file streams):
//
// Approach A overlaps CPU zstd decompression with PCIe H2D copies (per-file
// streams), then concatenates entirely on-GPU via cheap HBM3→HBM3 D2D copies
// (~3 TB/s vs ~64 GB/s PCIe Gen5). The unified col_ptr is built on host
// because it requires per-entry offset addition and is tiny (≤ a few MB).
// Per-file stream policy chosen for maximum producer overlap; cost of stream
// creation (microseconds) is negligible vs decompression.

#include "data.h"
#include "load_pz.h"

#include <singlet/singlet.h>
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/core/types.h>

#include <cuda_runtime.h>
#include <cstring>
#include <fstream>
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

PZHeader validate_1pz(const std::string& path) {
    // Open the file in binary mode
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("validate_1pz: cannot open " + path);
    }

    // Seek to end and get file size
    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size < static_cast<std::streamsize>(sizeof(PZHeader) + sizeof(PZFooter))) {
        std::ostringstream oss;
        oss << "validate_1pz: file " << path << " too small (" << file_size
            << " bytes, need at least " << (sizeof(PZHeader) + sizeof(PZFooter))
            << " bytes)";
        throw std::runtime_error(oss.str());
    }

    // Read header
    PZHeader hdr;
    file.read(reinterpret_cast<char*>(&hdr), sizeof(PZHeader));
    if (!file.good()) {
        throw std::runtime_error("validate_1pz: failed to read header from " + path);
    }

    // Check magic
    if (hdr.magic != TP1_MAGIC_LOCAL) {
        std::ostringstream oss;
        oss << "validate_1pz: bad magic in " << path << " (got 0x" << std::hex
            << hdr.magic << ", expected 0x" << TP1_MAGIC_LOCAL << ")";
        throw std::runtime_error(oss.str());
    }

    // Check version
    if (hdr.version != 1 && hdr.version != 3 && hdr.version != 4) {
        std::ostringstream oss;
        oss << "validate_1pz: unsupported version in " << path << " (got "
            << hdr.version << ", expected 1, 3, or 4)";
        throw std::runtime_error(oss.str());
    }

    // Seek to footer and read it
    file.seekg(file_size - static_cast<std::streamsize>(sizeof(PZFooter)), std::ios::beg);
    PZFooter ftr;
    file.read(reinterpret_cast<char*>(&ftr), sizeof(PZFooter));
    if (!file.good()) {
        throw std::runtime_error("validate_1pz: failed to read footer from " + path);
    }

    // Check footer magic
    if (ftr.magic != TP1_MAGIC_LOCAL) {
        throw std::runtime_error("validate_1pz: bad footer magic in " + path);
    }

    // Check footer num_chunks matches header
    if (ftr.num_chunks != hdr.num_chunks) {
        std::ostringstream oss;
        oss << "validate_1pz: footer num_chunks (" << ftr.num_chunks
            << ") does not match header (" << hdr.num_chunks << ") in " << path;
        throw std::runtime_error(oss.str());
    }

    // Note: CRC32 is intentionally NOT validated because we read only the
    // header and footer (no slurp of the entire file).

    return hdr;
}

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

    // ---- Step 1: OpenMP-parallel load with per-file streams ----
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < F; ++i) {
        try {
            // load_pz with stream=nullptr creates a high-priority stream
            // and stores it in file_matrices[i].producer_stream.
            // We keep the pinned host buffers (pinned_indptr, etc).
            file_matrices[i] =
                singlet::gpu::io::load_pz_v3(paths[i], /*stream=*/nullptr,
                                          /*keep_host_pinned=*/false);

            std::lock_guard<std::mutex> lk(log_mu);
            // std::cout << "  loaded " << paths[i]
            //           << "  (" << file_matrices[i].n_genes << " x "
            //           << file_matrices[i].n_cells
            //           << ", nnz=" << file_matrices[i].mat.nnz << ')'
            //           << std::endl;
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

    // ---- Step 3: allocate unified DeviceCSC ----
    ds.X = singlet::gpu::core::DeviceCSC(ds.m, ds.n,
                                         static_cast<int>(total_nnz));

    // ---- Step 4: build unified col_ptr on host from per-file pinned
    // indptr ----

    // Allocate a single PinnedBuffer to stage the unified col_ptr.
    size_t col_ptr_bytes = (ds.n + 1) * sizeof(int);
    singlet::gpu::core::PinnedBuffer staging_col_ptr =
        singlet::gpu::core::PinnedPool::acquire(col_ptr_bytes);
    int* col_ptr_host = staging_col_ptr.as<int>();

    // Build the col_ptr from per-file pinned indptr buffers.
    col_ptr_host[0] = 0;
    for (int i = 0; i < F; ++i) {
        const int* file_indptr =
            file_matrices[i].pinned_indptr.as<int>();
        const int file_n = file_matrices[i].n_cells;
        const int64_t nnz_o = nnz_off[i];

        // col_ptr_host[col_off[i] + 1 .. col_off[i] + file_n] =
        //   file_indptr[1..file_n] + nnz_o
        for (int c = 0; c < file_n; ++c) {
            col_ptr_host[col_off[i] + c + 1] =
                static_cast<int>(file_indptr[c + 1] + nnz_o);
        }
    }

    // One H2D copy of the unified col_ptr.
    cudaStream_t concat_stream;
    CUDA_CHECK(cudaStreamCreateWithPriority(
        &concat_stream, cudaStreamNonBlocking, -1));

    CUDA_CHECK(cudaMemcpyAsync(ds.X.col_ptr.get(), col_ptr_host,
                               col_ptr_bytes, cudaMemcpyHostToDevice,
                               concat_stream));

    // ---- Step 5: sync per-file streams, then D2D copies on concat stream
    // ----

    for (int i = 0; i < F; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(file_matrices[i].producer_stream));

        const int64_t nnz_o = nnz_off[i];
        const int nnz_i = file_matrices[i].mat.nnz;

        // D2D: row_indices and values
        CUDA_CHECK(cudaMemcpyAsync(
            ds.X.row_indices.get() + nnz_o,
            file_matrices[i].mat.row_indices.get(),
            nnz_i * sizeof(int), cudaMemcpyDeviceToDevice, concat_stream));

        CUDA_CHECK(cudaMemcpyAsync(
            ds.X.values.get() + nnz_o, file_matrices[i].mat.values.get(),
            nnz_i * sizeof(float), cudaMemcpyDeviceToDevice, concat_stream));
    }

    // Sync the concat stream to ensure all D2D copies are done.
    CUDA_CHECK(cudaStreamSynchronize(concat_stream));

    // ---- Step 6: cleanup streams ----

    for (int i = 0; i < F; ++i) {
        CUDA_CHECK(cudaStreamDestroy(file_matrices[i].producer_stream));
    }
    CUDA_CHECK(cudaStreamDestroy(concat_stream));

    // ---- Step 7: release per-file matrices (they go out of scope here,
    // freeing their device buffers) ----
    file_matrices.clear();

    return ds;
}
