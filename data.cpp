// SPDX-License-Identifier: MIT
#include "data.h"

#include <singlet/singlet.h>

#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// Parallel-load all .1pz files, then assemble a single CSC SpMatF directly
// from the per-file ReadResults.
//
// Why this is faster than the previous "load each as SpMatF then re-insert
// element by element" pipeline:
//   * No uint32 -> int32 index copy per file.
//   * No PZMatrix .eval() copy per file.
//   * No per-file .cast<float>() copy.
//   * No per-element SpMatF::insert() into the concatenated matrix; instead
//     a single bulk Eigen::Map + assignment.
//   * File I/O / decompression runs in parallel across files via OpenMP.
Dataset load_dataset(const std::vector<std::string>& paths) {
    using ReadResult = singlet::pz::ReadResult;

    const int F = static_cast<int>(paths.size());
    std::vector<ReadResult> results(F);
    std::vector<std::string> errors;
    std::mutex err_mu;
    std::mutex log_mu;

    // ---- parallel file read ----
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < F; ++i) {
        try {
            results[i] = singlet::pz::read_1pz(paths[i]);
            std::lock_guard<std::mutex> lk(log_mu);
            std::cout << "  loaded " << paths[i]
                      << "  (" << results[i].m << " x " << results[i].n
                      << ", nnz=" << results[i].nnz << ')' << std::endl;
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(err_mu);
            std::ostringstream oss;
            oss << "failed to read " << paths[i] << ": " << e.what();
            errors.push_back(oss.str());
        }
    }
    if (!errors.empty()) {
        throw std::runtime_error(errors.front());
    }

    // ---- consistency check + size accounting ----
    Dataset ds;
    ds.m = results[0].m;
    uint64_t total_cols = 0;
    uint64_t total_nnz  = 0;
    for (int i = 0; i < F; ++i) {
        if (results[i].m != ds.m) {
            std::ostringstream oss;
            oss << "feature-count mismatch in " << paths[i]
                << ": got " << results[i].m << ", expected " << ds.m;
            throw std::runtime_error(oss.str());
        }
        total_cols += results[i].n;
        total_nnz  += results[i].nnz;
    }
    if (total_cols > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        total_nnz  > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("dataset too large for 32-bit Eigen indices");
    }

    // ---- bulk-concatenate CSC arrays ----
    using StorageIndex = SpMatF::StorageIndex;  // typically int

    std::vector<StorageIndex> outer(total_cols + 1);
    std::vector<StorageIndex> inner(total_nnz);
    std::vector<float>        values(total_nnz);

    // Per-file offsets so we can fill independently in parallel.
    std::vector<uint64_t> col_off(F + 1, 0);
    std::vector<uint64_t> nnz_off(F + 1, 0);
    for (int i = 0; i < F; ++i) {
        col_off[i + 1] = col_off[i] + results[i].n;
        nnz_off[i + 1] = nnz_off[i] + results[i].nnz;
    }
    outer[0] = 0;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < F; ++i) {
        const auto& r = results[i];
        const uint64_t co = col_off[i];
        const uint64_t no = nnz_off[i];

        // outer[co + 1 .. co + r.n] = r.indptr[1..r.n] + no
        // (entry co already written either by outer[0]=0 above or by the
        // previous file's last write; using +1 indexing keeps it lock-free.)
        for (uint32_t c = 0; c < r.n; ++c) {
            outer[co + c + 1] =
                static_cast<StorageIndex>(r.indptr[c + 1] + no);
        }
        // inner / values: cast in bulk.
        for (uint64_t k = 0; k < r.nnz; ++k) {
            inner[no + k]  = static_cast<StorageIndex>(r.indices[k]);
            values[no + k] = static_cast<float>(r.data[k]);
        }
    }

    // Free per-file ReadResult buffers before the (potential) Eigen copy.
    results.clear();
    results.shrink_to_fit();

    // Wrap raw arrays as a CSC matrix and assign once.
    Eigen::Map<const SpMatF> map(
        static_cast<StorageIndex>(ds.m),
        static_cast<StorageIndex>(total_cols),
        static_cast<StorageIndex>(total_nnz),
        outer.data(), inner.data(), values.data());
    ds.X = map;
    ds.X.makeCompressed();
    return ds;
}
