// SPDX-License-Identifier: MIT
// load_pz.cpp — implementation of load_pz_v3.

#include "load_pz.h"
#include <singlet/pileup/pz_reader.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <sstream>

namespace singlet::gpu {
namespace io {

namespace {

// Saturating uint32 → float cast with a one-time stderr warning.
// Values beyond 2^24 lose mantissa bits in fp32.
inline float saturate_cast_f32(uint32_t v, bool& warned) {
    static constexpr uint32_t FP32_EXACT_INT_LIMIT = (1u << 24);
    if (v > FP32_EXACT_INT_LIMIT && !warned) {
        std::fprintf(stderr,
            "[singlet/gpu/load_pz_v3] WARNING: value %u > 2^24 (%u); "
            "fp32 cast loses integer precision.\n", v, FP32_EXACT_INT_LIMIT);
        warned = true;
    }
    return static_cast<float>(v);
}

// CUDA error check macro.
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw std::runtime_error( \
            std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

}  // anonymous namespace

// load_pz_v3 implementation
singlet::gpu::io::PzDeviceMatrix load_pz_v3(
    const std::string& path,
    cudaStream_t stream,
    bool keep_host_pinned)
{
    // --- 1. Decode file on CPU using read_1pz ---
    singlet::pz::ReadResult r = singlet::pz::read_1pz(path);

    // --- 2. Allocate pinned host staging buffers ---
    const size_t indptr_bytes = static_cast<size_t>(r.n + 1) * sizeof(int32_t);
    const size_t indices_bytes = static_cast<size_t>(r.nnz) * sizeof(int32_t);
    const size_t values_bytes = static_cast<size_t>(r.nnz) * sizeof(float);

    auto pinned_indptr = core::PinnedPool::acquire(indptr_bytes);
    auto pinned_indices = core::PinnedPool::acquire(indices_bytes);
    auto pinned_values = core::PinnedPool::acquire(values_bytes);

    int32_t* h_indptr = pinned_indptr.as<int32_t>();
    int32_t* h_indices = pinned_indices.as<int32_t>();
    float* h_values = pinned_values.as<float>();

    // --- 3. Fill indptr from r.indptr (convert uint32 to int32) ---
    for (uint32_t j = 0; j <= r.n; ++j) {
        h_indptr[j] = static_cast<int32_t>(r.indptr[j]);
    }

    // --- 4. Fill indices and values with saturation cast ---
    bool fp32_warn = false;
    for (uint64_t i = 0; i < r.nnz; ++i) {
        h_indices[i] = static_cast<int32_t>(r.indices[i]);
        h_values[i] = saturate_cast_f32(r.data[i], fp32_warn);
    }

    // --- 5. Allocate device CSC ---
    core::DeviceCSC d_mat(
        static_cast<int>(r.m),
        static_cast<int>(r.n),
        static_cast<int>(r.nnz));

    // --- 6. Create or use provided stream ---
    bool stream_created = false;
    if (stream == nullptr) {
        int least, greatest;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest));
        stream_created = true;
    }

    // --- 7. Issue async copies ---
    CUDA_CHECK(cudaMemcpyAsync(
        d_mat.col_ptr.get(),
        h_indptr,
        indptr_bytes,
        cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMemcpyAsync(
        d_mat.row_indices.get(),
        h_indices,
        indices_bytes,
        cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMemcpyAsync(
        d_mat.values.get(),
        h_values,
        values_bytes,
        cudaMemcpyHostToDevice, stream));

    // --- 8. Populate metadata from ReadResult ---
    core::Metadata meta;
    meta.rownames = r.rownames;
    meta.colnames = r.colnames;

    // Extract known GEO/user KV fields
    auto get_kv = [&r](const std::string& key) -> const std::string& {
        static const std::string empty;
        auto it = r.user_kv.find(key);
        return (it != r.user_kv.end()) ? it->second : empty;
    };

    meta.gsm_id = get_kv("gsm_id");
    meta.gse_id = get_kv("gse_id");
    meta.organism = get_kv("organism");
    meta.protocol = get_kv("protocol");
    meta.modality = get_kv("modality");
    meta.geo_title = get_kv("geo_title");
    meta.geo_source_name = get_kv("geo_source_name");
    meta.singlet_version = get_kv("singlet_version");
    meta.pipeline_date = get_kv("pipeline_date");

    if (!get_kv("taxon_id").empty()) {
        try { meta.taxon_id = std::stoi(get_kv("taxon_id")); } catch (...) {}
    }
    if (!get_kv("read_count").empty()) {
        try { meta.read_count = std::stoll(get_kv("read_count")); } catch (...) {}
    }
    if (!get_kv("srr_ids").empty()) {
        std::istringstream ss(get_kv("srr_ids"));
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (!tok.empty()) meta.srr_ids.push_back(std::move(tok));
        }
    }

    // --- 9. Build result ---
    PzDeviceMatrix result;
    result.mat = std::move(d_mat);
    result.meta = std::move(meta);
    result.producer_stream = stream;
    result.pinned_indptr = std::move(pinned_indptr);
    result.pinned_indices = std::move(pinned_indices);
    result.pinned_values = std::move(pinned_values);

    // --- 10. Optionally expose host pointers via shared_ptr (SVD path) ---
    if (keep_host_pinned) {
        result.host_indptr = std::shared_ptr<int>(
            result.pinned_indptr.as<int>(),
            [](int*) noexcept {});  // no-op deleter
        result.host_indices = std::shared_ptr<int>(
            result.pinned_indices.as<int>(),
            [](int*) noexcept {});  // no-op deleter
        result.host_values = std::shared_ptr<float>(
            result.pinned_values.as<float>(),
            [](float*) noexcept {});  // no-op deleter
        result.host_retained = true;
    }

    // --- 11. Set convenience aliases ---
    result.n_genes = result.mat.rows;
    result.n_cells = result.mat.cols;

    return result;
}

}  // namespace io
}  // namespace singlet::gpu
