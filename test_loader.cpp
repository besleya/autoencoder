// SPDX-License-Identifier: MIT
// test_loader.cpp — test load_pz_v3 implementation.
//
// Usage: ./test_loader [path]
// Default path: /mnt/home/besleya/quant/GSE211956/GSM6506117/gene_counts.1pz

#include "load_pz.h"
#include <singlet/pileup/pz_reader.h>
#include <singlet/gpu/core/types.h>

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <cstdlib>

// CUDA error check macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        std::exit(1); \
    } \
} while(0)

// Compute simple FNV-1a checksum for buffer
static uint64_t fnv1a(const uint8_t* data, size_t len) {
    uint64_t h = 0xCBF29CE484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= data[i];
        h *= 0x100000001B3ULL;
    }
    return h;
}

int main(int argc, char** argv) {
    try {
        std::string path = "/mnt/home/besleya/quant/GSE211956/GSM6506117/gene_counts.1pz";
        if (argc > 1) {
            path = argv[1];
        }

        std::cout << "Loading " << path << " with load_pz_v3..." << std::endl;

        // --- 1. Load to GPU ---
        singlet::gpu::io::PzDeviceMatrix dev_matrix = singlet::gpu::io::load_pz_v3(path);

        std::cout << "  Device matrix: " << dev_matrix.mat.rows << " x "
                  << dev_matrix.mat.cols << " (" << dev_matrix.mat.nnz << " nnz)"
                  << std::endl;

        // --- 2. Synchronize stream ---
        if (dev_matrix.producer_stream != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(dev_matrix.producer_stream));
        }

        // --- 3. Copy device data back to host ---
        std::vector<int32_t> h_indptr_dev(dev_matrix.mat.cols + 1);
        std::vector<int32_t> h_indices_dev(dev_matrix.mat.nnz);
        std::vector<float> h_values_dev(dev_matrix.mat.nnz);

        CUDA_CHECK(cudaMemcpy(
            h_indptr_dev.data(),
            dev_matrix.mat.col_ptr.get(),
            h_indptr_dev.size() * sizeof(int32_t),
            cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(
            h_indices_dev.data(),
            dev_matrix.mat.row_indices.get(),
            h_indices_dev.size() * sizeof(int32_t),
            cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(
            h_values_dev.data(),
            dev_matrix.mat.values.get(),
            h_values_dev.size() * sizeof(float),
            cudaMemcpyDeviceToHost));

        // --- 4. Get ground truth from CPU decoder ---
        std::cout << "Reading ground truth with read_1pz..." << std::endl;
        singlet::pz::ReadResult cpu_result = singlet::pz::read_1pz(path);

        std::cout << "  CPU matrix: " << cpu_result.m << " x "
                  << cpu_result.n << " (" << cpu_result.nnz << " nnz)"
                  << std::endl;

        // --- 5. Validate dimensions ---
        bool pass = true;
        if (dev_matrix.mat.rows != static_cast<int>(cpu_result.m)) {
            std::cerr << "ERROR: rows mismatch: device=" << dev_matrix.mat.rows
                      << ", cpu=" << cpu_result.m << std::endl;
            pass = false;
        }
        if (dev_matrix.mat.cols != static_cast<int>(cpu_result.n)) {
            std::cerr << "ERROR: cols mismatch: device=" << dev_matrix.mat.cols
                      << ", cpu=" << cpu_result.n << std::endl;
            pass = false;
        }
        if (dev_matrix.mat.nnz != static_cast<int>(cpu_result.nnz)) {
            std::cerr << "ERROR: nnz mismatch: device=" << dev_matrix.mat.nnz
                      << ", cpu=" << cpu_result.nnz << std::endl;
            pass = false;
        }

        // --- 6. Validate indptr (CSC column pointers) ---
        int indptr_mismatch_count = 0;
        for (uint32_t j = 0; j <= cpu_result.n; ++j) {
            int32_t exp = static_cast<int32_t>(cpu_result.indptr[j]);
            if (h_indptr_dev[j] != exp && indptr_mismatch_count < 5) {
                std::cerr << "  indptr[" << j << "]: got " << h_indptr_dev[j]
                          << ", expected " << exp << std::endl;
                indptr_mismatch_count++;
                pass = false;
            }
        }
        if (indptr_mismatch_count > 0) {
            if (indptr_mismatch_count >= 5)
                std::cerr << "  ... (more indptr mismatches)" << std::endl;
        } else {
            std::cout << "✓ indptr: all " << (cpu_result.n + 1) << " entries match" << std::endl;
        }

        // --- 7. Validate indices ---
        int indices_mismatch_count = 0;
        for (uint64_t i = 0; i < cpu_result.nnz; ++i) {
            int32_t exp = static_cast<int32_t>(cpu_result.indices[i]);
            if (h_indices_dev[i] != exp && indices_mismatch_count < 5) {
                std::cerr << "  indices[" << i << "]: got " << h_indices_dev[i]
                          << ", expected " << exp << std::endl;
                indices_mismatch_count++;
                pass = false;
            }
        }
        if (indices_mismatch_count > 0) {
            if (indices_mismatch_count >= 5)
                std::cerr << "  ... (more indices mismatches)" << std::endl;
        } else {
            std::cout << "✓ indices: all " << cpu_result.nnz << " entries match" << std::endl;
        }

        // --- 8. Validate values (with tolerance for float conversion) ---
        int values_mismatch_count = 0;
        for (uint64_t i = 0; i < cpu_result.nnz; ++i) {
            float exp = static_cast<float>(cpu_result.data[i]);
            // Allow exact equality since uint32->float cast is deterministic
            if (h_values_dev[i] != exp && values_mismatch_count < 5) {
                std::cerr << "  values[" << i << "]: got " << std::fixed << std::setprecision(6)
                          << h_values_dev[i] << ", expected " << exp << std::endl;
                values_mismatch_count++;
                pass = false;
            }
        }
        if (values_mismatch_count > 0) {
            if (values_mismatch_count >= 5)
                std::cerr << "  ... (more values mismatches)" << std::endl;
        } else {
            std::cout << "✓ values: all " << cpu_result.nnz << " entries match" << std::endl;
        }

        // --- 9. Compute checksums for quick sanity check ---
        uint64_t indptr_checksum_dev = fnv1a(
            reinterpret_cast<const uint8_t*>(h_indptr_dev.data()),
            h_indptr_dev.size() * sizeof(int32_t));
        uint64_t indptr_checksum_cpu = fnv1a(
            reinterpret_cast<const uint8_t*>(cpu_result.indptr.data()),
            cpu_result.indptr.size() * sizeof(uint32_t));

        uint64_t indices_checksum_dev = fnv1a(
            reinterpret_cast<const uint8_t*>(h_indices_dev.data()),
            h_indices_dev.size() * sizeof(int32_t));
        uint64_t indices_checksum_cpu = fnv1a(
            reinterpret_cast<const uint8_t*>(cpu_result.indices.data()),
            cpu_result.indices.size() * sizeof(uint32_t));

        uint64_t values_checksum_dev = fnv1a(
            reinterpret_cast<const uint8_t*>(h_values_dev.data()),
            h_values_dev.size() * sizeof(float));
        uint64_t values_checksum_cpu = fnv1a(
            reinterpret_cast<const uint8_t*>(cpu_result.data.data()),
            cpu_result.data.size() * sizeof(uint32_t));

        std::cout << "Checksums (FNV-1a):" << std::endl;
        std::cout << "  indptr:  device=" << std::hex << indptr_checksum_dev
                  << ", cpu=" << indptr_checksum_cpu << std::dec << std::endl;
        std::cout << "  indices: device=" << std::hex << indices_checksum_dev
                  << ", cpu=" << indices_checksum_cpu << std::dec << std::endl;
        std::cout << "  values:  device=" << std::hex << values_checksum_dev
                  << ", cpu=" << values_checksum_cpu << std::dec << std::endl;

        // --- 10. Report result ---
        if (pass) {
            std::cout << "\n✓ PASS: All validations successful" << std::endl;
            return 0;
        } else {
            std::cout << "\n✗ FAIL: Validation errors detected" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
