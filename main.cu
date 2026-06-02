// SPDX-License-Identifier: MIT
#include <singlet/pileup/pz_reader.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <numeric>

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#endif

#if __has_include(<thrust/reduce.h>) && __has_include(<thrust/execution_policy.h>) && __has_include(<thrust/device_ptr.h>) && __has_include(<thrust/device_vector.h>)
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#define HAS_THRUST 1
#endif


void process_file(const std::string& path) {
    try {
        auto result = singlet::pz::read_1pz(path);
        
#ifdef HAS_THRUST
        // Explicitly put the result on the GPU
        thrust::device_vector<uint32_t> d_data = result.data;
        auto* data_ptr = thrust::raw_pointer_cast(d_data.data());
#else
        auto* data_ptr = result.data.data();
#endif

        bool is_gpu = false;
#if __has_include(<cuda_runtime.h>) || defined(__CUDACC__)
        cudaPointerAttributes attr;
        if (cudaPointerGetAttributes(&attr, data_ptr) == cudaSuccess) {
#if CUDART_VERSION >= 10000
            if (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged) {
#else
            if (attr.memoryType == cudaMemoryTypeDevice) {
#endif
                is_gpu = true;
            }
        } else {
            cudaGetLastError(); // Clear error
        }
#endif

        double sum = 0.0;
        if (is_gpu) {
#ifdef HAS_THRUST
            auto d_ptr = thrust::device_pointer_cast(data_ptr);
            sum = thrust::reduce(thrust::device, d_ptr, d_ptr + d_data.size(), 0.0);
#else
            for (uint32_t val : result.data) {
                sum += val;
            }
#endif
        } else {
            // CPU fallback (use std::accumulate for clearer intent)
            sum = std::accumulate(result.data.begin(), result.data.end(), 0.0);
        }

        double mean = (result.m > 0 && result.n > 0) ? sum / (static_cast<double>(result.m) * result.n) : 0.0;
        
        std::cout << "File: " << path << (is_gpu ? " [GPU]" : " [CPU]") << "\n";
        std::cout << "Shape: " << result.m << " x " << result.n << "\n";
        std::cout << "Mean value: " << mean << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error processing " << path << ": " << e.what() << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path-to-.1pz> [additional-paths...]" << std::endl;
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        process_file(argv[i]);
    }

    return 0;
}
