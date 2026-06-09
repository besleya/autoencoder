// SPDX-License-Identifier: MIT

#include "data.h"

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " file1.1pz [file2.1pz ...]"
                  << std::endl;
        return 1;
    }

    try {
        std::vector<std::string> paths;
        for (int i = 1; i < argc; ++i) {
            paths.push_back(argv[i]);
        }

        std::cout << "Loading dataset..." << std::endl;
        Dataset ds = load_dataset(paths);

        std::cout << "m (rows/features): " << ds.m << std::endl;
        std::cout << "n (cols/cells): " << ds.n << std::endl;
        std::cout << "nnz (nonzeros): " << ds.nnz << std::endl;

        // Copy first min(10, nnz) values to host and print
        int values_to_copy = (ds.nnz > 10) ? 10 : static_cast<int>(ds.nnz);
        std::vector<float> values_host(values_to_copy);
        std::vector<int> indices_host(values_to_copy);

        if (values_to_copy > 0) {
            cudaMemcpy(values_host.data(), ds.X.values.get(),
                       values_to_copy * sizeof(float),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(indices_host.data(), ds.X.row_indices.get(),
                       values_to_copy * sizeof(int),
                       cudaMemcpyDeviceToHost);

            std::cout << "\nFirst " << values_to_copy << " values:" << std::endl;
            for (int i = 0; i < values_to_copy; ++i) {
                std::cout << "  [" << i << "] value=" << values_host[i]
                          << ", row_idx=" << indices_host[i] << std::endl;
            }
        }

        // Copy first min(10, n+1) col_ptr entries to host and print
        int col_ptr_to_copy = (ds.n + 1 > 10) ? 10 : ds.n + 1;
        std::vector<int> col_ptr_host(col_ptr_to_copy);

        if (col_ptr_to_copy > 0) {
            cudaMemcpy(col_ptr_host.data(), ds.X.col_ptr.get(),
                       col_ptr_to_copy * sizeof(int),
                       cudaMemcpyDeviceToHost);

            std::cout << "\nFirst " << col_ptr_to_copy << " col_ptr entries:"
                      << std::endl;
            for (int i = 0; i < col_ptr_to_copy; ++i) {
                std::cout << "  col_ptr[" << i << "] = " << col_ptr_host[i]
                          << std::endl;
            }
        }

        std::cout << "\nSuccess!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
