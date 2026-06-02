// SPDX-License-Identifier: MIT
#include <singlet/singlet.h>

#include <Eigen/Sparse>

#include <iostream>
#include <stdexcept>
#include <string>
#include <numeric>

using PZMatrix = Eigen::SparseMatrix<uint32_t, Eigen::ColMajor>;

void process_file(const std::string& path) {
    try {
        auto result = singlet::pz::read_1pz(path);
        std::cout << "\nFile: " << path << '\n';
        std::cout << "Shape: " << result.m << " x " << result.n << " with " << result.nnz << " non-zero elements\n";
        std::cout << "NNZ is " << static_cast<double>(result.nnz) / INT32_MAX << " small enough\n";
        
        std::vector<int32_t> outer(result.indptr.begin(),
                                result.indptr.end());

        std::vector<int32_t> inner(result.indices.begin(),
                                result.indices.end());
        Eigen::Map<PZMatrix> map(result.m, result.n, result.nnz, outer.data(), inner.data(), result.data.data());
        PZMatrix mat = map.eval(); // This becomes a matrix with cells in the columns and genes in the rows

        int64_t sum = mat.sum();

        double mean = (result.m > 0 && result.n > 0) ? sum / (static_cast<double>(result.m) * result.n) : 0.0;
        
        std::cout << "Mean value: " << mean << '\n';
        std::cout << "Shape: " << mat.rows() << " x " << mat.cols() << '\n';
        std::cout << "mat[0,6] = " << mat.coeffRef(6, 0) << '\n';
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
