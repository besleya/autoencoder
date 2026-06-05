// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Sparse>
#include <cstdint>
#include <string>
#include <vector>

using SpMatF = Eigen::SparseMatrix<float, Eigen::ColMajor>;

// All cells (columns) from the input files concatenated into one CSC matrix.
struct Dataset {
    uint32_t m = 0;     // features (rows); checked consistent across files
    SpMatF   X;         // (m, total_cells), each column = one cell
};

// Load and concatenate multiple .1pz files. File reads run in parallel via
// OpenMP; the resulting CSC arrays are then stitched together in a single
// bulk pass (no per-element inserts, no intermediate uint32 → int32 copy,
// no .eval() round-trip).
Dataset load_dataset(const std::vector<std::string>& paths);
