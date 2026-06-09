// SPDX-License-Identifier: MIT
#pragma once

#include <singlet/gpu/core/types.h>   // for core::DeviceCSC
#include <cstdint>
#include <string>
#include <vector>

// Concatenated CSC of all input files, owned on device.
// rows = features (genes), cols = total cells across all files, nnz = sum.
struct Dataset {
    int m = 0;                              // rows (features)
    int n = 0;                              // cols (total cells)
    int64_t nnz = 0;                        // total nonzeros
    singlet::gpu::core::DeviceCSC X;        // device CSC (move-only)
};

// Load multiple .1pz files and return a single concatenated DeviceCSC on the GPU.
// Throws std::runtime_error on any file failure or feature-count mismatch.
Dataset load_dataset(const std::vector<std::string>& paths);
