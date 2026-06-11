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

// .1pz constants and structs (mirror of singlet/pz_writer.h).
// Local copies kept in this header so callers do not have to pull in singlet.
static constexpr uint32_t TP1_MAGIC_LOCAL   = 0x5A315054;  // "TP1Z"

// Copied verbatim from pz_writer.h
#pragma pack(push, 1)
struct PZHeader {
    uint32_t magic;            // 0
    uint16_t version;          // 4
    uint8_t  vt_code;          // 6
    uint8_t  flags;            // 7
    uint32_t m;                // 8
    uint32_t n;                // 12
    uint64_t nnz;              // 16
    uint8_t  ptr_width;        // 24
    uint8_t  codec_level;      // 25
    uint16_t _pad0;            // 26
    uint32_t num_chunks;       // 28
    uint32_t perm_z_sz;        // 32
    uint32_t ptr_z_sz;         // 36
    uint32_t chunk_cols;       // 40
    uint32_t feature_flags;    // 44
    uint64_t metadata_offset;  // 48
    uint32_t metadata_z_sz;    // 56
    uint32_t colsums_z_sz;     // 60
    uint64_t transpose_offset; // 64
    uint32_t transpose_z_sz;   // 72
    uint32_t transpose_chunks; // 76
    uint8_t  reserved[16];     // 80-95
};
static_assert(sizeof(PZHeader) == 96, "PZHeader must be 96 bytes");

struct PZFooter {
    uint32_t file_crc32;
    uint32_t _reserved;
    uint32_t num_chunks;
    uint32_t magic;
};
static_assert(sizeof(PZFooter) == 16, "PZFooter must be 16 bytes");
#pragma pack(pop)


// Validate and read header from a .1pz file.
// Throws std::runtime_error on any validation failure.
PZHeader validate_1pz(const std::string& path);
