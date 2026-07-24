// SPDX-License-Identifier: MIT
//
// Standalone extraction of validate_1pz() and its dependencies.
// Reads only the header and footer of a .1pz file to extract metadata.

#ifndef VALIDATE_1PZ_H
#define VALIDATE_1PZ_H

#include <cstdint>
#include <string>

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
#pragma pack(pop)

// Validate and read header from a .1pz file.
// Reads only the header and footer (not the full payload).
// Throws std::runtime_error on any validation failure.
PZHeader validate_1pz(const std::string& path);

#endif // VALIDATE_1PZ_H
