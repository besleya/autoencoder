// SPDX-License-Identifier: MIT
//
// Standalone extraction of validate_1pz() and its dependencies
// (PZHeader, PZFooter, TP1_MAGIC_LOCAL) from the autoencoder repo's
// data.h / data.cpp, for use outside that repo's normal build.

#include "validate_1pz.h"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// .1pz constants and structs (mirror of singlet/pz_writer.h).
static constexpr uint32_t TP1_MAGIC_LOCAL   = 0x5A315054;  // "TP1Z"

#pragma pack(push, 1)
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
PZHeader validate_1pz(const std::string& path) {
    // Open the file in binary mode
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("validate_1pz: cannot open " + path);
    }

    // Seek to end and get file size
    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size < static_cast<std::streamsize>(sizeof(PZHeader) + sizeof(PZFooter))) {
        std::ostringstream oss;
        oss << "validate_1pz: file " << path << " too small (" << file_size
            << " bytes, need at least " << (sizeof(PZHeader) + sizeof(PZFooter))
            << " bytes)";
        throw std::runtime_error(oss.str());
    }

    // Read header
    PZHeader hdr;
    file.read(reinterpret_cast<char*>(&hdr), sizeof(PZHeader));
    if (!file.good()) {
        throw std::runtime_error("validate_1pz: failed to read header from " + path);
    }

    // Check magic
    if (hdr.magic != TP1_MAGIC_LOCAL) {
        std::ostringstream oss;
        oss << "validate_1pz: bad magic in " << path << " (got 0x" << std::hex
            << hdr.magic << ", expected 0x" << TP1_MAGIC_LOCAL << ")";
        throw std::runtime_error(oss.str());
    }

    // Check version
    if (hdr.version != 1 && hdr.version != 3 && hdr.version != 4) {
        std::ostringstream oss;
        oss << "validate_1pz: unsupported version in " << path << " (got "
            << hdr.version << ", expected 1, 3, or 4)";
        throw std::runtime_error(oss.str());
    }

    // Seek to footer and read it
    file.seekg(file_size - static_cast<std::streamsize>(sizeof(PZFooter)), std::ios::beg);
    PZFooter ftr;
    file.read(reinterpret_cast<char*>(&ftr), sizeof(PZFooter));
    if (!file.good()) {
        throw std::runtime_error("validate_1pz: failed to read footer from " + path);
    }

    // Check footer magic
    if (ftr.magic != TP1_MAGIC_LOCAL) {
        throw std::runtime_error("validate_1pz: bad footer magic in " + path);
    }

    // Check footer num_chunks matches header
    if (ftr.num_chunks != hdr.num_chunks) {
        std::ostringstream oss;
        oss << "validate_1pz: footer num_chunks (" << ftr.num_chunks
            << ") does not match header (" << hdr.num_chunks << ") in " << path;
        throw std::runtime_error(oss.str());
    }

    // Note: CRC32 is intentionally NOT validated because we read only the
    // header and footer (no slurp of the entire file).

    return hdr;
}
