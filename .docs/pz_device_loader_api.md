# Definitions in pz_device_loader.h

## Constants & Enums (in `pz_fmt` namespace)
- **TP1_MAGIC** — Magic number for .1pz file format ("TP1Z" in little-endian)
- **TP1_VERSION** — Format version (1)
- **FLAG_HAS_PERM, FLAG_GAP16, FLAG_HAS_METADATA, FLAG_HAS_COLSUMS** — Header flags for feature presence
- **FEAT_ZSTD_CHECKSUMS, FEAT_BITPLANE_BITMAP** — Feature flags
- **META_TAG_END, META_TAG_ROWNAMES, META_TAG_COLNAMES, META_TAG_USER_KV** — Metadata TLV tags
- **FP32_EXACT_INT_LIMIT** — Maximum integer representable exactly in float32 (2^24)

## Structs (in `pz_fmt` namespace)
- **PZHeader** — 96-byte packed header containing file metadata (magic, version, dimensions, chunk info)
- **PZFooter** — 16-byte packed footer containing CRC32 and chunk count

## Internal Helper Classes & Functions (anonymous namespace)
- **Crc32Loader** — Computes CRC32 checksums using slicing-by-8 algorithm
- **ZstdDCtxGuard** — RAII wrapper managing zstd decompression context lifecycle
- **zstd_decomp()** — Decompresses zstd-encoded data into a vector
- **varint_read()** — Decodes LEB128 variable-length integers
- **bitmap_unpack()** — Expands bitmap-packed data back to full representation
- **bit_planes_decode()** — Decodes bit-plane-encoded data (8 planes → bytes)
- **parse_metadata_tlv()** — Parses metadata from TLV byte stream into core::Metadata
- **saturate_cast_f32()** — Converts uint32 to float with precision-loss warning

## Public API Structs
- **PzDeviceMatrix** — Result struct containing device CSC matrix, metadata, stream, and pinned host buffers
- **PzLoadConfig** — Configuration options for load() with auto-tuning (stream, pinned memory, chunk size, etc.)
- **PzLoadResult** — Result struct containing device CSC, metadata map, row/col names, dimensions, and stream

## Public API Classes
- **PzChunkIterator** — Streaming iterator that yields fixed-width column slices as independent DeviceCSC objects

## Public API Functions
- **load_pz()** — One-shot full-matrix loader with optional host buffer retention for SVD adapters
- **load()** (explicit config) — New load API with Rule-31 auto-tuning and on-device uint→float conversion
- **load()** (no-args) — Convenience wrapper for load() with default auto-tune config
- **load_detail::decode_pz_raw()** — Decodes .1pz into pinned host CSC with raw (unconverted) value bytes

---

**Note:** This file also includes `singlet/gpu/io/detail/uint_to_float_kernel.h` for on-device type conversion. That file may contain relevant GPU kernel declarations if you need more context.
