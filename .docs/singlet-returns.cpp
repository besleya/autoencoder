// This file contains defs/headers for some singlet APIs
// Before each def is its file and namespace
// Most files are included with #include <singlet/singlet.h>

// ============================================================================
// ============================================================================


singlet/include/singlet/gpu/io/pz_device_loader.h
singlet::gpu::io::PzDeviceMatrix
// PzDeviceMatrix — result of load_pz().
// Owns the device buffers (via SparseMatrixGPU DeviceMemory) and the pinned
// host staging buffers (via PinnedBuffer). Metadata lives on host only.
//
// The caller must synchronize producer_stream before accessing mat on the
// device from a different stream.
//
// SVD path (cycle 5): when load_pz is called with keep_host_pinned=true, the
// pinned host CSC arrays are additionally exposed via shared_ptr fields below.
// This avoids re-staging for SVD adapters that take HOST CSC pointers
// (integration-notes.md finding 0a).
// Memory cost when host_retained=true: 2× the matrix (device + host pinned).
struct PzDeviceMatrix {
    core::DeviceCSC   mat;
    core::Metadata    meta;
    cudaStream_t      producer_stream = nullptr;

    // Pinned staging kept alive until the async copy completes.
    // Caller should sync the stream and then let this struct go out of scope,
    // or explicitly reset these buffers after sync.
    core::PinnedBuffer pinned_indptr;
    core::PinnedBuffer pinned_indices;
    core::PinnedBuffer pinned_values;

    // Optional retained host pointers for SVD adapters (keep_host_pinned=true).
    // Deleters call cudaFreeHost because the backing storage is pinned memory.
    // null when host_retained=false (default / non-SVD path).
    std::shared_ptr<int>   host_indptr;   // size: mat.cols + 1 (int32)
    std::shared_ptr<int>   host_indices;  // size: mat.nnz    (int32)
    std::shared_ptr<float> host_values;   // size: mat.nnz    (float)
    bool                   host_retained = false;

    // ── Compat alias fields (test-harness API) ────────────────────────────────
    int n_genes = 0;   // mat.rows (populated after load)
    int n_cells = 0;   // mat.cols (populated after load)
    void free(cudaStream_t /*s*/ = nullptr) {}  // RAII handles memory; no-op for compat
};


singlet/include/singlet/gpu/io/pz_device_loader.h
singlet::gpu::io::load_pz
// ---------------------------------------------------------------------------
// load_pz — one-shot full-matrix load.
//
// Reads the file on CPU (decompression), stages into pinned host memory, then
// fires three cudaMemcpyAsync calls to put the CSC on device.
//
// stream: if nullptr, a new high-priority stream is created and stored in
//         result.producer_stream. Caller is responsible for cudaStreamDestroy
//         if they passed nullptr (stream ownership is NOT transferred here —
//         the caller must check result.producer_stream and destroy it if it
//         was allocated by load_pz).
//
// WHY three separate async copies: allows the runtime to pipeline them over
// PCIe while the CPU continues other work. A single large copy would also
// work but prevents any overlap.
// ---------------------------------------------------------------------------
// keep_host_pinned: when true, the pinned host CSC buffers are retained in
// result.host_indptr / host_indices / host_values (shared_ptr with cudaFreeHost
// deleter) so that host-pointer SVD adapters can consume them directly without
// re-staging.  Default false — existing callers (lognorm, hvg) are unaffected.
inline PzDeviceMatrix load_pz(const std::string& path,
                               cudaStream_t stream = nullptr,
                               bool keep_host_pinned = false) {



// ============================================================================
// ============================================================================


singlet/include/singlet/pileup/pz_reader.h
// ============================================================================
// ReadResult — everything read_1pz() returns
// ============================================================================
struct ReadResult {
    uint32_t m = 0;                     // rows (features)
    uint32_t n = 0;                     // cols (cells)
    uint64_t nnz = 0;                   // total non-zeros
    uint8_t  vt_code = 0;               // value type code (1=uint8, 2=uint16, 3=uint32, etc.)

    // CSC layout: indptr has length n+1, indices/data have length nnz.
    // Values are widened to uint32_t regardless of vt_code to keep the API
    // uniform; callers that want the native narrow width can downcast.
    std::vector<uint32_t> indptr;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> data;

    std::vector<std::string> rownames;
    std::vector<std::string> colnames;
    std::map<std::string, std::string> user_kv;
};


singlet::pz::read_1pz
// At CPP-1, we parse the header+footer, validate CRC, decode the permutation
// and ptr blocks, and decode the metadata TLV. The VOCSC matrix decode itself
// is implemented in CPP-2. Calling read_1pz() in this file right now therefore
// leaves ``result.indptr`` / ``result.indices`` / ``result.data`` EMPTY (with
// only the ``indptr`` size set to ``n+1``). CPP-2 will fill those arrays.
// ============================================================================
inline ReadResult read_1pz(const std::string& path)



// ============================================================================
// Decode all VOCSC chunks, filling r.indices / r.data.
// ``body`` points at the start of the chunk table. ``perm`` and ``cc`` are
// the already-decoded permutation and column-count arrays.
//
// Chunk blob layout depends on feature_flags:
//
//   Without FEAT_BITPLANE_BITMAP (feature_flags & 0x02 == 0):
//     [ng:4][msz:4][zsz:4][crc:4][compressed:zsz]   — 16-byte header
//     The zstd payload decompresses directly to raw = meta(msz) + gaps(ng*gw).
//     CRC32 is over the decompressed raw bytes.
//
//   With FEAT_BITPLANE_BITMAP (feature_flags & 0x02 != 0):
//     [ng:4][msz:4][zsz:4][crc:4][packed_sz:4][compressed:zsz] — 20-byte header
//     The zstd payload decompresses to packed_sz bytes (bitmap-packed
//     pre-filter), which are then unpacked and un-bit-planed to produce raw.
//     CRC32 is over the compressed but pre-bitmap-packed bytes.
// ============================================================================
static inline void decode_vocsc_section(
    const uint8_t* body, size_t chunk_table_offset,
    const PZHeader& hdr,
    const std::vector<uint32_t>& perm,
    const std::vector<uint32_t>& cc,
    ReadResult& r)




// ============================================================================
// ============================================================================


singlet/include/singlet/gpu/core/types.h
singlet::gpu::core::DeviceCSC
// ---------------------------------------------------------------------------
// DeviceCSC — CSC sparse matrix on device (genes × cells, fp32).
//
// Layout: col_ptr[n+1] (int32), row_indices[nnz] (int32), values[nnz] (float).
// algorithm derived from factornet/gpu/types.cuh
// ---------------------------------------------------------------------------
struct DeviceCSC {
    DeviceMemory<int32_t> col_ptr;      // size: cols + 1
    DeviceMemory<int32_t> row_indices;  // size: nnz
    DeviceMemory<float>   values;       // size: nnz

    int rows = 0;
    int cols = 0;
    int nnz  = 0;

    DeviceCSC() = default;

    DeviceCSC(int m, int n, int nz)
        : col_ptr(n + 1)
        , row_indices(nz)
        , values(nz)
        , rows(m)
        , cols(n)
        , nnz(nz)
    {}

    // Non-owning factory — wraps externally-owned device pointers (e.g. from cupy).
    // The caller is responsible for the lifetime of the underlying device buffers.
    // Used by _bind_kernels.hpp to bridge Python-side cupy device arrays to GPU kernels.
    // algorithm derived from factornet/gpu/types.cuh
    static DeviceCSC from_device_ptrs(int m, int n, int nz,
                                      int32_t* d_col_ptr,
                                      int32_t* d_row_indices,
                                      float*   d_values) {
        DeviceCSC result;
        result.col_ptr      = DeviceMemory<int32_t>(d_col_ptr,    static_cast<std::size_t>(n + 1), false);
        result.row_indices  = DeviceMemory<int32_t>(d_row_indices, static_cast<std::size_t>(nz),    false);
        result.values       = DeviceMemory<float>  (d_values,      static_cast<std::size_t>(nz),    false);
        result.rows = m;
        result.cols = n;
        result.nnz  = nz;
        return result;
    }

    // Movable, not copyable.
    DeviceCSC(DeviceCSC&&)            = default;
    DeviceCSC& operator=(DeviceCSC&&) = default;
    DeviceCSC(const DeviceCSC&)            = delete;
    DeviceCSC& operator=(const DeviceCSC&) = delete;
};




// ============================================================================
// ============================================================================


singlet/include/singlet/gpu/core/memory.h
singlet::gpu::core::Metadata
// ---------------------------------------------------------------------------
// Metadata — GEO key-value fields embedded in every .1pz file plus optional
// row / column name vectors.
//
// Fields mirror the JSON schema from CLAUDE.md §GEO Metadata Embedding and
// style-rules.md §B. All strings are UTF-8. Fields not present in the .1pz
// metadata TLV remain empty / zero-initialized.
// ---------------------------------------------------------------------------
struct Metadata {
    // GEO identifiers
    std::string gsm_id;
    std::string gse_id;

    // Organism / taxonomy
    std::string organism;
    int32_t     taxon_id = 0;

    // Assay description
    std::string protocol;   // e.g. "10xv3", "Drop-seq"
    std::string modality;   // e.g. "scrna", "cite", "multiome"

    // Run accessions (serialized as comma-joined string from user_kv)
    std::vector<std::string> srr_ids;

    // Approximate read count from the catalog
    int64_t read_count = 0;

    // GEO textual fields
    std::string geo_title;
    std::string geo_source_name;

    // Pipeline provenance
    std::string singlet_version;
    std::string pipeline_date;

    // Row / column names from the .1pz metadata TLV (optional).
    // rows == feature names (genes), cols == barcode strings.
    std::vector<std::string> rownames;
    std::vector<std::string> colnames;
};


// ============================================================================
// ============================================================================


singlet/include/singlet/gpu/core/memory.h
singlet::gpu::core::PinnedPool
// ---------------------------------------------------------------------------
// PinnedPool — stateless factory. "Pool" in name only: we simply wrap
// cudaMallocHost. A real pool (suballocator with a freelist) can be swapped
// in later without changing the call sites.
// ---------------------------------------------------------------------------
struct PinnedPool {
    // Allocate bytes of page-locked host memory.
    // Throws std::runtime_error on CUDA allocation failure.
    static PinnedBuffer acquire(std::size_t bytes) {
        if (bytes == 0) return PinnedBuffer{};
        void* ptr = nullptr;
        cudaError_t err = cudaMallocHost(&ptr, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("PinnedPool::acquire cudaMallocHost failed: ")
                + cudaGetErrorString(err));
        }
        return PinnedBuffer{ptr, bytes};
    }
};
