// SPDX-License-Identifier: MIT
// gpu_data_loader.cu — async sparse GPU DataLoader delegating to Ring.
//
// Two-thread architecture (T_CL: chunk-loader, T_BB: batch-builder)
// with 2-deep chunk queue. All slot/buffer/event management delegated to Ring.
// Implements PLAN.md specification.

#include "gpu_data_loader.h"

#include <singlet/pileup/pz_reader.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <numeric>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <vector>

// ============================================================================
// CUDA error checking
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::ostringstream oss; \
        oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
            << cudaGetErrorString(err); \
        throw std::runtime_error(oss.str()); \
    } \
} while (0)

#define CUDA_CHECK_SILENT(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err)); \
    } \
} while (0)

// ============================================================================
// Log-normalization kernel (copied from original gpu_data_loader.cu)
// ============================================================================

static constexpr float kSparseLogNormScaler = 10000.0f;

__global__ void log_normalize_columns_kernel(int n_cols,
                                             float scaler,
                                             const int32_t* __restrict__ col_ptr,
                                             float* __restrict__ values) {
    int col = blockIdx.x;
    if (col >= n_cols) return;

    int start = col_ptr[col];
    int end   = col_ptr[col + 1];

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();

    // Phase 1: column sum via block-shared atomicAdd
    float thread_sum = 0.0f;
    for (int k = start + threadIdx.x; k < end; k += blockDim.x) {
        thread_sum += values[k];
    }
    if (thread_sum != 0.0f) atomicAdd(&s_sum, thread_sum);
    __syncthreads();

    float sum = s_sum;
    if (sum > 0.0f) {
        float inv = scaler / sum;
        for (int k = start + threadIdx.x; k < end; k += blockDim.x) {
            values[k] = log1pf(values[k] * inv);
        }
    }
}

// ============================================================================
// Saturate cast helpers (file-scope with atomic warning flags)
// ============================================================================

static std::atomic<bool> saturate_warned_idx{false};
static std::atomic<bool> saturate_warned_val{false};

static inline int32_t saturate_cast_u32_to_i32(uint32_t v) {
    if (v > static_cast<uint32_t>(INT32_MAX)) {
        bool exp = false;
        if (saturate_warned_idx.compare_exchange_strong(exp, true)) {
            std::fprintf(stderr, "[DataLoader] WARNING: uint32 index %u > INT32_MAX; truncating.\n", v);
        }
        return INT32_MAX;
    }
    return static_cast<int32_t>(v);
}

static inline float saturate_cast_u32_to_f32(uint32_t v) {
    if (v > (1u << 24)) {
        bool exp = false;
        if (saturate_warned_val.compare_exchange_strong(exp, true)) {
            std::fprintf(stderr, "[DataLoader] WARNING: uint32 value %u > 2^24; fp32 cast loses precision.\n", v);
        }
    }
    return static_cast<float>(v);
}

// ============================================================================
// Internal data structures
// ============================================================================

// Per-file decoded data (POINTER_LIST policy).
struct DecodedFile {
    std::vector<int32_t> col_ptr;   // length n_cols+1
    std::vector<int32_t> row_idx;   // length nnz
    std::vector<float> values;      // length nnz
    int n_cols;                      // number of columns in this file
    int m;                           // number of rows (features)
};

// Unified chunk data (concatenated all files into one CSC).
struct ChunkDataConcat {
    std::vector<int32_t> col_ptr;   // length total_cols+1
    std::vector<int32_t> row_idx;   // length total_nnz
    std::vector<float> values;      // length total_nnz
    int n_cols;                      // total columns across all files
    int m;                           // number of rows (features)
};

// Chunk data (union of both policies).
struct ChunkData {
    ConcatPolicy policy;
    ChunkDataConcat concat;  // used for CONCAT_HOST
    std::vector<DecodedFile> files;  // used for POINTER_LIST
    std::vector<std::pair<int, int>> column_permutation;  // (file_idx, local_col)
    int n_cols;
    int m;
};

// ============================================================================
// DataLoader::Impl — the pimpl struct holding all internal state
// ============================================================================

struct DataLoader::Impl {
    // Configuration
    std::vector<std::string> file_paths;
    int chunk_size = 0;
    int batch_size = 0;
    std::mt19937& rng;
    ConcatPolicy policy = ConcatPolicy::CONCAT_HOST;
    int omp_threads = 16;

    // External Ring and lane
    Ring* ring = nullptr;
    int lane_id_ = -1;

    // Explicit constructor to initialize the reference member
    explicit Impl(std::mt19937& r) : rng(r) {}

    // Dataset metadata
    int m = 0;  // number of features (peeked from first file)

    // CUDA stream (non-blocking, default priority)
    cudaStream_t loader_stream = nullptr;

    // Chunk queue (2-deep: max size 2 to overlap decode with training)
    std::mutex chunk_mtx;
    std::condition_variable chunk_cv;
    std::deque<std::unique_ptr<ChunkData>> chunk_queue;
    static constexpr size_t kChunkQueueMaxSize = 2;

    // Epoch state
    std::vector<std::string> epoch_order;
    size_t chunk_cursor = 0;  // which chunk we're loading next
    uint64_t chunk_sub_seed = 0;
    bool epoch_started = false;
    bool last_chunk_pushed = false;
    bool epoch_eof = false;

    // Worker threads
    std::thread t_cl;
    std::thread t_bb;
    std::atomic<bool> stop{false};

    // Destructor cleanup
    ~Impl() {
        stop = true;
        chunk_cv.notify_all();
        if (ring) ring->shutdown();  // wake batch_builder_thread if blocked in Ring::acquire_free()
        if (t_cl.joinable()) t_cl.join();
        if (t_bb.joinable()) t_bb.join();

        // Cleanup CUDA stream
        if (loader_stream) {
            CUDA_CHECK_SILENT(cudaStreamDestroy(loader_stream));
        }
    }
};

// ============================================================================
// Worker thread functions (T_CL and T_BB)
// ============================================================================

// T_CL: Chunk loader thread
// Decodes files in parallel (OpenMP) and pushes decoded chunks to queue (2-deep).
// When file list is exhausted, reshuffles and continues (auto-rolling).
static void chunk_loader_thread(DataLoader::Impl* impl) {
    try {
        while (!impl->stop && !impl->ring->is_shutdown()) {
            // Wait for epoch to start AND queue has space
            size_t chunk_start;
            size_t chunk_end;
            {
                std::unique_lock<std::mutex> lock(impl->chunk_mtx);
                impl->chunk_cv.wait(lock, [impl]() {
                    return (impl->epoch_started && impl->chunk_queue.size() < impl->kChunkQueueMaxSize) ||
                           impl->stop;
                });
                if (impl->stop || impl->ring->is_shutdown()) break;

                // Check if there are files to load
                chunk_start = impl->chunk_cursor;
                if (chunk_start >= impl->epoch_order.size()) {
                    // End of current pass: reshuffle, reset cursor, increment pass counter
                    std::shuffle(impl->epoch_order.begin(), impl->epoch_order.end(), impl->rng);
                    impl->chunk_sub_seed = impl->rng();
                    impl->chunk_cursor = 0;
                    impl->ring->increment_lane_pass(impl->lane_id_);
                    // Continue to load the first chunk of the new pass
                    chunk_start = 0;
                }

                chunk_end = std::min(chunk_start + (size_t)impl->chunk_size,
                                     impl->epoch_order.size());
            }

            // Load and decode files (OpenMP parallel)
            std::vector<singlet::pz::ReadResult> decoded_files;
            decoded_files.resize(chunk_end - chunk_start);
            bool read_error = false;
            std::string error_msg;

            #pragma omp parallel for schedule(dynamic) num_threads(impl->omp_threads)
            for (size_t i = chunk_start; i < chunk_end; ++i) {
                size_t idx = i - chunk_start;
                try {
                    decoded_files[idx] = singlet::pz::read_1pz(impl->epoch_order[i]);
                    if (decoded_files[idx].m != (uint32_t)impl->m) {
                        throw std::runtime_error("feature count mismatch");
                    }
                } catch (const std::exception& e) {
                    #pragma omp critical
                    {
                        if (!read_error) {
                            read_error = true;
                            error_msg = std::string(impl->epoch_order[i]) + ": " + e.what();
                        }
                    }
                }
            }

            if (read_error) {
                std::fprintf(stderr, "[DataLoader] FATAL: %s\n", error_msg.c_str());
                std::terminate();
            }

            // Build ChunkData based on policy
            auto chunk_data_ptr = std::make_unique<ChunkData>();
            chunk_data_ptr->policy = impl->policy;
            chunk_data_ptr->m = impl->m;

            if (impl->policy == ConcatPolicy::CONCAT_HOST) {
                // Phase 1: Compute offsets
                std::vector<int> nnz_off(chunk_end - chunk_start + 1, 0);
                std::vector<int> col_off(chunk_end - chunk_start + 1, 0);
                {
                    int total_nnz = 0;
                    int total_cols = 0;
                    for (size_t i = 0; i < decoded_files.size(); ++i) {
                        nnz_off[i] = total_nnz;
                        col_off[i] = total_cols;
                        total_nnz += decoded_files[i].nnz;
                        total_cols += decoded_files[i].n;
                    }
                    nnz_off[decoded_files.size()] = total_nnz;
                    col_off[decoded_files.size()] = total_cols;
                    chunk_data_ptr->concat.n_cols = total_cols;
                    chunk_data_ptr->n_cols = total_cols;
                }

                // Allocate unified buffers
                chunk_data_ptr->concat.col_ptr.resize(chunk_data_ptr->concat.n_cols + 1);
                chunk_data_ptr->concat.row_idx.resize(nnz_off[decoded_files.size()]);
                chunk_data_ptr->concat.values.resize(nnz_off[decoded_files.size()]);

                // Phase 2: Parallel copy+cast
                #pragma omp parallel for schedule(dynamic) num_threads(impl->omp_threads)
                for (size_t i = 0; i < decoded_files.size(); ++i) {
                    auto& r = decoded_files[i];
                    int col_base = col_off[i];
                    int nnz_base = nnz_off[i];

                    // Copy and cast col_ptr
                    for (uint32_t c = 0; c <= r.n; ++c) {
                        chunk_data_ptr->concat.col_ptr[col_base + c] =
                            saturate_cast_u32_to_i32(r.indptr[c]) + nnz_base;
                    }

                    // Copy and cast row_idx and values
                    for (uint64_t j = 0; j < r.nnz; ++j) {
                        chunk_data_ptr->concat.row_idx[nnz_base + j] =
                            saturate_cast_u32_to_i32(r.indices[j]);
                        chunk_data_ptr->concat.values[nnz_base + j] =
                            saturate_cast_u32_to_f32(r.data[j]);
                    }
                }

                // Fix col_ptr base (0-based for this chunk)
                int base = chunk_data_ptr->concat.col_ptr[0];
                for (int& c : chunk_data_ptr->concat.col_ptr) c -= base;

            } else {
                // POINTER_LIST: store per-file decoded arrays
                chunk_data_ptr->files.resize(decoded_files.size());

                #pragma omp parallel for schedule(dynamic) num_threads(impl->omp_threads)
                for (size_t i = 0; i < decoded_files.size(); ++i) {
                    auto& r = decoded_files[i];
                    auto& df = chunk_data_ptr->files[i];

                    df.n_cols = r.n;
                    df.m = r.m;
                    df.col_ptr.resize(r.n + 1);
                    df.row_idx.resize(r.nnz);
                    df.values.resize(r.nnz);

                    for (uint32_t c = 0; c <= r.n; ++c) {
                        df.col_ptr[c] = saturate_cast_u32_to_i32(r.indptr[c]);
                    }
                    for (uint64_t j = 0; j < r.nnz; ++j) {
                        df.row_idx[j] = saturate_cast_u32_to_i32(r.indices[j]);
                        df.values[j] = saturate_cast_u32_to_f32(r.data[j]);
                    }
                }

                // Compute total columns for permutation generation
                int total_cols = 0;
                for (auto& df : chunk_data_ptr->files) {
                    total_cols += df.n_cols;
                }
                chunk_data_ptr->n_cols = total_cols;
            }

            // Push to chunk queue
            std::cout << "[HANGDB] chunk_loader_thread: decoded and queuing chunk (start=" << chunk_start << ", end=" << chunk_end << ", n_cols=" << chunk_data_ptr->n_cols << ")" << std::endl;
            {
                std::unique_lock<std::mutex> lock(impl->chunk_mtx);
                impl->chunk_queue.push_back(std::move(chunk_data_ptr));
                impl->chunk_cursor = chunk_end;
                impl->chunk_cv.notify_all();
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[DataLoader T_CL] FATAL: %s\n", e.what());
        std::terminate();
    }
}

// T_BB: Batch builder thread
// Pops chunks from queue, builds batches, publishes to Ring.
// Marks chunk_end on the last batch of each decoded chunk.
static void batch_builder_thread(DataLoader::Impl* impl) {
    try {
        while (!impl->stop && !impl->ring->is_shutdown()) {
            std::cout << "[HANGDB] batch_builder_thread: getting next chunk" << std::endl;
            // Wait for a chunk from queue
            std::unique_ptr<ChunkData> chunk;
            {
                std::unique_lock<std::mutex> lock(impl->chunk_mtx);
                impl->chunk_cv.wait(lock, [impl]() {
                    return !impl->chunk_queue.empty() || impl->stop;
                });
                if (impl->stop || impl->ring->is_shutdown()) break;

                // If queue is empty, keep waiting
                if (impl->chunk_queue.empty()) {
                    continue;
                }

                chunk = std::move(impl->chunk_queue.front());
                impl->chunk_queue.pop_front();

                // Notify T_CL that queue has space now
                impl->chunk_cv.notify_all();
            }

            if (!chunk) continue;

            // Generate column permutation for this chunk
            std::mt19937 chunk_rng(impl->chunk_sub_seed);
            std::vector<int> col_perm;
            {
                if (impl->policy == ConcatPolicy::CONCAT_HOST) {
                    col_perm.resize(chunk->n_cols);
                    std::iota(col_perm.begin(), col_perm.end(), 0);
                    std::shuffle(col_perm.begin(), col_perm.end(), chunk_rng);
                } else {
                    // POINTER_LIST: generate (file_idx, local_col) pairs
                    for (size_t fi = 0; fi < chunk->files.size(); ++fi) {
                        for (int ci = 0; ci < chunk->files[fi].n_cols; ++ci) {
                            chunk->column_permutation.push_back({fi, ci});
                        }
                    }
                    std::shuffle(chunk->column_permutation.begin(),
                                 chunk->column_permutation.end(), chunk_rng);
                }
            }

            // Build batches from this chunk
            std::cout << "[HANGDB] batch_builder_thread: building batches from chunk (n_cols=" << chunk->n_cols << ")" << std::endl;
            int col_idx = 0;
            while (col_idx < chunk->n_cols) {
                // Check if this batch would be too small (drop tail)
                if (col_idx + impl->batch_size > chunk->n_cols) {
                    std::cout << "[HANGDB] batch_builder_thread: dropping tail batch (col_idx=" << col_idx << ", n_cols=" << chunk->n_cols << ")" << std::endl;
                    break;
                }

                // Acquire a FREE slot from Ring
                int slot_idx = impl->ring->acquire_free(impl->lane_id_);
                if (slot_idx < 0) {
                    // Ring is shutting down
                    return;
                }

                // Get slot view for filling
                Ring::SlotView slot_view = impl->ring->slot_view(impl->lane_id_, slot_idx);
                int B = impl->batch_size;

                // Compute nnz for this batch and gather column data
                int batch_nnz = 0;
                std::vector<int32_t> batch_col_ptr(B + 1);

                if (impl->policy == ConcatPolicy::CONCAT_HOST) {
                    batch_col_ptr[0] = 0;
                    for (int b = 0; b < B; ++b) {
                        int global_col = col_perm[col_idx + b];
                        int start = chunk->concat.col_ptr[global_col];
                        int end = chunk->concat.col_ptr[global_col + 1];
                        batch_nnz += (end - start);
                        batch_col_ptr[b + 1] = batch_nnz;
                    }

                    // Ensure capacity (grow if needed)
                    impl->ring->ensure_capacity(impl->lane_id_, slot_idx,
                                                B + 1, batch_nnz, batch_nnz);

                    // Refresh pointers after potential reallocation
                    slot_view = impl->ring->slot_view(impl->lane_id_, slot_idx);

                    // Copy column pointers
                    std::memcpy(slot_view.h_col_ptr, batch_col_ptr.data(), (B + 1) * sizeof(int32_t));

                    // Copy row indices and values
                    int offset = 0;
                    for (int b = 0; b < B; ++b) {
                        int global_col = col_perm[col_idx + b];
                        int start = chunk->concat.col_ptr[global_col];
                        int end = chunk->concat.col_ptr[global_col + 1];
                        int col_nnz = end - start;
                        std::memcpy(slot_view.h_row_idx + offset,
                                    chunk->concat.row_idx.data() + start,
                                    col_nnz * sizeof(int32_t));
                        std::memcpy(slot_view.h_values + offset,
                                    chunk->concat.values.data() + start,
                                    col_nnz * sizeof(float));
                        offset += col_nnz;
                    }

                } else {
                    // POINTER_LIST: first pass compute col_ptr and batch_nnz
                    batch_col_ptr[0] = 0;
                    for (int b = 0; b < B; ++b) {
                        auto [fi, ci] = chunk->column_permutation[col_idx + b];
                        auto& file = chunk->files[fi];
                        int start = file.col_ptr[ci];
                        int end = file.col_ptr[ci + 1];
                        int col_nnz = end - start;
                        batch_nnz += col_nnz;
                        batch_col_ptr[b + 1] = batch_nnz;
                    }

                    // Ensure capacity (grow if needed)
                    impl->ring->ensure_capacity(impl->lane_id_, slot_idx,
                                                B + 1, batch_nnz, batch_nnz);

                    // Refresh pointers after potential reallocation
                    slot_view = impl->ring->slot_view(impl->lane_id_, slot_idx);

                    // Copy column pointers
                    std::memcpy(slot_view.h_col_ptr, batch_col_ptr.data(), (B + 1) * sizeof(int32_t));

                    // Second pass: memcpy data
                    int offset = 0;
                    for (int b = 0; b < B; ++b) {
                        auto [fi, ci] = chunk->column_permutation[col_idx + b];
                        auto& file = chunk->files[fi];
                        int start = file.col_ptr[ci];
                        int end = file.col_ptr[ci + 1];
                        int col_nnz = end - start;

                        std::memcpy(slot_view.h_row_idx + offset, file.row_idx.data() + start,
                                    col_nnz * sizeof(int32_t));
                        std::memcpy(slot_view.h_values + offset, file.values.data() + start,
                                    col_nnz * sizeof(float));
                        offset += col_nnz;
                    }
                }

                col_idx += B;
                std::cout << "[HANGDB] batch_builder_thread: batch built (B=" << B << ", nnz=" << batch_nnz << ")" << std::endl;

                // Issue H2D transfers
                CUDA_CHECK(cudaMemcpyAsync(slot_view.d_col_ptr, slot_view.h_col_ptr,
                                          (B + 1) * sizeof(int32_t),
                                          cudaMemcpyHostToDevice, impl->loader_stream));
                CUDA_CHECK(cudaMemcpyAsync(slot_view.d_row_idx, slot_view.h_row_idx,
                                          batch_nnz * sizeof(int32_t),
                                          cudaMemcpyHostToDevice, impl->loader_stream));
                CUDA_CHECK(cudaMemcpyAsync(slot_view.d_values, slot_view.h_values,
                                          batch_nnz * sizeof(float),
                                          cudaMemcpyHostToDevice, impl->loader_stream));

                // Launch lognorm kernel
                log_normalize_csc_columns(B, kSparseLogNormScaler,
                                          slot_view.d_col_ptr, slot_view.d_values,
                                          impl->loader_stream);

                // Record event on loader stream
                std::cout << "[HANGDB] batch_builder_thread: recording ready_event for slot " << slot_idx << " (B=" << B << ", nnz=" << batch_nnz << ")" << std::endl;
                CUDA_CHECK(cudaEventRecord(slot_view.ready_event, impl->loader_stream));
                std::cout << "[HANGDB] batch_builder_thread: ready_event recorded for slot " << slot_idx << std::endl;

                // Set chunk_end = true if this is the last batch (or next batch would be dropped as tail)
                bool chunk_end = (col_idx >= chunk->n_cols) || (col_idx + impl->batch_size > chunk->n_cols);

                // Publish to Ring
                impl->ring->publish_ready(impl->lane_id_, slot_idx, B, batch_nnz, chunk_end);
            }
            std::cout << "[HANGDB] batch_builder_thread: finished processing chunk (n_cols=" << chunk->n_cols << ")" << std::endl;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[DataLoader T_BB] FATAL: %s\n", e.what());
        std::terminate();
    }
}

// ============================================================================
// Public API implementation
// ============================================================================

DataLoader::DataLoader(const std::vector<std::string>& paths,
                       int chunk_size,
                       int batch_size,
                       std::mt19937& rng,
                       Ring* ring,
                       int lane_id,
                       ConcatPolicy policy,
                       int omp_threads)
    : impl_(std::make_unique<Impl>(rng)) {
    impl_->file_paths = paths;
    impl_->chunk_size = chunk_size;
    impl_->batch_size = batch_size;
    impl_->policy = policy;
    impl_->ring = ring;
    impl_->lane_id_ = lane_id;
    impl_->omp_threads = (omp_threads > 0) ? omp_threads : 16;

    // Validate arguments
    if (paths.empty()) {
        throw std::runtime_error("DataLoader: no files provided");
    }
    if (chunk_size <= 0) {
        throw std::runtime_error("DataLoader: chunk_size must be > 0");
    }
    if (batch_size <= 0) {
        throw std::runtime_error("DataLoader: batch_size must be > 0");
    }
    if (ring == nullptr) {
        throw std::runtime_error("DataLoader: ring must not be null");
    }
    if (lane_id < 0 || lane_id >= ring->n_lanes()) {
        throw std::runtime_error("DataLoader: lane_id out of range");
    }

    // Peek m from first file
    try {
        singlet::pz::ReadResult r = singlet::pz::read_1pz(paths[0]);
        impl_->m = r.m;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[DataLoader] FATAL: %s: %s\n", paths[0].c_str(), e.what());
        std::terminate();
    }

    // Tell Ring the feature count
    ring->set_m(impl_->m);

    // Create non-blocking stream at default priority
    CUDA_CHECK(cudaStreamCreateWithFlags(&impl_->loader_stream, cudaStreamNonBlocking));

    // Spawn worker threads
    impl_->t_cl = std::thread(chunk_loader_thread, impl_.get());
    impl_->t_bb = std::thread(batch_builder_thread, impl_.get());
}

DataLoader::~DataLoader() {
    // Impl destructor will handle cleanup
}

void DataLoader::start() {
    std::cout << "[HANGDB] DataLoader::start() start" << std::endl;
    // Shuffle file paths once at startup
    impl_->epoch_order = impl_->file_paths;
    std::shuffle(impl_->epoch_order.begin(), impl_->epoch_order.end(), impl_->rng);

    // Draw chunk sub-seed for column permutation
    impl_->chunk_sub_seed = impl_->rng();

    // Initialize state
    {
        std::lock_guard<std::mutex> lock(impl_->chunk_mtx);
        impl_->chunk_cursor = 0;
        impl_->last_chunk_pushed = false;
        impl_->epoch_eof = false;
        impl_->chunk_queue.clear();
    }

    // Signal workers that they can start running
    {
        std::lock_guard<std::mutex> lock(impl_->chunk_mtx);
        impl_->epoch_started = true;
        impl_->chunk_cv.notify_all();
    }
    std::cout << "[HANGDB] DataLoader::start() complete" << std::endl;
}

int DataLoader::m() const {
    return impl_->m;
}

cudaStream_t DataLoader::loader_stream() const {
    return impl_->loader_stream;
}

int DataLoader::lane_id() const {
    return impl_->lane_id_;
}

// ---------------------------------------------------------------------------
// Standalone log-normalization helper (declared in gpu_data_loader.h).
// Wraps the same kernel launched inside batch_builder_thread.
// ---------------------------------------------------------------------------
void log_normalize_csc_columns(int n_cols,
                                float scaler,
                                const int32_t* d_col_ptr,
                                float* d_values,
                                cudaStream_t stream) {
    if (n_cols <= 0) return;
    log_normalize_columns_kernel<<<n_cols, 256, 0, stream>>>(
        n_cols, scaler, d_col_ptr, d_values);
}
