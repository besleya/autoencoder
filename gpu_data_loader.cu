// SPDX-License-Identifier: MIT
// gpu_data_loader.cu — implementation of async sparse GPU DataLoader.
//
// Two-thread architecture (T_CL: chunk-loader, T_BB: batch-builder)
// with ring buffer (depth K, default 4) and chunk queue (depth 1).
// Implements DESIGN.md specification sections 2–10 and 13.

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

// Ring buffer slot state machine: FREE → BUILDING → READY → CONSUMED → FREE
enum class SlotState { FREE, BUILDING, READY, CONSUMED };

// A single ring buffer slot.
struct RingSlot {
    // Pinned host buffers
    singlet::gpu::core::PinnedBuffer h_col_ptr_buf;
    singlet::gpu::core::PinnedBuffer h_row_idx_buf;
    singlet::gpu::core::PinnedBuffer h_values_buf;

    int32_t* h_col_ptr = nullptr;
    int32_t* h_row_idx = nullptr;
    float*   h_values  = nullptr;
    size_t   h_col_capacity = 0;   // in int32_t (for col_ptr[B+1])
    size_t   h_row_capacity = 0;   // in int32_t
    size_t   h_val_capacity = 0;   // in float

    // Device buffers
    singlet::gpu::core::DeviceMemory<int32_t> d_col_ptr;
    singlet::gpu::core::DeviceMemory<int32_t> d_row_idx;
    singlet::gpu::core::DeviceMemory<float>   d_values;

    // Event (created at construction, reused)
    cudaEvent_t ready_event = nullptr;

    // Current batch metadata
    int B = 0;           // batch size (columns)
    int nnz = 0;         // nonzeros
    bool eof_after = false;

    SlotState state = SlotState::FREE;
};

// ============================================================================
// DataLoader::Impl — the pimpl struct holding all internal state
// ============================================================================

struct DataLoader::Impl {
    // Configuration
    std::vector<std::string> file_paths;
    int chunk_size;
    int batch_size;
    std::mt19937& rng;
    ConcatPolicy policy;
    int ring_depth;
    int n_concurrent_loaders;
    int omp_threads;

    // Dataset metadata
    int m;  // number of features (peeked from first file)

    // CUDA stream (non-blocking, default priority)
    cudaStream_t loader_stream = nullptr;

    // Ring buffer
    std::vector<RingSlot> ring;
    std::mutex ring_mtx;
    std::condition_variable ring_cv;
    int free_count;      // number of FREE slots
    int ready_count;     // number of READY slots
    std::deque<int> consumed_slots;  // recently consumed slot indices (for delayed recycling)

    // Chunk queue (depth 1)
    std::mutex chunk_mtx;
    std::condition_variable chunk_cv;
    std::mutex chunk_consumed_mtx;
    std::condition_variable chunk_consumed_cv;
    std::unique_ptr<ChunkData> chunk_data;
    bool chunk_full = false;      // true if chunk_data is populated
    bool chunk_consumed = false;  // signaled by T_BB after last column of chunk

    // Epoch state
    std::vector<std::string> epoch_order;
    size_t chunk_cursor = 0;  // which chunk we're loading next
    uint64_t chunk_sub_seed;
    bool epoch_started = false;
    bool last_chunk_flag = false;
    bool epoch_eof = false;  // set by T_BB after last batch of last chunk

    // Worker threads
    std::thread t_cl;
    std::thread t_bb;
    std::atomic<bool> stop{false};

    // Destructor cleanup
    ~Impl() {
        stop = true;
        chunk_cv.notify_all();
        ring_cv.notify_all();
        chunk_consumed_cv.notify_all();
        if (t_cl.joinable()) t_cl.join();
        if (t_bb.joinable()) t_bb.join();

        // Cleanup CUDA stream
        if (loader_stream) {
            CUDA_CHECK_SILENT(cudaStreamDestroy(loader_stream));
        }

        // Cleanup ring slots
        for (auto& slot : ring) {
            if (slot.ready_event) {
                CUDA_CHECK_SILENT(cudaEventDestroy(slot.ready_event));
            }
        }
    }
};

// ============================================================================
// Worker thread functions (T_CL and T_BB)
// ============================================================================

// T_CL: Chunk loader thread
static void chunk_loader_thread(DataLoader::Impl* impl) {
    try {
        while (!impl->stop) {
            // Wait for epoch to start or stop signal
            size_t chunk_start;
            size_t chunk_end;
            bool is_last_chunk;
            {
                std::unique_lock<std::mutex> lock(impl->chunk_mtx);
                impl->chunk_cv.wait(lock, [impl]() {
                    return (impl->epoch_started && !impl->chunk_full) || impl->last_chunk_flag || impl->stop;
                });
                if (impl->stop) break;
                if (impl->last_chunk_flag && impl->chunk_full) {
                    // Wait for chunk to be consumed
                    continue;
                }

                // Determine which chunk to load
                chunk_start = impl->chunk_cursor;
                if (chunk_start >= impl->epoch_order.size()) {
                    impl->last_chunk_flag = true;
                    continue;
                }

                chunk_end = std::min(chunk_start + (size_t)impl->chunk_size, 
                                     impl->epoch_order.size());
                is_last_chunk = (chunk_end >= impl->epoch_order.size());
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

            // Push to chunk queue and wait for T_BB to consume
            {
                std::unique_lock<std::mutex> lock(impl->chunk_mtx);
                impl->chunk_data = std::move(chunk_data_ptr);
                impl->chunk_full = true;
                if (is_last_chunk) {
                    impl->last_chunk_flag = true;
                }
                impl->chunk_cursor = chunk_end;
                impl->chunk_cv.notify_all();
            }

            // Wait for T_BB to signal chunk consumed (after last column copied)
            {
                std::unique_lock<std::mutex> lock(impl->chunk_consumed_mtx);
                impl->chunk_consumed_cv.wait(lock, [impl]() {
                    return impl->chunk_consumed || impl->stop;
                });
                impl->chunk_consumed = false;
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[DataLoader T_CL] FATAL: %s\n", e.what());
        std::terminate();
    }
}

// T_BB: Batch builder thread
static void batch_builder_thread(DataLoader::Impl* impl) {
    try {
        while (!impl->stop) {
            // Wait for chunk to be ready
            std::unique_ptr<ChunkData> chunk;
            {
                std::unique_lock<std::mutex> lock(impl->chunk_mtx);
                impl->chunk_cv.wait(lock, [impl]() {
                    return impl->chunk_full || impl->stop;
                });
                if (impl->stop) break;
                if (!impl->chunk_data) continue;

                chunk = std::move(impl->chunk_data);
                impl->chunk_full = false;
                impl->chunk_cv.notify_all();
            }

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
            int col_idx = 0;
            bool is_last_chunk = impl->last_chunk_flag;

            while (col_idx < chunk->n_cols) {
                // Check if this batch would be too small (drop tail)
                if (col_idx + impl->batch_size > chunk->n_cols) {
                    break;
                }

                // Acquire a FREE slot
                int slot_idx;
                {
                    std::unique_lock<std::mutex> lock(impl->ring_mtx);
                    impl->ring_cv.wait(lock, [impl]() {
                        return impl->free_count > 0 || impl->stop;
                    });
                    if (impl->stop) break;

                    // Find a FREE slot
                    slot_idx = -1;
                    for (int i = 0; i < impl->ring_depth; ++i) {
                        if (impl->ring[i].state == SlotState::FREE) {
                            slot_idx = i;
                            break;
                        }
                    }
                    if (slot_idx < 0) continue;

                    impl->ring[slot_idx].state = SlotState::BUILDING;
                    --impl->free_count;
                }

                auto& slot = impl->ring[slot_idx];
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

                    // Grow pinned + device buffers if needed
                    if (batch_col_ptr[B] > (int)slot.h_row_capacity) {
                        slot.h_col_ptr_buf = singlet::gpu::core::PinnedPool::acquire(
                            (B + 1) * sizeof(int32_t));
                        slot.h_row_idx_buf = singlet::gpu::core::PinnedPool::acquire(
                            batch_col_ptr[B] * sizeof(int32_t));
                        slot.h_values_buf = singlet::gpu::core::PinnedPool::acquire(
                            batch_col_ptr[B] * sizeof(float));

                        slot.h_col_ptr = slot.h_col_ptr_buf.as<int32_t>();
                        slot.h_row_idx = slot.h_row_idx_buf.as<int32_t>();
                        slot.h_values = slot.h_values_buf.as<float>();

                        slot.h_col_capacity = B + 1;
                        slot.h_row_capacity = batch_col_ptr[B];
                        slot.h_val_capacity = batch_col_ptr[B];

                        slot.d_col_ptr = singlet::gpu::core::DeviceMemory<int32_t>(B + 1);
                        slot.d_row_idx = singlet::gpu::core::DeviceMemory<int32_t>(batch_col_ptr[B]);
                        slot.d_values = singlet::gpu::core::DeviceMemory<float>(batch_col_ptr[B]);
                    }

                    // Sequential copy of column data
                    std::memcpy(slot.h_col_ptr, batch_col_ptr.data(), (B + 1) * sizeof(int32_t));
                    int offset = 0;
                    for (int b = 0; b < B; ++b) {
                        int global_col = col_perm[col_idx + b];
                        int start = chunk->concat.col_ptr[global_col];
                        int end = chunk->concat.col_ptr[global_col + 1];
                        int col_nnz = end - start;
                        std::memcpy(slot.h_row_idx + offset,
                                    chunk->concat.row_idx.data() + start,
                                    col_nnz * sizeof(int32_t));
                        std::memcpy(slot.h_values + offset,
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

                    // Check and allocate buffers once (before second pass)
                    if (batch_col_ptr[B] > (int)slot.h_row_capacity) {
                        slot.h_col_ptr_buf = singlet::gpu::core::PinnedPool::acquire(
                            (B + 1) * sizeof(int32_t));
                        slot.h_row_idx_buf = singlet::gpu::core::PinnedPool::acquire(
                            batch_col_ptr[B] * sizeof(int32_t));
                        slot.h_values_buf = singlet::gpu::core::PinnedPool::acquire(
                            batch_col_ptr[B] * sizeof(float));

                        slot.h_col_ptr = slot.h_col_ptr_buf.as<int32_t>();
                        slot.h_row_idx = slot.h_row_idx_buf.as<int32_t>();
                        slot.h_values = slot.h_values_buf.as<float>();

                        slot.h_col_capacity = B + 1;
                        slot.h_row_capacity = batch_col_ptr[B];
                        slot.h_val_capacity = batch_col_ptr[B];

                        slot.d_col_ptr = singlet::gpu::core::DeviceMemory<int32_t>(B + 1);
                        slot.d_row_idx = singlet::gpu::core::DeviceMemory<int32_t>(batch_col_ptr[B]);
                        slot.d_values = singlet::gpu::core::DeviceMemory<float>(batch_col_ptr[B]);
                    }

                    // Second pass: memcpy data
                    std::memcpy(slot.h_col_ptr, batch_col_ptr.data(), (B + 1) * sizeof(int32_t));
                    int offset = 0;
                    for (int b = 0; b < B; ++b) {
                        auto [fi, ci] = chunk->column_permutation[col_idx + b];
                        auto& file = chunk->files[fi];
                        int start = file.col_ptr[ci];
                        int end = file.col_ptr[ci + 1];
                        int col_nnz = end - start;

                        std::memcpy(slot.h_row_idx + offset, file.row_idx.data() + start,
                                    col_nnz * sizeof(int32_t));
                        std::memcpy(slot.h_values + offset, file.values.data() + start,
                                    col_nnz * sizeof(float));
                        offset += col_nnz;
                    }
                }

                // Signal T_CL if this is the last batch of the chunk
                col_idx += B;
                if (col_idx >= chunk->n_cols) {
                    std::lock_guard<std::mutex> lock(impl->chunk_consumed_mtx);
                    impl->chunk_consumed = true;
                    impl->chunk_consumed_cv.notify_all();
                }

                // Issue H2D transfers
                CUDA_CHECK(cudaMemcpyAsync(slot.d_col_ptr.get(), slot.h_col_ptr,
                                          (B + 1) * sizeof(int32_t),
                                          cudaMemcpyHostToDevice, impl->loader_stream));
                CUDA_CHECK(cudaMemcpyAsync(slot.d_row_idx.get(), slot.h_row_idx,
                                          batch_nnz * sizeof(int32_t),
                                          cudaMemcpyHostToDevice, impl->loader_stream));
                CUDA_CHECK(cudaMemcpyAsync(slot.d_values.get(), slot.h_values,
                                          batch_nnz * sizeof(float),
                                          cudaMemcpyHostToDevice, impl->loader_stream));

                // Launch lognorm kernel
                log_normalize_columns_kernel<<<B, 256, 0, impl->loader_stream>>>(
                    B, kSparseLogNormScaler, slot.d_col_ptr.get(), slot.d_values.get());

                // Record event
                CUDA_CHECK(cudaEventRecord(slot.ready_event, impl->loader_stream));

                // Mark slot READY
                {
                    std::lock_guard<std::mutex> lock(impl->ring_mtx);
                    slot.B = B;
                    slot.nnz = batch_nnz;
                    slot.eof_after = (col_idx >= chunk->n_cols) && is_last_chunk;
                    slot.state = SlotState::READY;
                    ++impl->ready_count;
                    impl->ring_cv.notify_all();
                }
            }

            // Signal EOF if this was the last chunk
            if (is_last_chunk) {
                std::lock_guard<std::mutex> lock(impl->ring_mtx);
                impl->epoch_eof = true;
                impl->ring_cv.notify_all();
            }
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
                       int chunk_size, int batch_size,
                       std::mt19937& rng,
                       ConcatPolicy policy,
                       int ring_depth,
                       int n_concurrent_loaders)
    : impl_(std::make_unique<Impl>()) {
    impl_->file_paths = paths;
    impl_->chunk_size = chunk_size;
    impl_->batch_size = batch_size;
    impl_->rng = rng;
    impl_->policy = policy;
    impl_->ring_depth = ring_depth;
    impl_->n_concurrent_loaders = n_concurrent_loaders;

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
    if (ring_depth <= 0) {
        throw std::runtime_error("DataLoader: ring_depth must be > 0");
    }
    if (n_concurrent_loaders <= 0) {
        throw std::runtime_error("DataLoader: n_concurrent_loaders must be > 0");
    }

    // Compute OMP thread budget
    impl_->omp_threads = std::max(1, omp_get_max_threads() / n_concurrent_loaders);

    // Peek m from first file
    try {
        singlet::pz::ReadResult r = singlet::pz::read_1pz(paths[0]);
        impl_->m = r.m;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[DataLoader] FATAL: %s: %s\n", paths[0].c_str(), e.what());
        std::terminate();
    }

    // Create non-blocking stream at default priority
    CUDA_CHECK(cudaStreamCreateWithFlags(&impl_->loader_stream, cudaStreamNonBlocking));

    // Initialize ring buffer
    impl_->ring.resize(ring_depth);
    impl_->free_count = ring_depth;
    impl_->ready_count = 0;

    for (int i = 0; i < ring_depth; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&impl_->ring[i].ready_event,
                                            cudaEventDisableTiming));
    }

    // Spawn worker threads
    impl_->t_cl = std::thread(chunk_loader_thread, impl_.get());
    impl_->t_bb = std::thread(batch_builder_thread, impl_.get());
}

DataLoader::~DataLoader() {
    // Impl destructor will handle cleanup
}

void DataLoader::begin_epoch() {
    // Shuffle file paths
    impl_->epoch_order = impl_->file_paths;
    std::shuffle(impl_->epoch_order.begin(), impl_->epoch_order.end(), impl_->rng);

    // Draw chunk sub-seed for column permutation
    impl_->chunk_sub_seed = impl_->rng();

    // Reset state
    {
        std::lock_guard<std::mutex> lock(impl_->ring_mtx);
        impl_->free_count = impl_->ring_depth;
        impl_->ready_count = 0;
        for (auto& slot : impl_->ring) {
            slot.state = SlotState::FREE;
        }
        impl_->consumed_slots.clear();
        impl_->epoch_eof = false;
    }

    // Reset chunk state
    {
        std::lock_guard<std::mutex> lock(impl_->chunk_mtx);
        impl_->chunk_cursor = 0;
        impl_->last_chunk_flag = false;
    }

    // Signal T_CL to start
    {
        std::lock_guard<std::mutex> lock(impl_->chunk_mtx);
        impl_->epoch_started = true;
        impl_->chunk_cv.notify_all();
    }
}

bool DataLoader::next_batch(SparseBatch* out) {
    std::unique_lock<std::mutex> lock(impl_->ring_mtx);

    // Wait for a READY slot or EOF
    impl_->ring_cv.wait(lock, [this]() {
        return impl_->ready_count > 0 || impl_->epoch_eof;
    });

    // Check if we have a ready batch
    if (impl_->ready_count == 0) {
        // Epoch exhausted
        return false;
    }

    // Find the first READY slot (FIFO order)
    int slot_idx = -1;
    for (int i = 0; i < impl_->ring_depth; ++i) {
        if (impl_->ring[i].state == SlotState::READY) {
            slot_idx = i;
            break;
        }
    }

    if (slot_idx < 0) {
        return false;
    }

    auto& slot = impl_->ring[slot_idx];

    // Fill output
    out->m = impl_->m;
    out->B = slot.B;
    out->nnz = slot.nnz;
    out->d_col_ptr = slot.d_col_ptr.get();
    out->d_row_idx = slot.d_row_idx.get();
    out->d_values = slot.d_values.get();
    out->ready_event = slot.ready_event;
    out->eof_after = slot.eof_after;

    // Mark slot CONSUMED
    slot.state = SlotState::CONSUMED;
    --impl_->ready_count;
    impl_->consumed_slots.push_back(slot_idx);

    // Recycle old slots if we've consumed K batches
    while ((int)impl_->consumed_slots.size() >= impl_->ring_depth) {
        int old_slot = impl_->consumed_slots.front();
        impl_->consumed_slots.pop_front();
        impl_->ring[old_slot].state = SlotState::FREE;
        ++impl_->free_count;
    }

    impl_->ring_cv.notify_all();
    return true;
}

int DataLoader::m() const {
    return impl_->m;
}

cudaStream_t DataLoader::loader_stream() const {
    return impl_->loader_stream;
}
