// SPDX-License-Identifier: MIT
// gpu_data_loader.cu — implementation of async sparse minibatch GPU DataLoader

#include "gpu_data_loader.h"
#include "gpu_timer.h"
#include "data.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <future>
#include <cstdlib>
#include <nvtx3/nvToolsExt.h>

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t err = (expr);                                                \
    if (err != cudaSuccess) {                                                \
      std::ostringstream oss;                                                \
      oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "         \
          << cudaGetErrorString(err);                                        \
      throw std::runtime_error(oss.str());                                   \
    }                                                                         \
  } while (0)

// Non-throwing version for use in destructors and other noexcept contexts
#define CUDA_CHECK_SILENT(expr)                                               \
  do {                                                                        \
    cudaError_t err = (expr);                                                \
    if (err != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(err));                                 \
    }                                                                         \
  } while (0)

// Scaler used by per-column log-normalization: ln(1 + S * x / sum_col)
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

GpuDataLoader::GpuDataLoader(const std::vector<std::string>& paths, int chunk_size,
                             int batch_size, std::mt19937& rng)
    : file_paths_(paths),
      chunk_size_(chunk_size),
      batch_size_(batch_size),
      rng_(rng),
      m_(0),
      active_slot_(0),
      stream_(nullptr),
      loader_stream_(nullptr),
      d_batch_col_ptr_(nullptr),
      d_batch_row_idx_(nullptr),
      d_batch_values_(nullptr),
      d_batch_capacity_(0),
      h_batch_col_ptr_(nullptr),
      next_chunk_start_idx_(0),
      epoch_eof_(false) {
    if (file_paths_.empty()) {
        throw std::runtime_error("GpuDataLoader: no files provided");
    }
    if (chunk_size <= 0) {
        throw std::runtime_error("GpuDataLoader: chunk_size must be > 0");
    }
    if (batch_size <= 0) {
        throw std::runtime_error("GpuDataLoader: batch_size must be > 0");
    }

    // Create two CUDA streams: one for consumer, one for loader
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, -1));
    CUDA_CHECK(cudaStreamCreateWithPriority(&loader_stream_, cudaStreamNonBlocking, 0));

    // Allocate pinned host buffer for batch col_ptr (B+1 int32)
    size_t col_ptr_bytes = (batch_size + 1) * sizeof(int32_t);
    CUDA_CHECK(cudaMallocHost(&h_batch_col_ptr_, col_ptr_bytes));

    // Initialize chunk slots (device buffers allocated on demand)
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventCreate(&slots_[i].ready_evt));
    }

    // Start worker thread
    worker_ = std::thread(&GpuDataLoader::worker_thread_, this);
}

GpuDataLoader::~GpuDataLoader() {
    // Stop worker thread
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }

    // Free slot buffers
    for (int i = 0; i < 2; ++i) {
        if (slots_[i].d_col_ptr) {
            CUDA_CHECK_SILENT(cudaFree(slots_[i].d_col_ptr));
        }
        if (slots_[i].d_row_idx) {
            CUDA_CHECK_SILENT(cudaFree(slots_[i].d_row_idx));
        }
        if (slots_[i].d_values) {
            CUDA_CHECK_SILENT(cudaFree(slots_[i].d_values));
        }
        if (slots_[i].ready_evt) {
            CUDA_CHECK_SILENT(cudaEventDestroy(slots_[i].ready_evt));
        }
    }

    // Free batch buffers
    if (h_batch_col_ptr_) {
        CUDA_CHECK_SILENT(cudaFreeHost(h_batch_col_ptr_));
    }
    if (d_batch_col_ptr_) {
        CUDA_CHECK_SILENT(cudaFree(d_batch_col_ptr_));
    }
    if (d_batch_row_idx_) {
        CUDA_CHECK_SILENT(cudaFree(d_batch_row_idx_));
    }
    if (d_batch_values_) {
        CUDA_CHECK_SILENT(cudaFree(d_batch_values_));
    }

    // Destroy streams
    if (stream_) {
        CUDA_CHECK_SILENT(cudaStreamDestroy(stream_));
    }
    if (loader_stream_) {
        CUDA_CHECK_SILENT(cudaStreamDestroy(loader_stream_));
    }
}

int GpuDataLoader::m() const { return m_; }

void GpuDataLoader::begin_epoch() {
    // Shuffle file paths if more than one file (RNG consumed on main thread).
    file_order_for_epoch_ = file_paths_;
    if (file_order_for_epoch_.size() > 1) {
        std::shuffle(file_order_for_epoch_.begin(), file_order_for_epoch_.end(), rng_);
    }

    next_chunk_start_idx_ = 0;
    epoch_eof_ = false;
    active_slot_ = 0;

    // Reset slot state
    for (int i = 0; i < 2; ++i) {
        slots_[i].valid = false;
        slots_[i].eof = false;
        slots_[i].current_batch_idx = 0;
        slots_[i].batches_remaining = 0;
        slots_[i].order.clear();
    }

    // Enqueue first two chunks for prefetch
    for (int slot = 0; slot < 2; ++slot) {
        size_t start_idx = next_chunk_start_idx_;
        size_t end_idx = std::min(start_idx + chunk_size_, file_order_for_epoch_.size());
        if (start_idx >= file_order_for_epoch_.size()) {
            // No more chunks; mark as EOF
            std::lock_guard<std::mutex> lock(mtx_);
            slots_[slot].eof = true;
            slots_[slot].valid = true;
            break;
        }

        // Collect files for this chunk
        LoadRequest req;
        req.slot_idx = slot;
        for (size_t i = start_idx; i < end_idx; ++i) {
            req.files.push_back(file_order_for_epoch_[i]);
        }

        // Enqueue load request (shuffle order will be generated later on main thread)
        {
            std::lock_guard<std::mutex> lock(mtx_);
            work_queue_.push(req);
        }
        cv_.notify_one();

        next_chunk_start_idx_ = end_idx;
        if (end_idx >= file_order_for_epoch_.size()) {
            epoch_eof_ = true;
            break;
        }
    }
}

void GpuDataLoader::worker_thread_() {
    while (true) {
        LoadRequest req;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this]() { return stop_ || !work_queue_.empty(); });
            if (stop_) break;
            if (!work_queue_.empty()) {
                req = work_queue_.front();
                work_queue_.pop();
            } else {
                continue;
            }
        }

        // Load chunk into slot
        try {
            load_chunk_into_slot_(req);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "Worker thread: load_chunk failed: %s\n", e.what());
            // Mark slot as invalid/eof so the main thread doesn't hang
            std::lock_guard<std::mutex> lock(mtx_);
            slots_[req.slot_idx].eof = true;
            slots_[req.slot_idx].valid = true;
            cv_.notify_all();
        }
    }
}

void GpuDataLoader::load_chunk_into_slot_(const LoadRequest& req) {
    nvtxRangePushA("GpuDataLoader::load_chunk_into_slot_");
    GpuScopedTimer timer_total("loader.prefetch.total", loader_stream_);
    
    int slot_idx = req.slot_idx;
    ChunkSlot& slot = slots_[slot_idx];
    
    if (req.files.empty()) {
        // EOF marker
        std::lock_guard<std::mutex> lock(mtx_);
        slot.eof = true;
        slot.valid = true;
        cv_.notify_all();
        nvtxRangePop();
        return;
    }

    // Load the chunk using existing load_dataset (which does parallel decode internally)
    Dataset chunk_data;
    {
        GpuScopedTimer timer_load("loader.prefetch.load_dataset", loader_stream_);
        chunk_data = load_dataset(req.files);
    }

    if (m_ == 0) {
        m_ = chunk_data.m;  // Set feature count on first load
    }
    
    int chunk_n = chunk_data.n;
    int chunk_nnz = static_cast<int>(chunk_data.nnz);

    // Allocate/reallocate slot buffers if needed
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (static_cast<size_t>(chunk_n) > slot.cap_n) {
            if (slot.d_col_ptr) {
                CUDA_CHECK(cudaFree(slot.d_col_ptr));
            }
            CUDA_CHECK(cudaMalloc(&slot.d_col_ptr, (chunk_n + 1) * sizeof(int32_t)));
            slot.cap_n = chunk_n;
        }
        if (static_cast<size_t>(chunk_nnz) > slot.cap_nnz) {
            if (slot.d_row_idx) {
                CUDA_CHECK(cudaFree(slot.d_row_idx));
            }
            if (slot.d_values) {
                CUDA_CHECK(cudaFree(slot.d_values));
            }
            CUDA_CHECK(cudaMalloc(&slot.d_row_idx, chunk_nnz * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&slot.d_values, chunk_nnz * sizeof(float)));
            slot.cap_nnz = chunk_nnz;
        }
        slot.n = chunk_n;
        slot.nnz = chunk_nnz;
        slot.host_col_ptr.resize(chunk_n + 1);
    }

    // Copy col_ptr to host
    CUDA_CHECK(cudaMemcpy(slot.host_col_ptr.data(), chunk_data.X.col_ptr.get(),
                          (chunk_n + 1) * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // Upload to slot buffers on loader_stream_
    {
        GpuScopedTimer timer_h2d("loader.prefetch.h2d", loader_stream_);
        CUDA_CHECK(cudaMemcpyAsync(slot.d_col_ptr, chunk_data.X.col_ptr.get(),
                                   (chunk_n + 1) * sizeof(int32_t),
                                   cudaMemcpyDeviceToDevice, loader_stream_));
        CUDA_CHECK(cudaMemcpyAsync(slot.d_row_idx, chunk_data.X.row_indices.get(),
                                   chunk_nnz * sizeof(int32_t),
                                   cudaMemcpyDeviceToDevice, loader_stream_));
        CUDA_CHECK(cudaMemcpyAsync(slot.d_values, chunk_data.X.values.get(),
                                   chunk_nnz * sizeof(float),
                                   cudaMemcpyDeviceToDevice, loader_stream_));
    }

    // GPU log-normalization in place on slot.d_values
    if (chunk_n > 0) {
        GpuScopedTimer timer_kernel("loader.prefetch.lognorm", loader_stream_);
        dim3 block(256);
        dim3 grid(chunk_n);
        log_normalize_columns_kernel<<<grid, block, 0, loader_stream_>>>(
            chunk_n, kSparseLogNormScaler,
            slot.d_col_ptr,
            slot.d_values);
        CUDA_CHECK(cudaGetLastError());
    }

    // Record ready event on loader_stream_
    CUDA_CHECK(cudaEventRecord(slot.ready_evt, loader_stream_));

    // Mark slot as valid (shuffle order will be generated by main thread)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        slot.valid = true;
        slot.eof = false;
        slot.order.clear();  // Will be populated by main thread
        slot.current_batch_idx = 0;
        cv_.notify_all();
    }
    
    nvtxRangePop();
}

int GpuDataLoader::compute_batch_nnz_(const ChunkSlot& slot, const std::vector<int>& cols) {
    int total_nnz = 0;
    for (int col : cols) {
        total_nnz += slot.host_col_ptr[col + 1] - slot.host_col_ptr[col];
    }
    return total_nnz;
}

bool GpuDataLoader::next_batch(SparseBatch* out) {
    nvtxRangePushA("GpuDataLoader::next_batch");
    GpuScopedTimer timer_total("loader.next_batch.total", stream_);
    
    if (!out) {
        throw std::runtime_error("next_batch: out pointer is null");
    }

    // Check if we need to switch to the next slot
    ChunkSlot* slot = &slots_[active_slot_];
    const int num_full_batches = slot->n / batch_size_;
    
    if (slot->current_batch_idx >= num_full_batches) {
        // Current slot exhausted; switch to next slot
        if (slot->eof) {
            // No more data
            nvtxRangePop();
            return false;
        }

        int next_slot = 1 - active_slot_;
        
        // Wait for next slot to be ready
        {
            GpuScopedTimer timer_wait("loader.next_batch.wait_slot", stream_);
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this, next_slot]() { return slots_[next_slot].valid; });
        }
        
        // Generate shuffle order for next slot if not already done (RNG on main thread)
        if (!slots_[next_slot].eof && slots_[next_slot].n > 0 && slots_[next_slot].order.empty()) {
            GpuScopedTimer timer_shuffle("loader.next_batch.gen_shuffle", stream_);
            slots_[next_slot].order.resize(slots_[next_slot].n);
            std::iota(slots_[next_slot].order.begin(), slots_[next_slot].order.end(), 0);
            std::shuffle(slots_[next_slot].order.begin(), slots_[next_slot].order.end(), rng_);
        }
        
        // Wait on GPU event to ensure loader stream's work is done
        {
            GpuScopedTimer timer_wait_evt("loader.next_batch.wait_event", stream_);
            CUDA_CHECK(cudaStreamWaitEvent(stream_, slots_[next_slot].ready_evt, 0));
        }
        
        // Switch to next slot
        active_slot_ = next_slot;
        slot = &slots_[active_slot_];  // Update pointer
        
        // Enqueue reload of the previous slot (if more data available)
        if (!epoch_eof_) {
            size_t start_idx = next_chunk_start_idx_;
            size_t end_idx = std::min(start_idx + chunk_size_, file_order_for_epoch_.size());
            if (start_idx < file_order_for_epoch_.size()) {
                LoadRequest req;
                req.slot_idx = 1 - active_slot_;  // Reload the slot we just finished
                for (size_t i = start_idx; i < end_idx; ++i) {
                    req.files.push_back(file_order_for_epoch_[i]);
                }
                
                {
                    std::lock_guard<std::mutex> lock(mtx_);
                    slots_[req.slot_idx].valid = false;  // Mark as being reloaded
                    work_queue_.push(req);
                }
                cv_.notify_one();
                
                next_chunk_start_idx_ = end_idx;
                if (end_idx >= file_order_for_epoch_.size()) {
                    epoch_eof_ = true;
                }
            } else {
                // Mark the old slot as EOF for next time
                slots_[1 - active_slot_].eof = true;
                slots_[1 - active_slot_].valid = true;
            }
        } else {
            // Mark the old slot as EOF
            slots_[1 - active_slot_].eof = true;
            slots_[1 - active_slot_].valid = true;
        }
        
        // Reset batch index for new slot
        slot->current_batch_idx = 0;
    }
    
    // Generate shuffle order for current slot if not already done (first access)
    if (!slot->eof && slot->n > 0 && slot->order.empty()) {
        GpuScopedTimer timer_shuffle("loader.next_batch.gen_shuffle_first", stream_);
        slot->order.resize(slot->n);
        std::iota(slot->order.begin(), slot->order.end(), 0);
        std::shuffle(slot->order.begin(), slot->order.end(), rng_);
        // Also issue stream wait event for first batch from this slot
        CUDA_CHECK(cudaStreamWaitEvent(stream_, slot->ready_evt, 0));
    }

    // Extract batch from current slot
    int start_col = slot->current_batch_idx * batch_size_;
    std::vector<int> batch_cols;
    for (int j = 0; j < batch_size_; ++j) {
        batch_cols.push_back(slot->order[start_col + j]);
    }

    // Compute total nnz in this batch
    int batch_nnz = compute_batch_nnz_(*slot, batch_cols);

    // Allocate device col_ptr buffer if not already done
    if (!d_batch_col_ptr_) {
        CUDA_CHECK(cudaMalloc(&d_batch_col_ptr_,
                              (batch_size_ + 1) * sizeof(int32_t)));
    }

    // Build batch col_ptr on host (pinned buffer)
    h_batch_col_ptr_[0] = 0;
    int offset = 0;
    for (int j = 0; j < batch_size_; ++j) {
        int col = batch_cols[j];
        int col_nnz = slot->host_col_ptr[col + 1] - slot->host_col_ptr[col];
        offset += col_nnz;
        h_batch_col_ptr_[j + 1] = offset;
    }

    // Allocate/reallocate batch buffers if needed
    if (static_cast<size_t>(batch_nnz) > d_batch_capacity_) {
        if (d_batch_row_idx_) {
            CUDA_CHECK(cudaFree(d_batch_row_idx_));
        }
        if (d_batch_values_) {
            CUDA_CHECK(cudaFree(d_batch_values_));
        }
        size_t new_cap = batch_nnz * 2;  // 2x headroom
        CUDA_CHECK(cudaMalloc(&d_batch_row_idx_, new_cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_batch_values_, new_cap * sizeof(float)));
        d_batch_capacity_ = new_cap;
    }

    // Copy col_ptr to device
    {
        GpuScopedTimer timer_col_ptr("loader.next_batch.col_ptr_h2d", stream_);
        CUDA_CHECK(cudaMemcpyAsync(d_batch_col_ptr_, h_batch_col_ptr_,
                                   (batch_size_ + 1) * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream_));
    }

    // Issue D2D copies for row indices and values from slot buffers to batch buffers
    {
        GpuScopedTimer timer_d2d("loader.next_batch.row_idx_values_d2d_loop", stream_);
        int batch_offset = 0;
        for (int j = 0; j < batch_size_; ++j) {
            int col = batch_cols[j];
            int col_start = slot->host_col_ptr[col];
            int col_end = slot->host_col_ptr[col + 1];
            int col_nnz = col_end - col_start;

            if (col_nnz > 0) {
                // Copy row indices from slot buffer to batch buffer
                CUDA_CHECK(cudaMemcpyAsync(
                    d_batch_row_idx_ + batch_offset,
                    slot->d_row_idx + col_start,
                    col_nnz * sizeof(int32_t),
                    cudaMemcpyDeviceToDevice,
                    stream_));

                // Copy values from slot buffer to batch buffer
                CUDA_CHECK(cudaMemcpyAsync(
                    d_batch_values_ + batch_offset,
                    slot->d_values + col_start,
                    col_nnz * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    stream_));

                batch_offset += col_nnz;
            }
        }
    }

    // Fill SparseBatch
    out->m = m_;
    out->B = batch_size_;
    out->nnz = batch_nnz;
    out->d_col_ptr = d_batch_col_ptr_;
    out->d_row_idx = d_batch_row_idx_;
    out->d_values = d_batch_values_;
    out->stream = stream_;

    slot->current_batch_idx++;
    nvtxRangePop();
    return true;
}
