// SPDX-License-Identifier: MIT
// ring.cu — implementation of dispatcher and thread pool for strict round-robin batch scheduling.

#include "ring.h"
#include "data_loader.h"
#include "batch.h"

#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

// ============================================================================
// Constructor and destructor
// ============================================================================

Ring::Ring(int n_worker_threads)
    : loaders_(), started_(false), shutting_down_(false), shut_down_complete_(false) {
    if (n_worker_threads <= 0) {
        throw std::runtime_error("Ring: n_worker_threads must be > 0");
    }

    // Spawn worker threads
    for (int i = 0; i < n_worker_threads; ++i) {
        workers_.emplace_back([this] { pool_worker_loop(); });
    }
}


Ring::~Ring() {
    if (started_ && !shut_down_complete_) {
        shutdown();
    }
}


// ============================================================================
// add_loader
// ============================================================================

void Ring::add_loader(DataLoader* loader) {
    if (started_) {
        throw std::runtime_error("Ring::add_loader: cannot add loader after start()");
    }
    if (!loader) {
        throw std::runtime_error("Ring::add_loader: null loader");
    }
    loaders_.push_back(loader);
}

// ============================================================================
// start
// ============================================================================

void Ring::start() {
    if (loaders_.empty()) {
        throw std::runtime_error("Ring::start: at least one loader required");
    }
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) {
        throw std::runtime_error("Ring::start: already started");
    }

    dispatcher_thread_ = std::thread([this] { dispatcher_loop(); });
}

// ============================================================================
// shutdown
// ============================================================================

void Ring::shutdown() {
    // Idempotent: exit early if already shut down
    if (shut_down_complete_.exchange(true)) {
        return;  // Already complete
    }

    shutting_down_ = true;

    // Wake all workers
    {
        std::lock_guard<std::mutex> lock(pool_mtx_);
        pool_cv_.notify_all();
    }

    // Join dispatcher
    if (dispatcher_thread_.joinable()) {
        dispatcher_thread_.join();
    }

    // Join workers
    for (auto& w : workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
}

// ============================================================================
// next_ready_batch
// ============================================================================

std::unique_ptr<Batch> Ring::next_ready_batch() {
    if (loaders_.empty()) {
        throw std::runtime_error("Ring::next_ready_batch: no loaders registered");
    }

    // Strict round-robin: wait on the current loader
    while (!shutting_down_) {
        DataLoader* dl = loaders_[next_consume_idx_];
        auto batch = dl->take_ready_batch();
        if (!batch && shutting_down_) {
            return nullptr;
        }
        if (batch) {
            next_consume_idx_ = (next_consume_idx_ + 1) % loaders_.size();
            return batch;
        }
    }

    return nullptr;
}

// ============================================================================
// Private: dispatcher_loop
// ============================================================================

void Ring::dispatcher_loop() {
    while (!shutting_down_) {
        if (loaders_.empty()) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }

        DataLoader* dl = loaders_[next_dispatch_idx_];

        // STRICT ROTATION: wait for THIS loader to have a free slot, never skip
        Slot* slot = wait_for_free_slot(dl);
        if (!slot) {
            if (shutting_down_) break;
            continue;
        }

        // Submit fill task to pool
        submit_task([dl, slot] {
            dl->fill(slot);
        });

        next_dispatch_idx_ = (next_dispatch_idx_ + 1) % loaders_.size();
    }
}

// ============================================================================
// Private: wait_for_free_slot (polling + sleep)
// ============================================================================

Slot* Ring::wait_for_free_slot(DataLoader* loader) {
    // Poll until we get a free slot or shutdown is signaled.
    // Since DataLoader doesn't expose a blocking reserve, we use polling.
    // TODO: replace with blocking reserve_free_slot() when DataLoader exposes it.

    while (!shutting_down_) {
        Slot* slot = loader->reserve_free_slot();
        if (slot) {
            return slot;
        }
        // Poll every 50 microseconds to avoid burning CPU
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    return nullptr;
}

// ============================================================================
// Private: pool_worker_loop
// ============================================================================

void Ring::pool_worker_loop() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(pool_mtx_);

            // Wait for a task or shutdown signal
            pool_cv_.wait(lock, [this] {
                return !task_queue_.empty() || shutting_down_;
            });

            // Check for shutdown with empty queue
            if (shutting_down_ && task_queue_.empty()) {
                break;
            }

            // Dequeue if available
            if (!task_queue_.empty()) {
                task = std::move(task_queue_.front());
                task_queue_.pop();
            }
        }

        // Execute task outside lock
        if (task) {
            task();
        }
    }
}

// ============================================================================
// Private: submit_task
// ============================================================================

void Ring::submit_task(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(pool_mtx_);
        task_queue_.push(std::move(task));
    }
    pool_cv_.notify_one();
}

    std::unique_lock<std::mutex> lock(mtx_);

    // Define the wait predicate
    auto predicate = [this, lane] {
        return lanes_[lane].free_count > 0 || stopped_;
    };

    // Check if we need to wait
    bool was_ready = predicate();
    auto t_start = std::chrono::steady_clock::now();

    if (!was_ready) {
        cv_.wait(lock, predicate);
    }

    auto t_end = std::chrono::steady_clock::now();

    if (stopped_) return -1;

    // Find the first FREE slot
    int slot_idx = -1;
    for (int i = 0; i < ring_depth_; ++i) {
        if (lanes_[lane].slots[i].state == Ring::Slot::State::FREE) {
            slot_idx = i;
            break;
        }
    }

    if (slot_idx == -1) {
        throw std::runtime_error("Ring::acquire_free: no FREE slot found (logic error)");
    }

    // Transition to BUILDING
    lanes_[lane].slots[slot_idx].state = Ring::Slot::State::BUILDING;
    lanes_[lane].free_count--;

    // Update stats only if we actually waited
    if (!was_ready) {
        auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            t_end - t_start).count();
        lanes_[lane].stats.producer_waits++;
        lanes_[lane].stats.producer_wait_ns_total += elapsed_ns;
    }

    return slot_idx;
}

Ring::SlotView Ring::slot_view(int lane, int slot_idx) {
    if (lane < 0 || lane >= n_lanes_ || slot_idx < 0 || slot_idx >= ring_depth_) {
        throw std::runtime_error("Ring::slot_view: invalid lane or slot_idx");
    }

    Ring::Slot& slot = lanes_[lane].slots[slot_idx];

    return Ring::SlotView{
        .h_col_ptr = slot.h_col_ptr,
        .h_row_idx = slot.h_row_idx,
        .h_values = slot.h_values,
        .h_col_capacity = slot.h_col_capacity,
        .h_row_capacity = slot.h_row_capacity,
        .h_val_capacity = slot.h_val_capacity,
        .d_col_ptr = slot.d_col_ptr,
        .d_row_idx = slot.d_row_idx,
        .d_values = slot.d_values,
        .d_col_capacity = slot.d_col_capacity,
        .d_row_capacity = slot.d_row_capacity,
        .d_val_capacity = slot.d_val_capacity,
        .ready_event = slot.ready_event,
    };
}

void Ring::ensure_capacity(int lane, int slot_idx,
                           size_t col_ptr_n, size_t row_idx_n, size_t val_n) {
    if (lane < 0 || lane >= n_lanes_ || slot_idx < 0 || slot_idx >= ring_depth_) {
        throw std::runtime_error("Ring::ensure_capacity: invalid lane or slot_idx");
    }

    Ring::Slot& slot = lanes_[lane].slots[slot_idx];

    // Pinned host buffer: col_ptr
    if (col_ptr_n > slot.h_col_capacity) {
        if (slot.h_col_ptr) CUDA_CHECK(cudaFreeHost(slot.h_col_ptr));
        CUDA_CHECK(cudaMallocHost(&slot.h_col_ptr, col_ptr_n * sizeof(int32_t)));
        slot.h_col_capacity = col_ptr_n;
    }

    // Pinned host buffer: row_idx
    if (row_idx_n > slot.h_row_capacity) {
        if (slot.h_row_idx) CUDA_CHECK(cudaFreeHost(slot.h_row_idx));
        CUDA_CHECK(cudaMallocHost(&slot.h_row_idx, row_idx_n * sizeof(int32_t)));
        slot.h_row_capacity = row_idx_n;
    }

    // Pinned host buffer: values
    if (val_n > slot.h_val_capacity) {
        if (slot.h_values) CUDA_CHECK(cudaFreeHost(slot.h_values));
        CUDA_CHECK(cudaMallocHost(&slot.h_values, val_n * sizeof(float)));
        slot.h_val_capacity = val_n;
    }

    // Device buffer: col_ptr
    if (col_ptr_n > slot.d_col_capacity) {
        if (slot.d_col_ptr) CUDA_CHECK(cudaFree(slot.d_col_ptr));
        CUDA_CHECK(cudaMalloc(&slot.d_col_ptr, col_ptr_n * sizeof(int32_t)));
        slot.d_col_capacity = col_ptr_n;
    }

    // Device buffer: row_idx
    if (row_idx_n > slot.d_row_capacity) {
        if (slot.d_row_idx) CUDA_CHECK(cudaFree(slot.d_row_idx));
        CUDA_CHECK(cudaMalloc(&slot.d_row_idx, row_idx_n * sizeof(int32_t)));
        slot.d_row_capacity = row_idx_n;
    }

    // Device buffer: values
    if (val_n > slot.d_val_capacity) {
        if (slot.d_values) CUDA_CHECK(cudaFree(slot.d_values));
        CUDA_CHECK(cudaMalloc(&slot.d_values, val_n * sizeof(float)));
        slot.d_val_capacity = val_n;
    }
}

void Ring::publish_ready(int lane, int slot_idx, int B, int nnz, bool chunk_end) {
    if (lane < 0 || lane >= n_lanes_ || slot_idx < 0 || slot_idx >= ring_depth_) {
        throw std::runtime_error("Ring::publish_ready: invalid lane or slot_idx");
    }

    std::lock_guard<std::mutex> lock(mtx_);

    Ring::Slot& slot = lanes_[lane].slots[slot_idx];
    if (slot.state != Ring::Slot::State::BUILDING) {
        throw std::runtime_error("Ring::publish_ready: slot not in BUILDING state");
    }

    // Update metadata
    slot.B = B;
    slot.nnz = nnz;
    slot.chunk_end = chunk_end;

    // Transition to READY
    slot.state = Ring::Slot::State::READY;
    lanes_[lane].ready_count++;
    lanes_[lane].stats.batches_published++;

    // Notify trainer-side waiters
    cv_.notify_all();
}

bool Ring::acquire_ready(SparseBatch* out, int* out_lane, int* out_slot) {
    if (!out || !out_lane || !out_slot) {
        throw std::runtime_error("Ring::acquire_ready: null pointer argument");
    }

    std::unique_lock<std::mutex> lock(mtx_);

    // Define the wait predicate: some lane has a READY slot OR shutdown
    auto predicate = [this] {
        for (int i = 0; i < n_lanes_; ++i) {
            if (lanes_[i].ready_count > 0) return true;
        }
        return stopped_.load();
    };

    // Check if we need to wait
    bool was_ready = predicate();
    auto t_start = std::chrono::steady_clock::now();

    if (!was_ready) {
        std::cout << "[HANGDB] WASN'T READY Ring::acquire_ready(): waiting for READY batch from any lane" << std::endl;
        cv_.wait(lock, predicate);
        std::cout << "[HANGDB] Ring::acquire_ready(): READY batch available" << std::endl;
    }

    auto t_end = std::chrono::steady_clock::now();

    if (stopped_) return false;

    // Pick a lane according to policy
    int chosen_lane = -1;
    switch (policy_) {
        case AlternationPolicy::SINGLE: {
            if (lanes_[0].ready_count > 0) chosen_lane = 0;
            break;
        }
        case AlternationPolicy::ROUND_ROBIN: {
            // Scan starting from next_lane_rr_
            for (int offset = 0; offset < n_lanes_; ++offset) {
                int lane = (next_lane_rr_ + offset) % n_lanes_;
                if (lanes_[lane].ready_count > 0) {
                    chosen_lane = lane;
                    next_lane_rr_ = (lane + 1) % n_lanes_;
                    break;
                }
            }
            break;
        }
        case AlternationPolicy::WEIGHTED: {
            // Use precomputed schedule
            if (!weighted_schedule_.empty()) {
                for (int offset = 0; offset < static_cast<int>(weighted_schedule_.size());
                     ++offset) {
                    int lane = weighted_schedule_[(next_lane_rr_ + offset) %
                                                  weighted_schedule_.size()];
                    if (lanes_[lane].ready_count > 0) {
                        chosen_lane = lane;
                        next_lane_rr_ = (next_lane_rr_ + offset + 1) %
                                       weighted_schedule_.size();
                        break;
                    }
                }
            }
            break;
        }
        case AlternationPolicy::CALLER_CHOOSES:
            // This shouldn't be called; use acquire_ready_from instead
            throw std::runtime_error(
                "Ring::acquire_ready: CALLER_CHOOSES policy requires acquire_ready_from");
    }

    if (chosen_lane == -1) {
        // No READY slot found; shouldn't happen if predicate was correct
        return false;
    }

    // Find the first READY slot in chosen_lane, starting from next_slot_idx with rotation
    int slot_idx = -1;
    for (int offset = 0; offset < ring_depth_; ++offset) {
        int i = (lanes_[chosen_lane].next_slot_idx + offset) % ring_depth_;
        if (lanes_[chosen_lane].slots[i].state == Ring::Slot::State::READY) {
            slot_idx = i;
            break;
        }
    }

    if (slot_idx == -1) {
        throw std::runtime_error("Ring::acquire_ready: no READY slot found (logic error)");
    }

    // Transition to CONSUMED
    Ring::Slot& slot = lanes_[chosen_lane].slots[slot_idx];
    slot.state = Ring::Slot::State::CONSUMED;
    lanes_[chosen_lane].ready_count--;
    lanes_[chosen_lane].stats.batches_consumed++;
    lanes_[chosen_lane].next_slot_idx = (slot_idx + 1) % ring_depth_;  // Rotate to next slot

    // Update stats only if we actually waited
    if (!was_ready) {
        auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            t_end - t_start).count();
        lanes_[chosen_lane].stats.trainer_waits++;
        lanes_[chosen_lane].stats.trainer_wait_ns_total += elapsed_ns;
    }

    // Fill output
    out->m = m_;
    out->B = slot.B;
    out->nnz = slot.nnz;
    out->d_col_ptr = slot.d_col_ptr;
    out->d_row_idx = slot.d_row_idx;
    out->d_values = slot.d_values;
    out->ready_event = slot.ready_event;
    out->chunk_end = slot.chunk_end;
    *out_lane = chosen_lane;
    *out_slot = slot_idx;

    return true;
}

bool Ring::acquire_ready_from(int lane, SparseBatch* out, int* out_slot) {
    if (lane < 0 || lane >= n_lanes_) {
        throw std::runtime_error("Ring::acquire_ready_from: invalid lane");
    }
    if (!out || !out_slot) {
        throw std::runtime_error("Ring::acquire_ready_from: null pointer argument");
    }

    std::unique_lock<std::mutex> lock(mtx_);

    // Define the wait predicate: this lane has a READY slot OR shutdown
    auto predicate = [this, lane] {
        if (lanes_[lane].ready_count > 0) return true;
        return stopped_.load();
    };

    // Check if we need to wait
    bool was_ready = predicate();
    auto t_start = std::chrono::steady_clock::now();

    if (!was_ready) {
        std::cout << "[HANGDB] WASN'T READY Ring::acquire_ready_from(lane=" << lane << "): waiting for READY batch from lane " << lane << std::endl;
        cv_.wait(lock, predicate);
    }

    auto t_end = std::chrono::steady_clock::now();

    if (stopped_) return false;

    if (lanes_[lane].ready_count == 0) {
        throw std::runtime_error("Ring::acquire_ready_from: no READY slot (logic error)");
    }

    // Find the first READY slot in lane, starting from next_slot_idx with rotation
    int slot_idx = -1;
    for (int offset = 0; offset < ring_depth_; ++offset) {
        int i = (lanes_[lane].next_slot_idx + offset) % ring_depth_;
        if (lanes_[lane].slots[i].state == Ring::Slot::State::READY) {
            slot_idx = i;
            break;
        }
    }

    if (slot_idx == -1) {
        throw std::runtime_error("Ring::acquire_ready_from: no READY slot found (logic error)");
    }

    // Transition to CONSUMED
    Ring::Slot& slot = lanes_[lane].slots[slot_idx];
    slot.state = Ring::Slot::State::CONSUMED;
    lanes_[lane].ready_count--;
    lanes_[lane].stats.batches_consumed++;
    lanes_[lane].next_slot_idx = (slot_idx + 1) % ring_depth_;  // Rotate to next slot

    // Update stats only if we actually waited
    if (!was_ready) {
        auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            t_end - t_start).count();
        lanes_[lane].stats.trainer_waits++;
        lanes_[lane].stats.trainer_wait_ns_total += elapsed_ns;
    }

    // Fill output
    out->m = m_;
    out->B = slot.B;
    out->nnz = slot.nnz;
    out->d_col_ptr = slot.d_col_ptr;
    out->d_row_idx = slot.d_row_idx;
    out->d_values = slot.d_values;
    out->ready_event = slot.ready_event;
    out->chunk_end = slot.chunk_end;
    *out_slot = slot_idx;

    return true;
}

void Ring::release_consumed(int lane, int slot_idx) {
    if (lane < 0 || lane >= n_lanes_ || slot_idx < 0 || slot_idx >= ring_depth_) {
        throw std::runtime_error("Ring::release_consumed: invalid lane or slot_idx");
    }

    std::lock_guard<std::mutex> lock(mtx_);

    Ring::Slot& slot = lanes_[lane].slots[slot_idx];
    if (slot.state != Ring::Slot::State::CONSUMED) {
        throw std::runtime_error("Ring::release_consumed: slot not in CONSUMED state");
    }

    // Transition to FREE
    slot.state = Ring::Slot::State::FREE;
    lanes_[lane].free_count++;

    // Notify producer-side waiters
    cv_.notify_all();
}

void Ring::begin_epoch(int lane) {
    if (lane < 0 || lane >= n_lanes_) {
        throw std::runtime_error("Ring::begin_epoch: invalid lane");
    }

    std::lock_guard<std::mutex> lock(mtx_);

    lanes_[lane].next_slot_idx = 0;  // Reset slot rotation at epoch start
}

Ring::Stats Ring::stats(int lane) const {
    if (lane < 0 || lane >= n_lanes_) {
        throw std::runtime_error("Ring::stats: invalid lane");
    }

    std::lock_guard<std::mutex> lock(mtx_);
    return lanes_[lane].stats;
}

Ring::Stats Ring::stats_total() const {
    std::lock_guard<std::mutex> lock(mtx_);

    Stats total{};
    for (const auto& lane : lanes_) {
        total.trainer_waits += lane.stats.trainer_waits;
        total.trainer_wait_ns_total += lane.stats.trainer_wait_ns_total;
        total.producer_waits += lane.stats.producer_waits;
        total.producer_wait_ns_total += lane.stats.producer_wait_ns_total;
        total.batches_published += lane.stats.batches_published;
        total.batches_consumed += lane.stats.batches_consumed;
    }
    return total;
}

void Ring::reset_stats() {
    std::lock_guard<std::mutex> lock(mtx_);

    for (auto& lane : lanes_) {
        lane.stats = Stats{};
    }
}

void Ring::set_weight(int lane, double w) {
    if (lane < 0 || lane >= n_lanes_) {
        throw std::runtime_error("Ring::set_weight: invalid lane");
    }

    std::lock_guard<std::mutex> lock(mtx_);
    lanes_[lane].weight = w;

    if (policy_ == AlternationPolicy::WEIGHTED) {
        rebuild_weighted_schedule_impl(weighted_schedule_, lanes_);
    }
}

void Ring::set_m(int m) {
    std::lock_guard<std::mutex> lock(mtx_);
    m_ = m;
}

int Ring::m() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return m_;
}

int Ring::n_lanes() const {
    return n_lanes_;
}

int Ring::ring_depth() const {
    return ring_depth_;
}

void Ring::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stopped_ = true;
    }
    cv_.notify_all();
}

int Ring::lane_pass(int lane_id) const {
    if (lane_id < 0 || lane_id >= n_lanes_) {
        throw std::runtime_error("Ring::lane_pass: invalid lane_id");
    }
    return lanes_[lane_id].pass_counter.load();
}

void Ring::increment_lane_pass(int lane_id) {
    if (lane_id < 0 || lane_id >= n_lanes_) {
        throw std::runtime_error("Ring::increment_lane_pass: invalid lane_id");
    }
    lanes_[lane_id].pass_counter.fetch_add(1, std::memory_order_relaxed);
}

bool Ring::is_shutdown() const {
    return stopped_.load();
}
