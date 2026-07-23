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
