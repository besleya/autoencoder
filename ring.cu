// SPDX-License-Identifier: MIT
// ring.cu — implementation of dispatcher and thread pools for strict round-robin batch scheduling.

#include "ring.h"
#include "data_loader.h"
#include "batch.h"

#include <iostream>
#include <stdexcept>
#include <thread>

// ============================================================================
// Constructor and destructor
// ============================================================================

Ring::Ring(int n_fill_workers, int n_decode_workers)
    : loaders_(),
      started_(false),
      shutting_down_(false),
      fill_pool_(n_fill_workers),
      decode_pool_(n_decode_workers),
      next_dispatch_idx_(0),
      next_consume_idx_(0),
      idle_window_(),
      idle_window_pos_(0),
      idle_window_count_(0) {
    if (n_fill_workers <= 0) {
        throw std::runtime_error("Ring: n_fill_workers must be > 0");
    }
    if (n_decode_workers <= 0) {
        throw std::runtime_error("Ring: n_decode_workers must be > 0");
    }
}

Ring::~Ring() {
    if (started_ && !shutting_down_) {
        shutdown();
    }
}

// ============================================================================
// Accessors
// ============================================================================

BS::thread_pool<>& Ring::decode_pool() {
    return decode_pool_;
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
    loader->set_ring(this);
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
    // Signal dispatcher to exit
    shutting_down_.store(true);

    // Join dispatcher thread (it checks shutting_down_ between cycles)
    if (dispatcher_thread_.joinable()) {
        dispatcher_thread_.join();
    }

    // Both pools drain/join automatically on destruction or via wait()
    // Explicitly wait for in-flight tasks to complete
    fill_pool_.wait();
    decode_pool_.wait();
}

// ============================================================================
// next_ready_batch
// ============================================================================

std::unique_ptr<Batch> Ring::next_ready_batch() {
    if (loaders_.empty()) {
        throw std::runtime_error("Ring::next_ready_batch: no loaders registered");
    }

    // Strict round-robin: block on the current loader until it has a ready batch
    while (!shutting_down_) {
        DataLoader* dl = loaders_[next_consume_idx_];
        auto batch = dl->take_ready_batch();
        
        if (batch) {
            next_consume_idx_ = (next_consume_idx_ + 1) % loaders_.size();
            return batch;
        }
        
        // If we got nullptr and shutting_down, exit gracefully
        if (shutting_down_) {
            return nullptr;
        }
    }

    return nullptr;
}

// ============================================================================
// dispatcher_loop
// ============================================================================

void Ring::dispatcher_loop() {
    int cycle = 0;  // track full cycles for rebalancing

    while (!shutting_down_) {
        if (loaders_.empty()) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }

        // One full cycle over all loaders
        for (size_t i = 0; i < loaders_.size(); ++i) {
            if (shutting_down_) {
                break;
            }

            DataLoader* dl = loaders_[next_dispatch_idx_];

            // STRICT ROTATION: block until this loader's next slot is empty
            // (DataLoader::reserve_slot() handles the blocking via CUDA events)
            Slot* slot = dl->reserve_slot();
            if (!slot) {
                if (shutting_down_) break;
                continue;
            }

            // Submit fill task to fill_pool_
            fill_pool_.submit_task([dl, slot] {
                dl->fill(slot);
            });

            next_dispatch_idx_ = (next_dispatch_idx_ + 1) % loaders_.size();
        }

        // Once per full cycle, check rebalancing
        if (!shutting_down_) {
            maybe_rebalance();
        }

        cycle++;
    }
}

// ============================================================================
// maybe_rebalance
// ============================================================================

namespace {
// Helper: compute mean of an array of size_t values
double compute_mean(const std::array<std::size_t, Ring::kWindowCycles>& samples) {
    std::size_t sum = 0;
    for (std::size_t val : samples) {
        sum += val;
    }
    return static_cast<double>(sum) / Ring::kWindowCycles;
}
}  // anonymous namespace

void Ring::maybe_rebalance() {
    // Averaging-window design: track rolling average of fill_pool_'s idle-thread count
    // over kWindowCycles cycles. Sample each cycle, act only when window is full.

    const std::size_t hw = std::thread::hardware_concurrency();
    const std::size_t fill_threads = fill_pool_.get_thread_count();
    const std::size_t idle = fill_threads - fill_pool_.get_tasks_running();

    // Sample idle count and store in ring buffer
    idle_window_[idle_window_pos_++ % kWindowCycles] = idle;
    if (++idle_window_count_ < kWindowCycles) {
        return;  // wait for a full window before acting
    }
    idle_window_count_ = 0;  // reset for next window

    // Compute mean idle count over the window
    const double avg_idle = compute_mean(idle_window_);

    // Shrink: persistent idleness
    if (avg_idle > kShrinkAbove) {
        // Target: shrink down toward what's actually being used, with a margin.
        std::size_t target = std::max<std::size_t>(
            kMinFillThreads,
            fill_threads - static_cast<std::size_t>(avg_idle - kIdleMargin)
        );
        std::size_t shrink_by = fill_threads - target;
        if (shrink_by > 0) {
            std::size_t decode_threads = decode_pool_.get_thread_count();
            std::size_t decode_room = hw - decode_threads;  // per-pool ceiling
            std::size_t move = std::min(shrink_by, decode_room);  // transfer only what fits
            fill_pool_.reset(target);
            if (move > 0) {
                decode_pool_.reset(decode_threads + move);
            }
            // Any threads in shrink_by - move are simply destroyed, not transferred.
        }
    }
    // Grow: persistent busyness
    else if (avg_idle < kGrowBelow && fill_threads < hw) {
        // Pull idle capacity from decode_pool_ to boost fill_pool_.
        std::size_t room = hw - fill_threads;
        std::size_t decode_threads = decode_pool_.get_thread_count();
        std::size_t available = (decode_threads > kMinDecodeThreads)
                                    ? (decode_threads - kMinDecodeThreads)
                                    : 0;
        std::size_t move = std::min(room, available);
        if (move > 0) {
            decode_pool_.reset(decode_threads - move);
            fill_pool_.reset(fill_threads + move);
        }
    }
}


