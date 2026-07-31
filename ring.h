// SPDX-License-Identifier: MIT
// ring.h — Dispatcher and thread pools for strict round-robin batch scheduling.

#pragma once

#include <atomic>
#include <cuda_runtime.h>
#include <memory>
#include <thread>
#include <vector>

#include "BS_thread_pool.hpp"

// Forward declarations
class DataLoader;
class Batch;

// ============================================================================
// Ring — dispatcher and thread pool manager for strict round-robin prefetch
// ============================================================================

class Ring {
public:
    // Constructor: creates two thread pools.
    // fill_pool_: small pool for DataLoader::fill() tasks (default 2).
    // decode_pool_: large pool for per-file decode tasks (default hardware_concurrency()).
    // Empty loader list; loaders added via add_loader() before start().
    Ring(int n_fill_workers, 
         int n_decode_workers = std::thread::hardware_concurrency());

    // Destructor: calls shutdown() if not already shut down.
    ~Ring();

    // Non-copyable, non-movable.
    Ring(const Ring&) = delete;
    Ring& operator=(const Ring&) = delete;
    Ring(Ring&&) = delete;
    Ring& operator=(Ring&&) = delete;

    // ========================================================================
    // Pool accessor (for DataLoaders to submit decode tasks)
    // ========================================================================

    // Returns the shared decode pool. DataLoaders use this to submit per-file
    // decode tasks in parallel.
    BS::thread_pool& decode_pool();

    // ========================================================================
    // Loader registration (must be called before start())
    // ========================================================================

    // Register a DataLoader for round-robin dispatch. Stores the pointer and
    // calls loader->set_ring(this) so the loader can access decode_pool().
    // Order of registration determines round-robin order.
    // Loader must outlive Ring.
    void add_loader(DataLoader* loader);

    // ========================================================================
    // Lifecycle
    // ========================================================================

    // Launch the dispatcher thread. Requires at least one registered loader.
    // Must be called exactly once.
    void start();

    // Signal dispatcher and workers to exit, drain the pools, join all threads.
    // Idempotent. Safe to call from any thread.
    void shutdown();

    // ========================================================================
    // Consumer API (called by trainer thread)
    // ========================================================================

    // Block until the next loader in round-robin order has a ready batch.
    // Returns a Batch in strict round-robin order; advances next_consume_idx_.
    // This implements strict rotation: if loaders_[next_consume_idx_] has no
    // ready batch yet, blocks on it (never skips to another loader).
    std::unique_ptr<Batch> next_ready_batch();

private:
    // ========================================================================
    // Private data
    // ========================================================================

    std::vector<DataLoader*> loaders_;
    std::atomic<bool> started_{false};
    std::atomic<bool> shutting_down_{false};

    // Two thread pools (BS::thread_pool)
    BS::thread_pool fill_pool_;    // small pool for fill tasks
    BS::thread_pool decode_pool_;  // large pool for decode tasks

    std::thread dispatcher_thread_;

    // Round-robin dispatch state (dispatcher thread only; no synchronization needed)
    int next_dispatch_idx_ = 0;

    // Round-robin consume state (consumer thread only; no synchronization needed)
    int next_consume_idx_ = 0;

    // Dynamic rebalancing state
    int idle_streak_ = 0;   // consecutive cycles with idle fill_pool_ threads
    int busy_streak_ = 0;   // consecutive cycles with full fill_pool_ AND backlog
    static constexpr int kStreakThreshold = 5;    // consecutive observations before rebalancing
    static constexpr int kMinFillThreads = 1;     // never shrink fill_pool_ below this
    static constexpr int kMinDecodeThreads = 1;   // never shrink decode_pool_ below this

    // ========================================================================
    // Private methods
    // ========================================================================

    // Main loop of the dispatcher thread: strict round-robin dispatch,
    // blocking on free slots.
    void dispatcher_loop();

    // Periodically rebalance threads from fill_pool to decode_pool if fill is idle.
    // Called once per full dispatch cycle (after all loaders have been offered a fill).
    void maybe_rebalance();
};
