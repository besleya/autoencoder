// SPDX-License-Identifier: MIT
// ring.h — Dispatcher and thread pool for strict round-robin batch scheduling.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>


// Forward declarations
class DataLoader;
class Batch;

// ============================================================================
// SparseBatch — public contract for one mini-batch
// ============================================================================

// A single mini-batch: sparse CSC of shape (m × B) resident on device.
// All pointers are device-resident. The event is signaled after H2D + lognorm
// are enqueued (GPU signals when complete). Trainer pattern:
//   SparseBatch b = batch->sparse_view();
//   cudaStreamWaitEvent(trainer_stream, b.ready_event, 0);
//   net.forward(b, trainer_stream);
//   ...training...
//   // Batch dtor marks slot FREE
//
struct SparseBatch {
    int m;                      // features (rows), same as dataset
    int B;                      // batch size (cols)
    int nnz;                    // nonzeros in this batch
    const int32_t* d_col_ptr;   // device, length B+1
    const int32_t* d_row_idx;   // device, length nnz
    const float*   d_values;    // device, length nnz
    cudaEvent_t    ready_event; // signaled after H2D + lognorm; owned by slot
    bool           chunk_end;   // true on the last batch of a chunk
};

// ============================================================================
// Ring — dispatcher and thread pool for strict round-robin batch scheduling
// ============================================================================

class Ring {
public:
    // Constructor: creates a bounded CPU thread pool of n_worker_threads workers.
    // Empty loader list; loaders are added via add_loader() before start().
    explicit Ring(int n_worker_threads);

    // Destructor: calls shutdown() if not already shut down.
    ~Ring();

    // Non-copyable, non-movable.
    Ring(const Ring&) = delete;
    Ring& operator=(const Ring&) = delete;
    Ring(Ring&&) = delete;
    Ring& operator=(Ring&&) = delete;

    // ========================================================================
    // Loader registration (must be called before start())
    // ========================================================================

    // Append a DataLoader to the round-robin cycle. Order of registration
    // determines round-robin order. Loader must outlive Ring.
    void add_loader(DataLoader* loader);

    // ========================================================================
    // Lifecycle
    // ========================================================================

    // Launch the dispatcher thread. Requires at least one registered loader.
    // Must be called exactly once.
    void start();

    // Signal dispatcher and workers to exit, drain the pool, join all threads.
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
    std::atomic<bool> shut_down_complete_{false};

    std::thread dispatcher_thread_;
    std::vector<std::thread> workers_;

    // Thread pool task queue
    std::mutex pool_mtx_;
    std::condition_variable pool_cv_;
    std::queue<std::function<void()>> task_queue_;

    // Round-robin dispatch state (dispatcher thread only; no synchronization needed)
    int next_dispatch_idx_ = 0;

    // Round-robin consume state (consumer thread only; no synchronization needed)
    int next_consume_idx_ = 0;

    // ========================================================================
    // Private methods
    // ========================================================================

    // Main loop of the dispatcher thread: strict round-robin dispatch,
    // blocking on free slots.
    void dispatcher_loop();

    // Main loop of pool worker threads: dequeue and execute tasks.
    void pool_worker_loop();

    // Enqueue a task for execution by a pool worker.
    void submit_task(std::function<void()> task);

    // Helper: block (with polling) until the given loader has a free slot.
    // Returns nullptr if shutting_down becomes true while waiting.
    // STRICT ROTATION: never skips to another loader.
    Slot* wait_for_free_slot(DataLoader* loader);
};
