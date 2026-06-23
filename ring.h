// SPDX-License-Identifier: MIT
// ring.h — Ring buffer for multi-lane batch scheduling in sparse GPU DataLoader.
//
// A Ring multiplexes multiple data sources ("lanes") via a configurable
// alternation policy. Each lane has its own producer (e.g., DataLoader's T_BB)
// and a shared consumer (trainer). Tracks per-lane statistics (trainer stalls,
// producer stalls) and provides state machine for batch lifecycle.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>

// ============================================================================
// SparseBatch — public contract for one mini-batch
// ============================================================================

// A single mini-batch: sparse CSC of shape (m × B) resident on device.
// All pointers are device-resident. The event is signaled after H2D + lognorm
// are enqueued (GPU signals when complete). Trainer pattern:
//   SparseBatch b;
//   int lane, slot;
//   ring.acquire_ready(&b, &lane, &slot);
//   cudaStreamWaitEvent(trainer_stream, b.ready_event, 0);
//   net.forward(b, trainer_stream);
//   ...training...
//   ring.release_consumed(lane, slot);
//
// Lifetime: pointers and event remain valid until ring_depth-th subsequent
// acquire_ready() call (default ring_depth=4). Trainer holds at most 1 batch
// at a time per lane.
struct SparseBatch {
    int m;                  // features (rows), same as dataset
    int B;                  // batch size (cols)
    int nnz;                // nonzeros in this batch
    const int32_t* d_col_ptr;   // device, length B+1
    const int32_t* d_row_idx;   // device, length nnz
    const float*   d_values;    // device, length nnz
    cudaEvent_t    ready_event; // signaled after H2D + lognorm; owned by slot
    bool           chunk_end;   // true on the last batch of a chunk
};

// ============================================================================
// AlternationPolicy — multi-lane scheduling mode
// ============================================================================

enum class AlternationPolicy {
    SINGLE,         // exactly one lane (n_lanes must be 1); no scheduling logic
    ROUND_ROBIN,    // cycle through lanes in order; skip lanes with no READY slot
    WEIGHTED,       // weighted round-robin based on per-lane weights
    CALLER_CHOOSES  // trainer explicitly selects a lane via acquire_ready_from
};

// ============================================================================
// Ring — multi-lane batch buffer with state management and statistics
// ============================================================================

class Ring {
public:
    // Per-lane statistics. Updated atomically during slot transitions.
    struct Stats {
        uint64_t trainer_waits = 0;        // # times trainer blocked in acquire_ready
        uint64_t trainer_wait_ns_total = 0;  // total nanoseconds trainer spent blocked
        uint64_t producer_waits = 0;       // # times producer blocked on acquire_free
        uint64_t producer_wait_ns_total = 0; // total nanoseconds producer spent blocked
        uint64_t batches_published = 0;    // # batches transitioned to READY
        uint64_t batches_consumed = 0;     // # batches transitioned to CONSUMED
    };

    // Constructor. Initializes n_lanes lanes, each with ring_depth slots.
    // For SINGLE policy, n_lanes must be 1.
    // Allocates CUDA events and device/host buffers; starts with all slots FREE.
    Ring(int n_lanes, int ring_depth,
         AlternationPolicy policy = AlternationPolicy::SINGLE);

    ~Ring();

    // Non-copyable, non-movable.
    Ring(const Ring&) = delete;
    Ring& operator=(const Ring&) = delete;

    // ========================================================================
    // Producer API (called by data loader, e.g., T_BB thread per lane)
    // ========================================================================

    // Block until a FREE slot is available in `lane`. Returns slot index (0..ring_depth-1).
    // Updates slot state: FREE → BUILDING. Returns -1 if stopped_ became true while waiting.
    // Increments producer_waits and producer_wait_ns_total if blocking was necessary.
    int acquire_free(int lane);

    // Thin view into a slot's buffers for populating during BUILDING state.
    struct SlotView {
        // Pinned host buffers — current pointers and capacities.
        // These are SNAPSHOTS taken at the moment slot_view() was called.
        // If ensure_capacity() is called and triggers a realloc, the snapshot
        // becomes stale — the caller MUST call slot_view() again to refresh.
        int32_t* h_col_ptr;
        int32_t* h_row_idx;
        float*   h_values;
        size_t   h_col_capacity;
        size_t   h_row_capacity;
        size_t   h_val_capacity;

        // Device buffers — same snapshot semantics as the host pointers above.
        int32_t* d_col_ptr;
        int32_t* d_row_idx;
        float*   d_values;
        size_t   d_col_capacity;
        size_t   d_row_capacity;
        size_t   d_val_capacity;

        cudaEvent_t ready_event;
    };

    // Get a mutable view into slot's buffers. Called between acquire_free and
    // publish_ready by the producer to fill host buffers and prepare enqueuing.
    // After call, if ensure_capacity was needed, call slot_view again to refresh pointers.
    SlotView slot_view(int lane, int slot_idx);

    // Grow (or leave unchanged) all pinned and device buffers in this slot to
    // at least the specified capacities. Reallocates and changes pointers if needed.
    // NOT thread-safe with other Ring operations on the same slot; the slot must
    // be in BUILDING state (only the producer for this lane touches it).
    // Calls cudaFreeHost/cudaMallocHost and cudaFree/cudaMalloc directly.
    void ensure_capacity(int lane, int slot_idx,
                         size_t col_ptr_n, size_t row_idx_n, size_t val_n);

    // Mark slot as READY. Updates slot state: BUILDING → READY. T_BB should have
    // already enqueued H2D transfers and lognorm kernel, and called
    // cudaEventRecord(slot.ready_event, loader_stream) before calling this.
    // Parameters B, nnz, chunk_end are copied into the slot's metadata.
    // Notifies all consumer-side waiters in the condition variable.
    void publish_ready(int lane, int slot_idx, int B, int nnz, bool chunk_end);

    // ========================================================================
    // Consumer API (called by trainer thread)
    // ========================================================================

    // Block until SOME lane (per policy_) has a READY slot, OR shutdown is called.
    // On success, fills *out with device pointers + event, sets *out_lane to the
    // lane id, and sets *out_slot to the slot index; returns true.
    // On shutdown, returns false.
    // Updates slot state: READY → CONSUMED.
    // Increments trainer_waits + trainer_wait_ns_total if blocking was needed.
    bool acquire_ready(SparseBatch* out, int* out_lane, int* out_slot);

    // CALLER_CHOOSES variant: pull only from `lane`. Blocks waiting on that lane
    // specifically until a READY slot is available OR shutdown is called.
    // On success, fills *out and sets *out_slot; returns true.
    // Returns false on shutdown.
    bool acquire_ready_from(int lane, SparseBatch* out, int* out_slot);

    // Release the slot in `lane` so it can recycle to FREE state.
    // Updates slot state: CONSUMED → FREE. Notifies producer-side waiters.
    void release_consumed(int lane, int slot_idx);

    // ========================================================================
    // Epoch lifecycle
    // ========================================================================

    // Reset per-lane state for a new pass. Called when a lane begins a new pass.
    // Does NOT reset stats. This function may be a no-op if no per-lane state needs resetting.
    void begin_epoch(int lane);

    // Get the current pass counter for a lane. Lanes auto-increment this counter
    // each time they wrap their file list and begin a new pass.
    int lane_pass(int lane_id) const;

    // Increment the pass counter for a lane. Called by the data loader when
    // it reshuffles and begins a new pass over its file list.
    void increment_lane_pass(int lane_id);

    // Check if shutdown has been called.
    bool is_shutdown() const;

    // ========================================================================
    // Statistics and configuration
    // ========================================================================

    // Get per-lane statistics snapshot.
    Stats stats(int lane) const;

    // Get total statistics (sum across all lanes).
    Stats stats_total() const;

    // Reset all per-lane stats to zero. Called optionally between epochs.
    void reset_stats();

    // Set the weight for `lane` (used by WEIGHTED policy). Default is 1.0.
    // Weights need not sum to 1; they are normalized internally.
    // For WEIGHTED policy, you should call this for all lanes before
    // begin_epoch() so the schedule is correct.
    void set_weight(int lane, double w);

    // Set the feature count m (number of rows). Called once by the data loader
    // before publishing the first batch. After set, m() returns this value.
    void set_m(int m);

    // Get feature count (peeked from first decoded file by data loader).
    int m() const;

    // Configuration accessors.
    int n_lanes() const;
    int ring_depth() const;

    // ========================================================================
    // Shutdown (for destructor and forced cleanup)
    // ========================================================================

    // Set stopped_ flag and notify all waiters. All acquire_free and acquire_ready
    // calls will return failure (-1 or false) after shutdown. Called from destructor.
    void shutdown();

private:
    // Internal structures: Slot and Lane definitions.
    
public:
    struct Slot {
        // Pinned host buffers
        int32_t* h_col_ptr = nullptr;
        int32_t* h_row_idx = nullptr;
        float*   h_values  = nullptr;
        size_t   h_col_capacity = 0;   // in int32_t
        size_t   h_row_capacity = 0;
        size_t   h_val_capacity = 0;

        // Device buffers
        int32_t* d_col_ptr = nullptr;
        int32_t* d_row_idx = nullptr;
        float*   d_values  = nullptr;
        size_t   d_col_capacity = 0;
        size_t   d_row_capacity = 0;
        size_t   d_val_capacity = 0;

        // Event (created at construction)
        cudaEvent_t ready_event = nullptr;

        // Current batch metadata
        int B = 0;
        int nnz = 0;
        bool chunk_end = false;

        // State machine: FREE, BUILDING, READY, CONSUMED, FREE, ...
        enum class State { FREE, BUILDING, READY, CONSUMED };
        State state = State::FREE;

        // Cleanup
        void destroy() {
            if (h_col_ptr) { cudaFreeHost(h_col_ptr); h_col_ptr = nullptr; }
            if (h_row_idx) { cudaFreeHost(h_row_idx); h_row_idx = nullptr; }
            if (h_values) { cudaFreeHost(h_values); h_values = nullptr; }
            if (d_col_ptr) { cudaFree(d_col_ptr); d_col_ptr = nullptr; }
            if (d_row_idx) { cudaFree(d_row_idx); d_row_idx = nullptr; }
            if (d_values) { cudaFree(d_values); d_values = nullptr; }
            if (ready_event) { cudaEventDestroy(ready_event); ready_event = nullptr; }
        }
    };

    struct Lane {
        std::vector<Slot> slots;
        int free_count = 0;
        int ready_count = 0;

        // Pass counter: incremented each time this lane wraps its file list
        std::atomic<int> pass_counter{0};

        // Weight for WEIGHTED policy
        double weight = 1.0;

        // Rotating slot selection for READY slot acquisition
        int next_slot_idx = 0;  // next slot index to try in this lane

        // Statistics
        Stats stats;

        // Explicitly enable move semantics, disable copy
        Lane() = default;
        Lane(const Lane&) = delete;
        Lane& operator=(const Lane&) = delete;
        
        Lane(Lane&& other) noexcept
            : slots(std::move(other.slots)),
              free_count(other.free_count),
              ready_count(other.ready_count),
              pass_counter(other.pass_counter.load()),
              weight(other.weight),
              next_slot_idx(other.next_slot_idx),
              stats(std::move(other.stats)) {}
        
        Lane& operator=(Lane&& other) noexcept {
            if (this != &other) {
                slots = std::move(other.slots);
                free_count = other.free_count;
                ready_count = other.ready_count;
                pass_counter.store(other.pass_counter.load());
                weight = other.weight;
                next_slot_idx = other.next_slot_idx;
                stats = std::move(other.stats);
            }
            return *this;
        }
    };

private:

    std::vector<Lane> lanes_;
    int n_lanes_ = 0;
    int ring_depth_ = 0;
    AlternationPolicy policy_;
    int next_lane_rr_ = 0;  // round-robin state
    std::vector<int> weighted_schedule_;  // precomputed schedule for WEIGHTED
    int m_ = 0;

    mutable std::mutex mtx_;
    mutable std::condition_variable cv_;
    std::atomic<bool> stopped_{false};
};
