// SPDX-License-Identifier: MIT
// gpu_timer.h — Fine-grained CUDA event-based timer with ring buffer pool.
// Zero explicit syncs during normal operation; only cudaEventSynchronize on
// the LAST stop event inside flush_and_accumulate() at epoch end.

#pragma once

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// Per-section accumulator with event ring buffer
struct TimerSection {
    std::vector<cudaEvent_t> start_events;  // Ring buffer of start events
    std::vector<cudaEvent_t> stop_events;   // Ring buffer of stop events
    size_t next_pair_idx = 0;                // Index for next event pair
    int call_count = 0;                      // Number of times this section was recorded (epoch)
    float total_ms = 0.0f;                   // Accumulated time (epoch)
    int grand_call_count = 0;                // Total calls across all epochs
    float grand_total_ms = 0.0f;             // Total time across all epochs
};

// GpuTimerPool: maintains event pairs for each named section, with async recording
// and epoch-end sync/accumulate pattern.
class GpuTimerPool {
public:
    // Ring buffer size per section (default 4096 event pairs)
    static constexpr size_t RING_SIZE = 4096;

    GpuTimerPool() = default;
    
    ~GpuTimerPool() {
        // Cleanup all events
        for (auto& [name, section] : sections_) {
            for (auto evt : section.start_events) {
                if (evt) cudaEventDestroy(evt);
            }
            for (auto evt : section.stop_events) {
                if (evt) cudaEventDestroy(evt);
            }
        }
    }

    // Non-copyable, movable
    GpuTimerPool(const GpuTimerPool&) = delete;
    GpuTimerPool& operator=(const GpuTimerPool&) = delete;
    GpuTimerPool(GpuTimerPool&&) = default;
    GpuTimerPool& operator=(GpuTimerPool&&) = default;

    // Record a start event on the given stream (async, no sync)
    void record_start(const std::string& name, cudaStream_t stream = 0) {
        auto& section = sections_[name];
        if (section.start_events.empty()) {
            // Lazy initialization of event rings
            for (size_t i = 0; i < RING_SIZE; ++i) {
                cudaEvent_t evt;
                cudaEventCreate(&evt);
                section.start_events.push_back(evt);
                cudaEventCreate(&evt);
                section.stop_events.push_back(evt);
            }
        }
        size_t idx = section.next_pair_idx % RING_SIZE;
        cudaEventRecord(section.start_events[idx], stream);
    }

    // Record a stop event on the given stream (async, no sync)
    void record_stop(const std::string& name, cudaStream_t stream = 0) {
        auto& section = sections_[name];
        if (section.start_events.empty()) {
            // Error: stop called before start
            fprintf(stderr, "GpuTimerPool::record_stop: no start event for '%s'\n", name.c_str());
            return;
        }
        size_t idx = section.next_pair_idx % RING_SIZE;
        cudaEventRecord(section.stop_events[idx], stream);
        section.next_pair_idx++;
    }

    // Flush and accumulate: sync on LAST stop event of each section,
    // sum elapsed times, accumulate to grand total, then reset per-epoch accumulators.
    void flush_and_accumulate() {
        for (auto& [name, section] : sections_) {
            if (section.next_pair_idx == 0) {
                // No events recorded for this section in this epoch
                continue;
            }

            // Sync on the LAST stop event to ensure all events are complete
            size_t last_idx = (section.next_pair_idx - 1) % RING_SIZE;
            cudaEventSynchronize(section.stop_events[last_idx]);

            // Accumulate elapsed times across all recorded pairs
            float total = 0.0f;
            for (size_t i = 0; i < section.next_pair_idx; ++i) {
                size_t idx = i % RING_SIZE;
                float elapsed_ms = 0.0f;
                cudaEventElapsedTime(&elapsed_ms, 
                                     section.start_events[idx],
                                     section.stop_events[idx]);
                total += elapsed_ms;
            }

            section.total_ms += total;
            section.call_count += section.next_pair_idx;
            
            // Accumulate to grand totals (across all epochs)
            section.grand_total_ms += total;
            section.grand_call_count += section.next_pair_idx;
            
            section.next_pair_idx = 0;  // Reset for next epoch
        }
    }

    // Report: print a table of sections and their accumulated times.
    // If use_grand is true, report from grand_total accumulators (all epochs).
    // Otherwise, report from per-epoch accumulators.
    void report(std::ostream& os, const std::string& label, bool use_grand = false) {
        // Collect non-zero sections
        std::vector<std::pair<std::string, TimerSection>> sorted;
        for (auto& [name, section] : sections_) {
            float check = use_grand ? section.grand_total_ms : section.total_ms;
            if (check > 0.0f) {
                sorted.push_back({name, section});
            }
        }

        if (sorted.empty()) {
            return;  // Nothing to report
        }

        // Sort by name for consistent output
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        os << "\n=== GPU Timings (" << label << ") ===\n";
        os << std::left << std::setw(50) << "Section"
           << std::right << std::setw(12) << "Total (ms)"
           << std::setw(12) << "Calls"
           << std::setw(12) << "Mean (ms)" << "\n";
        os << std::string(86, '-') << "\n";

        for (auto& [name, section] : sorted) {
            float total = use_grand ? section.grand_total_ms : section.total_ms;
            int calls = use_grand ? section.grand_call_count : section.call_count;
            float mean_ms = (calls > 0) ? total / calls : 0.0f;
            os << std::left << std::setw(50) << name
               << std::right << std::setw(12) << std::fixed << std::setprecision(3) << total
               << std::setw(12) << calls
               << std::setw(12) << std::fixed << std::setprecision(3) << mean_ms << "\n";
        }
        os << std::flush;
    }

    // Reset per-epoch accumulators (called after reporting an epoch).
    // Grand totals are preserved.
    void reset_epoch() {
        for (auto& [name, section] : sections_) {
            section.total_ms = 0.0f;
            section.call_count = 0;
            section.next_pair_idx = 0;
        }
    }

    // Reset all accumulated times including grand totals (but keep event pairs allocated)
    void reset_all() {
        for (auto& [name, section] : sections_) {
            section.total_ms = 0.0f;
            section.call_count = 0;
            section.grand_total_ms = 0.0f;
            section.grand_call_count = 0;
            section.next_pair_idx = 0;
        }
    }

private:
    std::map<std::string, TimerSection> sections_;
};

// Singleton accessor: returns a file-scope static pool
inline GpuTimerPool& gpu_timers() {
    static GpuTimerPool pool;
    return pool;
}

// RAII scoped timer: records start in ctor, stop in dtor
class GpuScopedTimer {
public:
    GpuScopedTimer(const std::string& name, cudaStream_t stream = 0)
        : name_(name), stream_(stream) {
        gpu_timers().record_start(name_, stream_);
    }

    ~GpuScopedTimer() {
        gpu_timers().record_stop(name_, stream_);
    }

    // Non-copyable, non-movable
    GpuScopedTimer(const GpuScopedTimer&) = delete;
    GpuScopedTimer& operator=(const GpuScopedTimer&) = delete;

private:
    std::string name_;
    cudaStream_t stream_;
};
