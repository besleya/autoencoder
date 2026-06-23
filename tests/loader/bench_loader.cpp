// SPDX-License-Identifier: MIT
// bench_loader.cpp — standalone benchmark for DataLoader
//
// Validates the invariant: GPU consumer never stalls waiting for batch from DataLoader.
// Uses multiple DataLoader instances, simulates fixed-duration GPU work, measures
// gap between "consumer ready" and "batch GPU-ready".

#include "gpu_data_loader.h"
#include "gpu_timer.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using std::chrono::steady_clock;
using std::chrono::duration;

// ---------------------------------------------------------------------------
// Command-line options
// ---------------------------------------------------------------------------
struct Options {
    int chunk_size = 4;
    int batch_size = 256;
    int prefetch_depth = 4;
    int num_loaders = 1;
    int total_batches = 500;
    int gpu_work_us = 200;
    std::string policy_str = "concat";
    std::string csv_output;
    std::string file_list;
    std::vector<std::string> files;
    unsigned seed = 42;
};

// ---------------------------------------------------------------------------
// Per-batch statistics
// ---------------------------------------------------------------------------
struct BatchStat {
    int batch_id;
    int loader_id;
    int nnz;
    float wait_ms;
    float gpu_work_ms;
};

// ---------------------------------------------------------------------------
// GPU busy kernel
// ---------------------------------------------------------------------------
static uint64_t g_clock_rate_khz = 0;  // Set once in main

__global__ void busy_kernel(uint64_t target_cycles) {
    // One thread per warp (first thread in warp executes).
    if (threadIdx.x % 32 != 0) return;

    uint64_t start = clock64();
    uint64_t end = start + target_cycles;
    while (clock64() < end) {
        // Busy loop: prevents optimization.
    }
}

void launch_busy_kernel(cudaStream_t stream, int target_us) {
    if (g_clock_rate_khz == 0) {
        int khz;
        cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0);
        g_clock_rate_khz = static_cast<uint64_t>(khz);
    }
    uint64_t cycles_per_us = g_clock_rate_khz / 1000;
    uint64_t target_cycles = cycles_per_us * target_us;
    busy_kernel<<<1, 32, 0, stream>>>(target_cycles);
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------
static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options] [--file-list FILE | --files FILE1 FILE2 ...]\n"
        "Options:\n"
        "  --chunk-size N          chunk_size (default 4)\n"
        "  --batch-size N          batch_size (default 256)\n"
        "  --prefetch-depth N      ring depth (default 4)\n"
        "  --num-loaders N         number of loaders (default 1)\n"
        "  --total-batches N       total batches to consume (default 500)\n"
        "  --gpu-work-us US        GPU work duration in microseconds (default 200)\n"
        "  --policy POLICY         concat or pointer (default concat)\n"
        "  --csv-output FILE       write CSV output\n"
        "  --file-list FILE        read file list from FILE (one path per line)\n"
        "  --files FILE1 FILE2 ... explicit file list\n"
        "  --seed N                random seed (default 42)\n"
        "  --help                  show this help\n",
        prog);
}

static Options parse_args(int argc, char** argv) {
    Options opt;
    static struct option long_opts[] = {
        {"chunk-size",     required_argument, 0, 'c'},
        {"batch-size",     required_argument, 0, 'b'},
        {"prefetch-depth", required_argument, 0, 'd'},
        {"num-loaders",    required_argument, 0, 'n'},
        {"total-batches",  required_argument, 0, 't'},
        {"gpu-work-us",    required_argument, 0, 'w'},
        {"policy",         required_argument, 0, 'p'},
        {"csv-output",     required_argument, 0, 'o'},
        {"file-list",      required_argument, 0, 'l'},
        {"files",          required_argument, 0, 'f'},
        {"seed",           required_argument, 0, 's'},
        {"help",           no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt_idx = 0;
    bool files_from_list = false;
    int c;
    while ((c = getopt_long(argc, argv, "", long_opts, &opt_idx)) != -1) {
        switch (c) {
            case 'c':
                opt.chunk_size = std::stoi(optarg);
                break;
            case 'b':
                opt.batch_size = std::stoi(optarg);
                break;
            case 'd':
                opt.prefetch_depth = std::stoi(optarg);
                break;
            case 'n':
                opt.num_loaders = std::stoi(optarg);
                break;
            case 't':
                opt.total_batches = std::stoi(optarg);
                break;
            case 'w':
                opt.gpu_work_us = std::stoi(optarg);
                break;
            case 'p':
                opt.policy_str = optarg;
                break;
            case 'o':
                opt.csv_output = optarg;
                break;
            case 'l':
                opt.file_list = optarg;
                files_from_list = true;
                break;
            case 'f':
                // Note: --files consumes all remaining argv. This is a simplification;
                // in production, you'd use proper option parsing.
                opt.files.push_back(optarg);
                break;
            case 's':
                opt.seed = static_cast<unsigned>(std::stoul(optarg));
                break;
            case 'h':
                usage(argv[0]);
                std::exit(0);
            case '?':
            default:
                usage(argv[0]);
                std::exit(2);
        }
    }

    // Collect remaining positional arguments as files
    while (optind < argc) {
        opt.files.push_back(argv[optind++]);
    }

    // Load from file list if specified
    if (files_from_list && !opt.file_list.empty()) {
        std::ifstream ifs(opt.file_list);
        if (!ifs.is_open()) {
            std::fprintf(stderr, "ERROR: cannot open file list '%s'\n", opt.file_list.c_str());
            std::exit(2);
        }
        std::string line;
        while (std::getline(ifs, line)) {
            // Skip empty lines
            if (!line.empty() && line[0] != '#') {
                opt.files.push_back(line);
            }
        }
    }

    // Validation
    if (opt.files.empty()) {
        std::fprintf(stderr, "ERROR: no input files provided\n");
        std::exit(2);
    }
    if (opt.chunk_size <= 0 || opt.batch_size <= 0 || opt.prefetch_depth <= 0 ||
        opt.num_loaders <= 0 || opt.total_batches <= 0) {
        std::fprintf(stderr, "ERROR: all numeric parameters must be > 0\n");
        std::exit(2);
    }
    if (opt.policy_str != "concat" && opt.policy_str != "pointer") {
        std::fprintf(stderr, "ERROR: policy must be 'concat' or 'pointer'\n");
        std::exit(2);
    }

    return opt;
}

// ---------------------------------------------------------------------------
// Percentile calculation
// ---------------------------------------------------------------------------
static float percentile(std::vector<float>& values, float p) {
    if (values.empty()) return 0.0f;
    std::sort(values.begin(), values.end());
    size_t idx = static_cast<size_t>((p / 100.0f) * (values.size() - 1));
    return values[idx];
}

// ---------------------------------------------------------------------------
// Main benchmark
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        // Print banner
        std::printf("=== bench_loader ===\n");
        std::printf("Config: num_loaders=%d chunk_size=%d batch_size=%d "
                    "prefetch_depth=%d policy=%s gpu_work_us=%d\n",
                    opt.num_loaders, opt.chunk_size, opt.batch_size,
                    opt.prefetch_depth, opt.policy_str.c_str(), opt.gpu_work_us);
        std::printf("Files: %zu total\n", opt.files.size());

        // Determine policy
        ConcatPolicy policy = (opt.policy_str == "concat")
            ? ConcatPolicy::CONCAT_HOST
            : ConcatPolicy::POINTER_LIST;

        // Split file list among loaders
        std::vector<std::vector<std::string>> loader_files(opt.num_loaders);
        for (size_t i = 0; i < opt.files.size(); ++i) {
            int loader_id = i % opt.num_loaders;
            loader_files[loader_id].push_back(opt.files[i]);
        }

        std::printf("Files per loader:");
        for (int i = 0; i < opt.num_loaders; ++i) {
            std::printf(" loader_%d=%zu", i, loader_files[i].size());
        }
        std::printf("\n");

        if (opt.files.empty()) {
            std::fprintf(stderr, "ERROR: file list is empty\n");
            std::exit(2);
        }

        // Create RNG (consumed only in begin_epoch, called sequentially)
        std::mt19937 rng(opt.seed);

        // Create DataLoader instances
        std::vector<std::unique_ptr<DataLoader>> loaders;
        for (int i = 0; i < opt.num_loaders; ++i) {
            if (loader_files[i].empty()) {
                std::fprintf(stderr, "ERROR: loader %d has no files\n", i);
                std::exit(2);
            }
            loaders.push_back(std::make_unique<DataLoader>(
                loader_files[i],
                opt.chunk_size,
                opt.batch_size,
                rng,
                policy,
                opt.prefetch_depth,
                opt.num_loaders));
        }

        // Create trainer stream
        cudaStream_t trainer_stream;
        cudaStreamCreateWithFlags(&trainer_stream, cudaStreamNonBlocking);

        // Create GPU events for work timing
        cudaEvent_t work_start_evt, work_stop_evt;
        cudaEventCreate(&work_start_evt);
        cudaEventCreate(&work_stop_evt);

        // Call begin_epoch on all loaders sequentially
        for (int i = 0; i < opt.num_loaders; ++i) {
            loaders[i]->begin_epoch();
        }

        // Main consumer loop
        std::vector<BatchStat> batch_stats;
        std::vector<int> loader_epochs(opt.num_loaders, 1);
        int total_batches_consumed = 0;
        auto t_loop_start = steady_clock::now();

        for (int batch_id = 0; batch_id < opt.total_batches; ++batch_id) {
            int loader_id = batch_id % opt.num_loaders;
            auto t_request = steady_clock::now();

            SparseBatch batch;
            if (!loaders[loader_id]->next_batch(&batch)) {
                // EOF: start a new epoch and continue
                loaders[loader_id]->begin_epoch();
                loader_epochs[loader_id]++;
                if (!loaders[loader_id]->next_batch(&batch)) {
                    // Still EOF after new epoch? Stop.
                    std::printf("loader %d EOF even after new epoch; stopping\n", loader_id);
                    break;
                }
            }

            auto t_got = steady_clock::now();
            float wait_ms = duration<float, std::milli>(t_got - t_request).count();

            // Fence on batch ready event
            cudaStreamWaitEvent(trainer_stream, batch.ready_event, 0);

            // Simulate GPU work
            cudaEventRecord(work_start_evt, trainer_stream);
            launch_busy_kernel(trainer_stream, opt.gpu_work_us);
            cudaEventRecord(work_stop_evt, trainer_stream);

            // Measure GPU work time
            float gpu_work_ms = 0.0f;
            cudaEventElapsedTime(&gpu_work_ms, work_start_evt, work_stop_evt);

            // Log soft failures (wait time exceeds 1.0 ms)
            if (wait_ms > 1.0f) {
                std::fprintf(stderr,
                    "SOFT FAIL: batch %d loader %d wait_ms=%.3f > 1.0 ms\n",
                    batch_id, loader_id, wait_ms);
            }

            batch_stats.push_back({batch_id, loader_id, batch.nnz, wait_ms, gpu_work_ms});
            total_batches_consumed++;
        }

        // Synchronize trainer stream at teardown
        cudaStreamSynchronize(trainer_stream);

        auto t_loop_end = steady_clock::now();
        double loop_wall_s = duration<double>(t_loop_end - t_loop_start).count();

        // Compute summary statistics
        std::vector<float> wait_times, gpu_work_times;
        int64_t total_nnz = 0;
        for (const auto& stat : batch_stats) {
            wait_times.push_back(stat.wait_ms);
            gpu_work_times.push_back(stat.gpu_work_ms);
            total_nnz += stat.nnz;
        }

        float mean_wait = wait_times.empty() ? 0.0f
            : std::accumulate(wait_times.begin(), wait_times.end(), 0.0f) / wait_times.size();
        float p50_wait = percentile(wait_times, 50.0f);
        float p99_wait = percentile(wait_times, 99.0f);
        float max_wait = wait_times.empty() ? 0.0f
            : *std::max_element(wait_times.begin(), wait_times.end());

        float mean_gpu_work = gpu_work_times.empty() ? 0.0f
            : std::accumulate(gpu_work_times.begin(), gpu_work_times.end(), 0.0f) / gpu_work_times.size();
        float p50_gpu_work = percentile(gpu_work_times, 50.0f);
        float max_gpu_work = gpu_work_times.empty() ? 0.0f
            : *std::max_element(gpu_work_times.begin(), gpu_work_times.end());

        // Print summary
        std::printf("\n=== bench_loader summary ===\n");
        std::printf("Config: num_loaders=%d chunk_size=%d batch_size=%d ring_depth=%d "
                    "policy=%s gpu_work_us=%d\n",
                    opt.num_loaders, opt.chunk_size, opt.batch_size, opt.prefetch_depth,
                    opt.policy_str.c_str(), opt.gpu_work_us);
        
        std::printf("Files per loader:");
        for (int i = 0; i < opt.num_loaders; ++i) {
            std::printf(" %d=%zu", i, loader_files[i].size());
        }
        std::printf("\n");

        std::printf("Total batches consumed: %d (wall: %.2f s; throughput: %.1f batches/s)\n",
                    total_batches_consumed, loop_wall_s,
                    total_batches_consumed / loop_wall_s);

        std::printf("\nWait time (consumer blocked in next_batch), milliseconds:\n");
        std::printf("  mean=%.3f  p50=%.3f  p99=%.3f  max=%.3f\n",
                    mean_wait, p50_wait, p99_wait, max_wait);

        if (max_wait <= 1.0f) {
            std::printf("  *** PASS *** : max wait <= 1.0 ms (GPU never stalled significantly)\n");
        } else {
            std::printf("  *** FAIL *** : max wait = %.3f ms exceeds 1.0 ms threshold\n", max_wait);
        }

        // Per-loader breakdown
        std::printf("\nPer-loader breakdown:\n");
        for (int i = 0; i < opt.num_loaders; ++i) {
            std::vector<float> loader_waits;
            int loader_batch_count = 0;
            for (const auto& stat : batch_stats) {
                if (stat.loader_id == i) {
                    loader_waits.push_back(stat.wait_ms);
                    loader_batch_count++;
                }
            }
            if (!loader_waits.empty()) {
                float mean_w = std::accumulate(loader_waits.begin(), loader_waits.end(), 0.0f) / loader_waits.size();
                float p99_w = percentile(loader_waits, 99.0f);
                std::printf("  loader %d: epochs=%d  batches=%d  mean_wait=%.3f ms  p99_wait=%.3f ms\n",
                            i, loader_epochs[i], loader_batch_count, mean_w, p99_w);
            }
        }

        // GPU work sanity check
        std::printf("\nGPU work time (sanity check, should be ~%d us):\n", opt.gpu_work_us);
        std::printf("  mean=%.0f us  p50=%.0f us  max=%.0f us\n",
                    mean_gpu_work * 1000.0f,
                    p50_gpu_work * 1000.0f,
                    max_gpu_work * 1000.0f);

        // Total NNZ throughput
        double nnz_per_sec = total_nnz / loop_wall_s;
        std::printf("\nTotal nnz consumed: %ld  (throughput: %.2f M nnz/s)\n",
                    total_nnz, nnz_per_sec / 1e6);

        // CSV output if requested
        if (!opt.csv_output.empty()) {
            std::ofstream csv(opt.csv_output);
            csv << "batch_id,loader_id,nnz,wait_ms,gpu_work_ms\n";
            for (const auto& stat : batch_stats) {
                csv << stat.batch_id << ","
                    << stat.loader_id << ","
                    << stat.nnz << ","
                    << stat.wait_ms << ","
                    << stat.gpu_work_ms << "\n";
            }
            std::printf("\nCSV output written to: %s\n", opt.csv_output.c_str());
        }

        // Cleanup
        cudaEventDestroy(work_start_evt);
        cudaEventDestroy(work_stop_evt);
        cudaStreamDestroy(trainer_stream);

        return 0;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr, "ERROR: unknown exception\n");
        return 1;
    }
}
