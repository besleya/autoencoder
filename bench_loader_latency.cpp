#include "ring.h"
#include "gpu_data_loader.h"

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <cstring>

using std::chrono::high_resolution_clock;
using std::chrono::steady_clock;

namespace fs = std::filesystem;

// ============================================================================
// Helpers
// ============================================================================

double ms_between(steady_clock::time_point t0, steady_clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

double percentile(std::vector<double>& data, double p) {
    if (data.empty()) return 0.0;
    size_t idx = (size_t)(p / 100.0 * data.size());
    if (idx >= data.size()) idx = data.size() - 1;
    std::nth_element(data.begin(), data.begin() + idx, data.end());
    return data[idx];
}

// ============================================================================
// File discovery (copied from bench_read_1pz.cpp)
// ============================================================================

std::vector<std::string> glob_1pz_files(const std::string& dir) {
    std::vector<std::string> files;
    try {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".1pz") {
                files.push_back(entry.path().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading directory " << dir << ": " << e.what() << std::endl;
        return files;
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::vector<std::string> read_file_list(const std::string& file_path) {
    std::vector<std::string> files;
    std::ifstream f(file_path);
    if (!f.is_open()) {
        std::cerr << "Cannot open file list: " << file_path << std::endl;
        return files;
    }
    std::string line;
    while (std::getline(f, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (!line.empty() && line[0] != '#') {
            files.push_back(line);
        }
    }
    return files;
}

std::vector<std::string> discover_files(const std::string& paths_arg) {
    fs::path p(paths_arg);
    
    if (fs::is_directory(p)) {
        return glob_1pz_files(paths_arg);
    } else if (fs::is_regular_file(p)) {
        return read_file_list(paths_arg);
    } else {
        std::cerr << "[bench] Error: path is neither a directory nor a regular file: " 
                  << paths_arg << std::endl;
        return {};
    }
}

// ============================================================================
// CLI parsing (hand-rolled)
// ============================================================================

struct Config {
    std::string paths;
    int chunk_size = 64;
    int batch_size = 512;
    int n_batches = 500;
    int decode_threads = 16;
    int ring_depth = 4;
    uint32_t seed = 42;
};

bool parse_args(int argc, char* argv[], Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--paths" && i + 1 < argc) {
            cfg.paths = argv[++i];
        } else if (arg == "--chunk-size" && i + 1 < argc) {
            cfg.chunk_size = std::atoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            cfg.batch_size = std::atoi(argv[++i]);
        } else if (arg == "--n-batches" && i + 1 < argc) {
            cfg.n_batches = std::atoi(argv[++i]);
        } else if (arg == "--decode-threads" && i + 1 < argc) {
            cfg.decode_threads = std::atoi(argv[++i]);
        } else if (arg == "--ring-depth" && i + 1 < argc) {
            cfg.ring_depth = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            cfg.seed = (uint32_t)std::atoi(argv[++i]);
        } else {
            std::cerr << "[bench] Unknown argument: " << arg << std::endl;
            return false;
        }
    }
    
    if (cfg.paths.empty()) {
        std::cerr << "[bench] Error: --paths is required" << std::endl;
        return false;
    }
    
    return true;
}

// ============================================================================
// Main benchmark
// ============================================================================

int main(int argc, char* argv[]) {
    // Print build hint
    std::cerr << "# nvcc -O3 -std=c++17 -x cu -rdc=true --gpu-architecture=sm_90 "
              << "-I<singlet_include> -Xcompiler \"-pthread -fopenmp\" "
              << "bench_loader_latency.cpp ring.cu gpu_data_loader.cu -o bench_loader_latency "
              << "-lcudart -lcublas -lcusparse -lzstd" << std::endl;

    Config cfg;
    if (!parse_args(argc, argv, cfg)) {
        return 1;
    }

    // Discover files
    auto files = discover_files(cfg.paths);
    if (files.empty()) {
        std::cerr << "[bench] Error: no .1pz files found in " << cfg.paths << std::endl;
        return 1;
    }

    // Create Ring
    Ring ring(/*n_lanes=*/1, cfg.ring_depth, AlternationPolicy::SINGLE);

    // Create RNG and DataLoader
    std::mt19937 rng(cfg.seed);
    DataLoader loader(files, cfg.chunk_size, cfg.batch_size, rng, &ring,
                      /*lane_id=*/0, ConcatPolicy::CONCAT_HOST, cfg.decode_threads);

    // Begin epoch
    loader.begin_epoch();

    // Create trainer stream
    cudaStream_t trainer_stream;
    cudaStreamCreate(&trainer_stream);
    if (!trainer_stream) {
        std::cerr << "[bench] Error: cudaStreamCreate failed" << std::endl;
        return 1;
    }

    // Measurement vectors
    std::vector<double> batch_interval_ms;
    std::vector<double> trainer_wait_ms;
    std::vector<double> gpu_ready_ms;

    auto prev_t2 = steady_clock::now();
    int batches_collected = 0;

    for (int i = 0; i < cfg.n_batches; ++i) {
        auto t0 = steady_clock::now();

        SparseBatch b;
        int lane, slot;
        bool ok = ring.acquire_ready(&b, &lane, &slot);
        if (!ok) {
            std::cerr << "[bench] Epoch ended early at batch " << i << std::endl;
            break;
        }

        auto t1 = steady_clock::now();

        cudaStreamWaitEvent(trainer_stream, b.ready_event, 0);
        cudaError_t err = cudaStreamSynchronize(trainer_stream);
        if (err != cudaSuccess) {
            std::cerr << "[bench] Error: cudaStreamSynchronize failed: " 
                      << cudaGetErrorString(err) << std::endl;
            return 1;
        }

        auto t2 = steady_clock::now();

        ring.release_consumed(lane, slot);

        // Collect metrics
        if (i > 0) {
            batch_interval_ms.push_back(ms_between(prev_t2, t2));
        }
        trainer_wait_ms.push_back(ms_between(t0, t1));
        gpu_ready_ms.push_back(ms_between(t1, t2));

        prev_t2 = t2;
        batches_collected++;

        if (b.eof_after) {
            loader.begin_epoch();
        }
    }

    cudaStreamDestroy(trainer_stream);

    // ========================================================================
    // Compute statistics
    // ========================================================================

    // Split warmup (first ring_depth) vs steady-state
    int warmup_count = std::min(cfg.ring_depth, batches_collected);
    int steady_start = warmup_count;

    std::vector<double> warmup_batch_interval_ms;
    std::vector<double> warmup_trainer_wait_ms;
    std::vector<double> steady_batch_interval_ms;
    std::vector<double> steady_trainer_wait_ms;
    std::vector<double> steady_gpu_ready_ms;

    for (int i = 0; i < (int)batch_interval_ms.size(); ++i) {
        if (i < warmup_count - 1) {
            warmup_batch_interval_ms.push_back(batch_interval_ms[i]);
        } else if (i >= warmup_count - 1) {
            steady_batch_interval_ms.push_back(batch_interval_ms[i]);
        }
    }

    for (int i = 0; i < (int)trainer_wait_ms.size(); ++i) {
        if (i < warmup_count) {
            warmup_trainer_wait_ms.push_back(trainer_wait_ms[i]);
        } else {
            steady_trainer_wait_ms.push_back(trainer_wait_ms[i]);
        }
    }

    for (int i = warmup_count; i < (int)gpu_ready_ms.size(); ++i) {
        steady_gpu_ready_ms.push_back(gpu_ready_ms[i]);
    }

    // Compute warmup stats
    double warmup_mean_batch_interval = 0.0;
    double warmup_mean_trainer_wait = 0.0;
    if (!warmup_batch_interval_ms.empty()) {
        warmup_mean_batch_interval = std::accumulate(
            warmup_batch_interval_ms.begin(), warmup_batch_interval_ms.end(), 0.0)
            / warmup_batch_interval_ms.size();
    }
    if (!warmup_trainer_wait_ms.empty()) {
        warmup_mean_trainer_wait = std::accumulate(
            warmup_trainer_wait_ms.begin(), warmup_trainer_wait_ms.end(), 0.0)
            / warmup_trainer_wait_ms.size();
    }

    // Compute steady-state stats
    double steady_mean_batch_interval = 0.0;
    double steady_p50_batch_interval = 0.0;
    double steady_p95_batch_interval = 0.0;
    double steady_max_batch_interval = 0.0;
    double steady_min_trainer_step = 0.0;

    double steady_mean_trainer_wait = 0.0;
    double steady_p50_trainer_wait = 0.0;
    double steady_p95_trainer_wait = 0.0;
    double steady_max_trainer_wait = 0.0;

    double steady_mean_gpu_ready = 0.0;
    double steady_p95_gpu_ready = 0.0;

    if (!steady_batch_interval_ms.empty()) {
        steady_mean_batch_interval = std::accumulate(
            steady_batch_interval_ms.begin(), steady_batch_interval_ms.end(), 0.0)
            / steady_batch_interval_ms.size();

        // Copy for percentile computation (mutates)
        auto bi_copy = steady_batch_interval_ms;
        steady_p50_batch_interval = percentile(bi_copy, 50.0);
        bi_copy = steady_batch_interval_ms;
        steady_p95_batch_interval = percentile(bi_copy, 95.0);
        bi_copy = steady_batch_interval_ms;
        std::sort(bi_copy.begin(), bi_copy.end());
        steady_max_batch_interval = bi_copy.back();

        // min_trainer_step_ms = max(batch_interval)
        steady_min_trainer_step = steady_max_batch_interval;
    }

    if (!steady_trainer_wait_ms.empty()) {
        steady_mean_trainer_wait = std::accumulate(
            steady_trainer_wait_ms.begin(), steady_trainer_wait_ms.end(), 0.0)
            / steady_trainer_wait_ms.size();

        auto tw_copy = steady_trainer_wait_ms;
        steady_p50_trainer_wait = percentile(tw_copy, 50.0);
        tw_copy = steady_trainer_wait_ms;
        steady_p95_trainer_wait = percentile(tw_copy, 95.0);
        tw_copy = steady_trainer_wait_ms;
        std::sort(tw_copy.begin(), tw_copy.end());
        steady_max_trainer_wait = tw_copy.back();
    }

    if (!steady_gpu_ready_ms.empty()) {
        steady_mean_gpu_ready = std::accumulate(
            steady_gpu_ready_ms.begin(), steady_gpu_ready_ms.end(), 0.0)
            / steady_gpu_ready_ms.size();

        auto gr_copy = steady_gpu_ready_ms;
        steady_p95_gpu_ready = percentile(gr_copy, 95.0);
    }

    // Get ring stats
    Ring::Stats ring_stats = ring.stats(0);
    int trainer_waits_count = (int)ring_stats.trainer_waits;
    int producer_waits_count = (int)ring_stats.producer_waits;

    // ========================================================================
    // Output
    // ========================================================================

    std::cout << "# config: paths=" << files.size() << " files, chunk=" << cfg.chunk_size
              << ", batch=" << cfg.batch_size << ", ring=" << cfg.ring_depth
              << ", dec_threads=" << cfg.decode_threads
              << ", n_batches=" << cfg.n_batches << "\n";

    std::cout << "# warmup (first " << warmup_count << " batches)\n";
    printf("warmup_mean_batch_interval_ms: %.1f\n", warmup_mean_batch_interval);
    printf("warmup_mean_trainer_wait_ms:   %.1f\n", warmup_mean_trainer_wait);

    std::cout << "\n# steady-state (batches " << warmup_count << ".." 
              << (batches_collected - 1) << ")\n";
    std::cout << "batches:               " << steady_batch_interval_ms.size() << "\n";
    printf("mean_batch_interval_ms: %.2f\n", steady_mean_batch_interval);
    printf("p50_batch_interval_ms:  %.2f\n", steady_p50_batch_interval);
    printf("p95_batch_interval_ms: %.2f\n", steady_p95_batch_interval);
    printf("max_batch_interval_ms: %.2f\n", steady_max_batch_interval);
    printf("min_trainer_step_ms:   %.2f    <-- HARD FLOOR\n", steady_min_trainer_step);

    printf("\nmean_trainer_wait_ms:   %.3f\n", steady_mean_trainer_wait);
    printf("p50_trainer_wait_ms:    %.3f\n", steady_p50_trainer_wait);
    printf("p95_trainer_wait_ms:    %.3f\n", steady_p95_trainer_wait);
    printf("max_trainer_wait_ms:    %.3f\n", steady_max_trainer_wait);
    std::cout << "trainer_waits_count:    " << trainer_waits_count << " / " 
              << steady_trainer_wait_ms.size() << "\n";

    std::cout << "\nproducer_waits_count:   " << producer_waits_count << " / " 
              << cfg.n_batches << "\n";

    printf("\nmean_gpu_ready_ms:      %.3f\n", steady_mean_gpu_ready);
    printf("p95_gpu_ready_ms:       %.3f\n", steady_p95_gpu_ready);

    // SUMMARY line (critical for SLURM grep)
    printf("\n# SUMMARY: chunk=%d ring=%d dec=%d min_step_ms=%.2f mean_step_ms=%.2f trainer_waits=%d\n",
           cfg.chunk_size, cfg.ring_depth, cfg.decode_threads,
           steady_min_trainer_step, steady_mean_batch_interval, trainer_waits_count);

    return 0;
}
