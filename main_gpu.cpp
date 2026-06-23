// SPDX-License-Identifier: MIT
// GPU driver: parse CLI, load dataset via DataLoader, train a GpuAutoencoder.

#include "gpu_autoencoder.h"
#include "gpu_data_loader.h"
#include "gpu_timer.h"
#include "ring.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

// ============================================================================
// CUDA error checking
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
struct Options {
    int   epochs     = 10;
    int   batch_size = 512;
    float lr         = 1e-3f;
    std::vector<int> hidden = {512, 128, 32};
    unsigned seed    = 42;
    int   chunk_size = 1;
    std::vector<std::string> files;
};

// Parse a comma-separated list of integers, e.g. "512,128,32".
static std::vector<int> parse_csv_ints(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::stoi(tok));
    }
    return out;
}

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " [--epochs N] [--batch N] [--lr F] [--hidden d1,d2,...]"
                 " [--chunk N] [--seed N] <file1.1pz> [file2.1pz ...]\n";
}

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return std::string(argv[++i]);
        };
        if      (a == "--epochs") opt.epochs     = std::stoi(need("--epochs"));
        else if (a == "--batch")  opt.batch_size = std::stoi(need("--batch"));
        else if (a == "--lr")     opt.lr         = std::stof(need("--lr"));
        else if (a == "--hidden") opt.hidden     = parse_csv_ints(need("--hidden"));
        else if (a == "--chunk")  opt.chunk_size = std::stoi(need("--chunk"));
        else if (a == "--seed")   opt.seed       = static_cast<unsigned>(
                                                       std::stoul(need("--seed")));
        else if (a == "-h" || a == "--help") { usage(argv[0]); std::exit(0); }
        else if (!a.empty() && a[0] == '-') {
            throw std::runtime_error("unknown option: " + a);
        } else {
            opt.files.push_back(a);
        }
    }
    if (opt.files.empty()) {
        usage(argv[0]);
        throw std::runtime_error("no input files provided");
    }
    if (opt.hidden.empty()) {
        throw std::runtime_error("--hidden must list at least one dimension");
    }
    if (opt.batch_size <= 0) {
        throw std::runtime_error("--batch must be > 0");
    }
    if (opt.chunk_size <= 0) {
        throw std::runtime_error("--chunk must be > 0");
    }
    return opt;
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

namespace {
    static double ms_since(std::chrono::steady_clock::time_point t0) {
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    try {
        auto t_start = std::chrono::steady_clock::now();
        Options opt = parse_args(argc, argv);
        std::cout << "Argument parsing: " << ms_since(t_start) << " ms" << std::endl;

        // Build full layer list: [m] + hidden + reverse(hidden[:-1]) + [m]
        // We'll get m from the loader after construction.
        std::vector<int> layer_dims;
        // Placeholder: will be filled after loader is constructed
        
        // Construct RNG and initialize the autoencoder BEFORE the first epoch.
        // This ensures RNG is consumed in the same order as the CPU main.cpp.
        nvtxRangePushA("main:init");
        auto t_init = std::chrono::steady_clock::now();
        std::mt19937 rng(opt.seed);
        
        // Create the data loader. RNG is passed by reference; from now on,
        // begin_epoch() will consume RNG state for shuffling.
        auto t_loader = std::chrono::steady_clock::now();
        std::cout << "Constructing GPU data loader..." << std::endl;
        Ring ring(1, 4, AlternationPolicy::SINGLE);
        DataLoader loader(opt.files, opt.chunk_size, opt.batch_size, rng, &ring, /*lane_id=*/0);
        std::cout << "Data loader construction: " << ms_since(t_loader) << " ms" << std::endl;
        
        // Peek the feature count from the loader
        const int m = loader.m();
        if (m <= 0) {
            throw std::runtime_error("loader reports m <= 0");
        }

        // Build full layer list: [m] + hidden + reverse(hidden[:-1]) + [m]
        layer_dims.push_back(m);
        for (int h : opt.hidden) layer_dims.push_back(h);
        for (int i = static_cast<int>(opt.hidden.size()) - 2; i >= 0; --i) {
            layer_dims.push_back(opt.hidden[i]);
        }
        layer_dims.push_back(m);

        GpuAutoencoder net;
        net.init(layer_dims, rng);
        std::cout << "Model initialization: " << ms_since(t_init) << " ms" << std::endl;
        nvtxRangePop();

        std::cout << "\n=== Training summary ===\n"
                  << "  files           : " << opt.files.size() << '\n'
                  << "  features (m)    : " << m << '\n'
                  << "  layer dims      : ";
        for (size_t i = 0; i < layer_dims.size(); ++i) {
            std::cout << layer_dims[i] << (i + 1 < layer_dims.size() ? " -> " : "");
        }
        std::cout << '\n'
                  << "  batch size      : " << opt.batch_size << '\n'
                  << "  chunk size      : " << opt.chunk_size << '\n'
                  << "  epochs          : " << opt.epochs << '\n'
                  << "  learning rate   : " << opt.lr << '\n'
                  << "  seed            : " << opt.seed << '\n' << std::endl;

        std::vector<float> losses;
        
        // Create trainer stream for computation
        cudaStream_t trainer_stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&trainer_stream, cudaStreamNonBlocking));
        
        nvtxRangePushA("main:train_loop");
        auto t_train_total = std::chrono::steady_clock::now();
        std::cout << "[HANGDB] Starting training loop for " << opt.epochs << " epochs" << std::endl;
        
        // Start the loader once at the beginning
        std::cout << "[HANGDB] Calling loader.start()" << std::endl;
        loader.start();
        std::cout << "[HANGDB] loader.start() returned, resetting loss" << std::endl;
        net.reset_epoch_loss();
        
        // Track per-lane loss accumulation
        std::vector<double> lane_loss_sum(1, 0.0);    // one lane for SINGLE policy
        std::vector<int> lane_loss_count(1, 0);

        int num_batches = 0;
        SparseBatch batch;
        int lane, slot;
        int primary_lane = 0;  // Default to lane 0

        // Accumulate timing across all passes
        double time_acquire_ready = 0.0;
        double time_forward = 0.0;
        double time_backward = 0.0;

        while (true) {
            // Check if primary lane has completed max passes
            if (ring.lane_pass(primary_lane) >= opt.epochs) {
                std::cout << "[HANGDB] Primary lane " << primary_lane << " has completed " 
                          << ring.lane_pass(primary_lane) << " passes, stopping" << std::endl;
                break;
            }

            std::cout << "[HANGDB] Calling ring.acquire_ready(): awaiting READY" << std::endl;
            nvtxRangePushA("acquire_ready");
            auto t_batch_load = std::chrono::steady_clock::now();
            bool has_batch = ring.acquire_ready(&batch, &lane, &slot);
            time_acquire_ready += ms_since(t_batch_load);
            nvtxRangePop();
            std::cout << "[HANGDB] READY batch acquired" << std::endl;
            
            if (!has_batch) {
                std::cout << "[HANGDB] No more batches, batch loop ending (num_batches=" << num_batches << ")" << std::endl;
                break;
            }

            // GPU fence: wait for loader's H2D + lognorm to complete
            CUDA_CHECK(cudaStreamWaitEvent(trainer_stream, batch.ready_event, 0));

            nvtxRangePushA("forward");
            auto t_fwd = std::chrono::steady_clock::now();
            net.forward(batch, trainer_stream);
            time_forward += ms_since(t_fwd);
            nvtxRangePop();

            nvtxRangePushA("backward_and_step");
            auto t_bwd = std::chrono::steady_clock::now();
            net.backward_and_step(batch, opt.lr, trainer_stream);
            time_backward += ms_since(t_bwd);
            nvtxRangePop();
            ring.release_consumed(lane, slot);
            
            ++num_batches;
            lane_loss_count[lane]++;

            // On chunk end marker, read and report loss for this lane's chunk
            if (batch.chunk_end) {
                // Sync to read the accumulated loss
                CUDA_CHECK(cudaStreamSynchronize(trainer_stream));
                float chunk_mean_loss = net.read_epoch_loss(lane_loss_count[lane]);
                
                int current_pass = ring.lane_pass(lane);
                std::cout << "[lane=" << lane << " pass=" << current_pass << "] mean_loss=" 
                          << std::fixed << std::setprecision(6) << chunk_mean_loss
                          << "  (chunk end, " << lane_loss_count[lane] << " batches)" << std::endl;
                
                // Reset accumulator for next chunk
                net.reset_epoch_loss();
                lane_loss_sum[lane] = 0.0;
                lane_loss_count[lane] = 0;
                
                losses.push_back(chunk_mean_loss);
            }
        }
        
        std::cout << "[HANGDB] Batch loop complete, calling ring.shutdown()" << std::endl;
        ring.shutdown();
        
        double train_total_ms = ms_since(t_train_total);
        std::cout << "[HANGDB] Training loop completed, total_ms=" << std::fixed << std::setprecision(1) << train_total_ms << std::endl;
        nvtxRangePop();
        
        // Clean up trainer stream
        CUDA_CHECK(cudaStreamDestroy(trainer_stream));
        
        // Print grand total timings across all passes
        gpu_timers().report(std::cout, "All Passes Grand Total", true);
        
        // Print a final summary of grand totals across all passes
        std::cout << "\n=== Training Complete ===\n" << std::endl;
        
        if (!losses.empty()) {
            float scaler = 80.0f / losses[0];
            int col;
            for (float l : losses) {
                col = static_cast<int>(std::lround(l * scaler));
                if (col < 0) col = 0;
                std::cout << std::string(col, ' ') << "|  " << l << '\n';
            }
        }

        std::cout << "\nDone." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
