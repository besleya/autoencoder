// SPDX-License-Identifier: MIT
// GPU driver: parse CLI, load dataset via DataLoader, train with Translator over shared layers.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <glob.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "batch.h"
#include "data_loader.h"
// #include "autoencoder.h"
#include "gpu_timer.h"
#include "ring.h"
// #include "translator.h"

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

// ============================================================================
// Data structures
// ============================================================================

struct SpeciesSpec {
    std::string name;
    std::vector<std::string> paths;
};

struct TrainerOptions {
    int epochs = 10;
    int batch_size = 512;
    float lr = 1e-3f;
    int chunk_size = 1;
    int workers = 4;
    int n_slots_per_loader = 4;
    unsigned seed = 42;
    
    // Architecture
    int shared_layers = 2;
    int species_layers = 2;
    std::vector<int> shared_dims = {128, 32};
    std::vector<int> species_dims = {256, 128};
    
    // Input: --species entries or legacy positional files
    std::vector<SpeciesSpec> species;
};

// ============================================================================
// CLI parsing
// ============================================================================

static std::vector<std::string> glob_expand(const std::string& pattern) {
    std::vector<std::string> result;
    glob_t globbuf = {};
    int ret = glob(pattern.c_str(), GLOB_ERR | GLOB_MARK, nullptr, &globbuf);
    if (ret == 0) {
        for (size_t i = 0; i < globbuf.gl_pathc; ++i) {
            std::string path = globbuf.gl_pathv[i];
            // Skip directories (glob marks them with /)
            if (!path.empty() && path.back() != '/') {
                result.push_back(path);
            }
        }
    }
    globfree(&globbuf);
    return result;
}

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
    std::cerr << "Usage: " << prog << " [OPTIONS] [files...]\n"
              << "\nOptions:\n"
              << "  --epochs N                  number of epochs (default 10)\n"
              << "  --batch N                   batch size (default 512)\n"
              << "  --lr F                      learning rate (default 0.001)\n"
              << "  --chunk N                   files per chunk (default 1)\n"
              << "  --seed N                    random seed (default 42)\n"
              << "  --workers N                 worker threads (default 4)\n"
              << "  --shared-layers N           shared encoder layers (default 2)\n"
              << "  --species-layers N          species encoder layers (default 2)\n"
              << "  --shared-dims D1,D2,...     shared layer inner dims (default 128,32)\n"
              << "  --species-dims D1,D2,...    species layer inner dims (default 256,128)\n"
              << "  --species NAME:PATTERN      register species with glob pattern (repeatable)\n"
              << "  -h, --help                  show this help\n"
              << "\nLegacy positional args [files...] are treated as species 'default'.\n";
}

static TrainerOptions parse_args(int argc, char** argv) {
    TrainerOptions opt;
    
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return std::string(argv[++i]);
        };
        
        if      (a == "--epochs")         opt.epochs = std::stoi(need("--epochs"));
        else if (a == "--batch")          opt.batch_size = std::stoi(need("--batch"));
        else if (a == "--lr")             opt.lr = std::stof(need("--lr"));
        else if (a == "--chunk")          opt.chunk_size = std::stoi(need("--chunk"));
        else if (a == "--seed")           opt.seed = static_cast<unsigned>(std::stoul(need("--seed")));
        else if (a == "--workers")        opt.workers = std::stoi(need("--workers"));
        else if (a == "--shared-layers")  opt.shared_layers = std::stoi(need("--shared-layers"));
        else if (a == "--species-layers") opt.species_layers = std::stoi(need("--species-layers"));
        else if (a == "--shared-dims")    opt.shared_dims = parse_csv_ints(need("--shared-dims"));
        else if (a == "--species-dims")   opt.species_dims = parse_csv_ints(need("--species-dims"));
        else if (a == "--species") {
            std::string spec = need("--species");
            auto colon = spec.find(':');
            if (colon == std::string::npos) {
                throw std::runtime_error("--species format: NAME:PATTERN");
            }
            std::string name = spec.substr(0, colon);
            std::string pattern = spec.substr(colon + 1);
            auto paths = glob_expand(pattern);
            if (paths.empty()) {
                throw std::runtime_error("--species " + name + ": pattern matched no files");
            }
            opt.species.push_back({name, paths});
        }
        else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            exit(0);
        }
        else if (!a.empty() && a[0] == '-') {
            throw std::runtime_error("unknown option: " + a);
        }
        else {
            // Positional file argument: accumulate in default species
            if (opt.species.empty() || opt.species[0].name != "default") {
                opt.species.push_back({"default", {}});
            }
            opt.species[0].paths.push_back(a);
        }
    }
    
    // Validate
    if (opt.species.empty()) {
        usage(argv[0]);
        throw std::runtime_error("no input files or species provided");
    }
    if (opt.epochs <= 0) throw std::runtime_error("--epochs must be > 0");
    if (opt.batch_size <= 0) throw std::runtime_error("--batch must be > 0");
    if (opt.chunk_size <= 0) throw std::runtime_error("--chunk must be > 0");
    if (opt.workers <= 0) throw std::runtime_error("--workers must be > 0");
    if (opt.shared_layers <= 0) throw std::runtime_error("--shared-layers must be > 0");
    if (opt.species_layers <= 0) throw std::runtime_error("--species-layers must be > 0");
    if (opt.shared_dims.size() != (size_t)opt.shared_layers) {
        throw std::runtime_error("--shared-dims size must match --shared-layers");
    }
    if (opt.species_dims.size() != (size_t)opt.species_layers) {
        throw std::runtime_error("--species-dims size must match --species-layers");
    }
    
    // Check all species have files
    for (const auto& sp : opt.species) {
        if (sp.paths.empty()) {
            throw std::runtime_error("species " + sp.name + " has no files");
        }
    }
    
    return opt;
}

// ============================================================================
// Helper functions
// ============================================================================

static double ms_since(std::chrono::steady_clock::time_point t0) {
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static std::vector<std::unique_ptr<DataLoader>> build_loaders(
    const std::vector<SpeciesSpec>& species_list,
    const TrainerOptions& opt,
    std::mt19937& rng)
{
    std::vector<std::unique_ptr<DataLoader>> loaders;
    for (const auto& spec : species_list) {
        auto loader = std::make_unique<DataLoader>(
            spec.name,
            spec.paths,
            opt.chunk_size,
            opt.batch_size,
            opt.n_slots_per_loader,
            rng);
        loaders.push_back(std::move(loader));
    }
    return loaders;
}

static void training_loop(
    Ring& ring,
    // Translator& translator,
    const std::vector<std::unique_ptr<DataLoader>>& loaders,
    const TrainerOptions& opt)
{
    // Create trainer stream
    cudaStream_t trainer_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&trainer_stream, cudaStreamNonBlocking));
    
    // Track batches seen per species (for loss averaging)
    std::map<std::string, int> batches_seen;
    
    // The primary species is the first loader; its pass() counter drives epoch termination
    DataLoader* primary_loader = loaders.front().get();
    
    nvtxRangePushA("main:train_loop");
    auto t_train_start = std::chrono::steady_clock::now();
    
    int primary_pass = 0;
    int num_batches = 0;
    std::vector<float> losses;
    
    while (primary_pass < opt.epochs) {
        std::unique_ptr<Batch> batch = ring.next_ready_batch();
        if (!batch) break;
        
        // Wait for batch to be ready (H2D + log-norm complete)
        CUDA_CHECK(cudaStreamWaitEvent(trainer_stream, batch->ready_event(), 0));
        
        // Forward pass
        nvtxRangePushA("forward");
        // translator.forward(*batch, trainer_stream);
        nvtxRangePop();
        
        // Backward and update
        nvtxRangePushA("backward_and_step");
        // translator.backward_and_step(*batch, opt.lr, trainer_stream);
        nvtxRangePop();
        
        // Sync to ensure slot buffers are safe to reuse
        CUDA_CHECK(cudaStreamSynchronize(trainer_stream));
        
        // Track batch count per species
        const auto& species_name = batch->species_name();
        batches_seen[species_name]++;
        
        // On chunk end, report loss
        if (batch->chunk_end()) {
            // Autoencoder& ae = translator.model(species_name);
            // float loss = ae.read_epoch_loss(batches_seen[species_name]);
            // std::cout << "[species=" << species_name << "] mean_loss="
            //           << std::fixed << std::setprecision(9) << loss
            //           << "  (chunk end, " << batches_seen[species_name] << " batches)" << std::endl;
            // ae.reset_epoch_loss();
            batches_seen[species_name] = 0;
            // losses.push_back(loss);
        }
        
        ++num_batches;
        
        // Batch destructor releases the slot
        batch.reset();
        
        // Check if primary species has finished its epochs
        primary_pass = primary_loader->epoch();
    }
    
    double train_ms = ms_since(t_train_start);
    std::cout << "\nTraining loop: " << std::fixed << std::setprecision(1) << train_ms
              << " ms for " << num_batches << " batches\n" << std::endl;
    
    nvtxRangePop();
    
    // Shutdown ring
    ring.shutdown();
    
    // Print loss history
    // if (!losses.empty()) {
    //     float scaler = 80.0f / losses[0];
    //     for (float l : losses) {
    //         int col = static_cast<int>(std::lround(l * scaler));
    //         if (col < 0) col = 0;
    //         std::cout << std::string(col, ' ') << "|  " << std::setprecision(9) << l << '\n';
    //     }
    // }
    
    gpu_timers().report(std::cout, "All Passes Grand Total", true);
    
    // Clean up
    CUDA_CHECK(cudaStreamDestroy(trainer_stream));
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    try {
        auto t_start = std::chrono::steady_clock::now();
        TrainerOptions opt = parse_args(argc, argv);
        std::cout << "Argument parsing: " << ms_since(t_start) << " ms" << std::endl;
        
        // Initialize RNG
        auto t_init = std::chrono::steady_clock::now();
        nvtxRangePushA("main:init");
        std::mt19937 rng(opt.seed);
        
        // Build loaders
        std::cout << "Building DataLoaders..." << std::endl;
        auto loaders = build_loaders(opt.species, opt, rng);
        
        // Start loaders and get feature counts
        std::vector<int> feature_counts;
        std::cout << "Starting loaders..." << std::endl;
        for (auto& loader : loaders) {
            loader->start();
            int m = loader->feature_count();
            if (m <= 0) {
                throw std::runtime_error("loader " + loader->species_name() + " reports m <= 0");
            }
            feature_counts.push_back(m);
            std::cout << "  " << loader->species_name() << ": m=" << m << std::endl;
        }
        
        // Build Translator
        // std::cout << "Building Translator..." << std::endl;
        // Translator translator(opt.shared_layers, opt.species_layers,
        //                       opt.shared_dims, opt.species_dims, rng);
        
        // // Register species with translator
        // for (size_t i = 0; i < loaders.size(); ++i) {
        //     translator.species(loaders[i]->species_name(), feature_counts[i]);
        // }
        
        std::cout << "Model initialization: " << ms_since(t_init) << " ms" << std::endl;
        nvtxRangePop();
        
        // Print training summary
        std::cout << "\n=== Training Summary ===\n"
                  << "  species         : " << loaders.size() << '\n'
                  << "  epochs          : " << opt.epochs << '\n'
                  << "  batch size      : " << opt.batch_size << '\n'
                  << "  chunk size      : " << opt.chunk_size << '\n'
                  << "  workers         : " << opt.workers << '\n'
                  << "  learning rate   : " << opt.lr << '\n'
                  << "  seed            : " << opt.seed << '\n'
                  << "  shared layers   : " << opt.shared_layers
                  << " with dims [";
        for (size_t i = 0; i < opt.shared_dims.size(); ++i) {
            std::cout << opt.shared_dims[i] << (i + 1 < opt.shared_dims.size() ? "," : "");
        }
        std::cout << "]\n  species layers  : " << opt.species_layers
                  << " with dims [";
        for (size_t i = 0; i < opt.species_dims.size(); ++i) {
            std::cout << opt.species_dims[i] << (i + 1 < opt.species_dims.size() ? "," : "");
        }
        std::cout << "]\n" << std::endl;
        
        // Create and start ring
        std::cout << "Creating Ring with " << opt.workers << " workers..." << std::endl;
        Ring ring(opt.workers);
        for (auto& loader : loaders) {
            ring.add_loader(loader.get());
        }
        ring.start();
        
        // Main training loop
        std::cout << "Starting training loop..." << std::endl;
        training_loop(ring, /*translator,*/ loaders, opt);
        
        std::cout << "\n=== Training Complete ===" << std::endl;
        std::cout << "Done." << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
