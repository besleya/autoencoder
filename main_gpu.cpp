// SPDX-License-Identifier: MIT
// GPU driver: parse CLI, load dataset via GpuDataLoader, train a GpuAutoencoder.

#include "gpu_autoencoder.h"
#include "gpu_data_loader.h"
#include "data.h"

#include <algorithm>
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
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        // Peek the feature count by loading the first file directly.
        // This is done before constructing the RNG to avoid consuming RNG state
        // out of order.
        std::cout << "Peeking feature count from " << opt.files[0] << "..." << std::endl;
        PZHeader hdr = validate_1pz(opt.files[0]);
        const int m = static_cast<int>(hdr.m);
        if (m <= 0) {
            throw std::runtime_error("first file has no features");
        }
        // first_file goes out of scope; its device memory is freed.

        // Build full layer list: [m] + hidden + reverse(hidden[:-1]) + [m]
        std::vector<int> layer_dims;
        layer_dims.push_back(m);
        for (int h : opt.hidden) layer_dims.push_back(h);
        for (int i = static_cast<int>(opt.hidden.size()) - 2; i >= 0; --i) {
            layer_dims.push_back(opt.hidden[i]);
        }
        layer_dims.push_back(m);

        // Construct RNG and initialize the autoencoder BEFORE the first epoch.
        // This ensures RNG is consumed in the same order as the CPU main.cpp.
        std::mt19937 rng(opt.seed);
        GpuAutoencoder net;
        net.init(layer_dims, rng);

        // Create the data loader. RNG is passed by reference; from now on,
        // begin_epoch() and next_batch() will consume RNG state.
        std::cout << "Constructing GPU data loader..." << std::endl;
        GpuDataLoader loader(opt.files, opt.chunk_size, opt.batch_size, rng);

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

        for (int epoch = 1; epoch <= opt.epochs; ++epoch) {
            loader.begin_epoch();

            float loss_sum = 0.0f;
            int num_batches = 0;
            SparseBatch batch;

            while (loader.next_batch(&batch)) {
                net.forward(batch);
                loss_sum += net.backward_and_step(batch, opt.lr);
                ++num_batches;
            }

            float mean_loss = (num_batches > 0)
                            ? loss_sum / static_cast<float>(num_batches) : 0.0f;
            std::cout << "epoch " << epoch << "/" << opt.epochs
                      << "  batches=" << num_batches
                      << "  mean_recon_loss=" << mean_loss << std::endl;
        }

        std::cout << "\nDone." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
