// SPDX-License-Identifier: MIT
// Thin driver: parse CLI, load dataset, train an Autoencoder.

#include "autoencoder.h"
#include "data.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

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
                 " [--seed N] <file1.1pz> [file2.1pz ...]\n";
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
    return opt;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        std::cout << "Loading " << opt.files.size() << " file(s)..." << std::endl;
        Dataset ds = load_dataset(opt.files);
        const int m = static_cast<int>(ds.X.rows());
        const int n = static_cast<int>(ds.X.cols());
        if (n == 0) throw std::runtime_error("dataset is empty (0 cells)");

        // Build full layer list: [m] + hidden + reverse(hidden[:-1]) + [m]
        std::vector<int> layer_dims;
        layer_dims.push_back(m);
        for (int h : opt.hidden) layer_dims.push_back(h);
        for (int i = static_cast<int>(opt.hidden.size()) - 2; i >= 0; --i) {
            layer_dims.push_back(opt.hidden[i]);
        }
        layer_dims.push_back(m);

        std::mt19937 rng(opt.seed);
        Autoencoder net;
        net.init(layer_dims, rng);

        std::cout << "\n=== Training summary ===\n"
                  << "  cells (samples) : " << n << '\n'
                  << "  features (m)    : " << m << '\n'
                  << "  layer dims      : ";
        for (size_t i = 0; i < layer_dims.size(); ++i) {
            std::cout << layer_dims[i] << (i + 1 < layer_dims.size() ? " -> " : "");
        }
        std::cout << '\n'
                  << "  batch size      : " << opt.batch_size << '\n'
                  << "  epochs          : " << opt.epochs << '\n'
                  << "  learning rate   : " << opt.lr << '\n'
                  << "  seed            : " << opt.seed << '\n' << std::endl;

        // Reusable minibatch buffer. The trailing partial batch is SKIPPED to
        // keep buffer shapes fixed across iterations.
        const int B = opt.batch_size;
        Eigen::MatrixXf batch(m, B);

        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);

        for (int epoch = 1; epoch <= opt.epochs; ++epoch) {
            std::shuffle(order.begin(), order.end(), rng);

            const int num_batches = n / B;   // skip the trailing partial batch
            double    loss_sum    = 0.0;

            for (int bi = 0; bi < num_batches; ++bi) {
                batch.setZero();
                for (int j = 0; j < B; ++j) {
                    const int col = order[bi * B + j];
                    for (SpMatF::InnerIterator it(ds.X, col); it; ++it) {
                        batch(it.row(), j) = it.value();
                    }
                }
                net.forward(batch);
                loss_sum += net.backward_and_step(batch, opt.lr);
            }
            const double mean_loss = (num_batches > 0)
                                     ? loss_sum / num_batches : 0.0;
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
