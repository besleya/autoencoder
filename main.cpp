// SPDX-License-Identifier: MIT
// Autoencoder for single-cell count data (CSC .1pz inputs), trained with
// manual backprop in Eigen. Single translation unit, C++17.

#include <singlet/singlet.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using PZMatrix = Eigen::SparseMatrix<uint32_t, Eigen::ColMajor>;
using SpMatF   = Eigen::SparseMatrix<float, Eigen::ColMajor>;

// ---------------------------------------------------------------------------
// Autoencoder
// ---------------------------------------------------------------------------
struct Autoencoder {
    // Full layer dimension list, e.g. {m, 512, 128, 32, 128, 512, m}.
    // weights[l] has shape (dims[l+1], dims[l]); biases[l] has length dims[l+1].
    std::vector<int> dims;
    std::vector<Eigen::MatrixXf> W;
    std::vector<Eigen::VectorXf> b;

    // Reusable forward/backward buffers (sized for the current batch).
    std::vector<Eigen::MatrixXf> z;   // pre-activations,  z[l] = W[l]*a[l] + b[l]
    std::vector<Eigen::MatrixXf> a;   // activations,     a[0] = input, a[L] = output
    std::vector<Eigen::MatrixXf> dW;
    std::vector<Eigen::VectorXf> db;
    std::vector<Eigen::MatrixXf> dz;  // gradient wrt pre-activation
    // std::vector<Eigen::MatrixXf> da;  // gradient wrt activation

    int num_layers() const { return static_cast<int>(W.size()); }
    int last_layer() const { return num_layers() - 1; }

    void init(const std::vector<int>& layer_dims, std::mt19937& rng) {
        dims = layer_dims;
        const int L = static_cast<int>(dims.size()) - 1;
        W.resize(L);
        b.resize(L);
        for (int l = 0; l < L; ++l) {
            const int in  = dims[l];
            const int out = dims[l + 1];
            // He init: N(0, sqrt(2/in))
            const float stddev = std::sqrt(2.0f / static_cast<float>(in));
            std::normal_distribution<float> nd(0.0f, stddev);
            W[l].resize(out, in);
            for (int j = 0; j < in; ++j)
                for (int i = 0; i < out; ++i)
                    W[l](i, j) = nd(rng);
            b[l] = Eigen::VectorXf::Zero(out);
        }
        z.resize(L);
        a.resize(L + 1);
        dW.resize(L);
        db.resize(L);
        dz.resize(L);
        // da.resize(L + 1);
    }

    // Forward pass. Input `x` has shape (dims[0], B). After the call, a[L] is
    // the reconstruction. ReLU on every hidden layer including the bottleneck;
    // linear on the final decoder output layer.
    void forward(const Eigen::MatrixXf& x) {
        a[0] = x;
        const int L = num_layers();
        for (int l = 0; l < L; ++l) {
            z[l].noalias() = W[l] * a[l];
            z[l].colwise() += b[l];
            if (l == L - 1) {
                a[l + 1] = z[l];                    // linear output
            } else {
                a[l + 1] = z[l].cwiseMax(0.0f);     // ReLU
            }
        }
    }

    // Backprop MSE loss against target `x`. Returns mean-squared-error
    // (averaged over batch and features) for monitoring.
    float backward_and_step(const Eigen::MatrixXf& x, float lr) {
        const int L = num_layers();
        const float B = static_cast<float>(x.cols());

        // dL/da_L = (a_L - x) / B  (MSE; factor of 2 absorbed into lr)
        // da[L] = (a[L] - x) / B;
        dz[L - 1] = (a[L] - x) / B;

        // Compute scalar loss for reporting (mean over all entries).
        const float loss = (a[L] - x).squaredNorm() /
                           (static_cast<float>(x.rows()) * B);

        for (int l = L - 1; l >= 0; --l) {
            if (l == L - 1) {
                // dz[l] = da[l + 1];                                  // linear
            } else {
                dz[l] = dz[l].cwiseMax(0.0f);
                // dz[l] = da[l + 1].array() *
                //         (z[l].array() > 0.0f).cast<float>();        // ReLU'
            }
            dW[l].noalias() = dz[l] * a[l].transpose();
            db[l]            = dz[l].rowwise().sum();
            if (l > 0) {
                dz[l - 1].noalias() = W[l].transpose() * dz[l];
                // da[l].noalias() = W[l].transpose() * dz[l];
            }
            W[l].noalias() -= lr * dW[l];
            b[l]            -= lr * db[l];
        }
        return loss;
    }
};

// ---------------------------------------------------------------------------
// Data loading: concatenate cells (columns) from all input files.
// ---------------------------------------------------------------------------
struct Dataset {
    uint32_t m = 0;                       // features (rows)
    SpMatF   X;                           // (m, total_cells), each col = sample
};

// Load a single .1pz file as a sparse float matrix. Check that the number of
// rows matches `expected_m` if it's nonzero, for consistency across files.
static SpMatF load_one(const std::string& path, uint32_t expected_m) {
    auto result = singlet::pz::read_1pz(path);
    std::cout << "  loaded " << path << "  (" << result.m << " x " << result.n
              << ", nnz=" << result.nnz << ')' << std::endl;
    if (expected_m != 0 && result.m != expected_m) {
        std::ostringstream oss;
        oss << "feature-count mismatch in " << path
            << ": got " << result.m << ", expected " << expected_m;
        throw std::runtime_error(oss.str());
    }
    std::vector<int32_t> outer(result.indptr.begin(),  result.indptr.end());
    std::vector<int32_t> inner(result.indices.begin(), result.indices.end());
    Eigen::Map<PZMatrix> map(result.m, result.n, result.nnz,
                             outer.data(), inner.data(), result.data.data());
    PZMatrix u32 = map.eval();
    // Cast to float sparse matrix.
    return u32.cast<float>();
}

// Load and concatenate multiple .1pz files. Each file is a sparse matrix with
// the same number of rows (features) but possibly different numbers of columns
// (cells). The result is a single sparse matrix with all columns concatenated,
// and the total number of rows/features. The `Dataset` struct also stores the
// number of rows/features `m` for convenience.
static Dataset load_dataset(const std::vector<std::string>& paths) {
    Dataset ds;
    std::vector<SpMatF> mats;
    mats.reserve(paths.size());
    uint32_t total_cols = 0;
    for (const auto& p : paths) {
        SpMatF m = load_one(p, ds.m);
        if (ds.m == 0) ds.m = static_cast<uint32_t>(m.rows());
        total_cols += static_cast<uint32_t>(m.cols());
        mats.push_back(std::move(m));
    }
    ds.X.resize(ds.m, total_cols);
    // Reserve per-column nnz to make insertion fast.
    Eigen::VectorXi reserve_per_col(total_cols);
    uint32_t col_off = 0;
    for (const auto& mat : mats) {
        for (int c = 0; c < mat.cols(); ++c) {
            reserve_per_col[col_off + c] =
                mat.outerIndexPtr()[c + 1] - mat.outerIndexPtr()[c];
        }
        col_off += static_cast<uint32_t>(mat.cols());
    }
    ds.X.reserve(reserve_per_col);
    col_off = 0;
    for (const auto& mat : mats) {
        for (int c = 0; c < mat.cols(); ++c) {
            for (SpMatF::InnerIterator it(mat, c); it; ++it) {
                ds.X.insert(it.row(), col_off + c) = it.value();
            }
        }
        col_off += static_cast<uint32_t>(mat.cols());
    }
    ds.X.makeCompressed();
    return ds;
}

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
        std::cout << "Files loaded" << std::endl;
        const int m = static_cast<int>(ds.X.rows());
        const int n = static_cast<int>(ds.X.cols());
        if (n == 0) throw std::runtime_error("dataset is empty (0 cells)");

        // Build full layer list: [m] + hidden + reverse(hidden[:-1]) + [m]
        std::vector<int> layer_dims;
        layer_dims.push_back(m);
        for (int h : opt.hidden) layer_dims.push_back(h);
        // Decoder mirrors encoder hidden layers (excluding bottleneck) then m.
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

        // Reusable minibatch buffer. The final partial batch (size < B) is
        // SKIPPED to keep the buffer shape and all per-layer activation
        // buffers fixed across iterations.
        const int B = opt.batch_size;
        Eigen::MatrixXf batch(m, B);

        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);

        for (int epoch = 1; epoch <= opt.epochs; ++epoch) {
            std::shuffle(order.begin(), order.end(), rng);

            const int num_batches = n / B;   // skip the trailing partial batch
            double   loss_sum   = 0.0;

            for (int bi = 0; bi < num_batches; ++bi) {
                // Refill the reusable buffer in place.
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
