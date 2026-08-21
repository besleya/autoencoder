// SPDX-License-Identifier: MIT

#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdexcept>

#include "autoencoder.h"
#include "batch.h"
#include "layer.h"
#include "translator.h"

// ============================================================================
// Error checking macros
// ============================================================================

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, \
                    __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

#define CUBLAS_CHECK(call)                                        \
    do {                                                          \
        cublasStatus_t err = call;                                \
        if (err != CUBLAS_STATUS_SUCCESS) {                       \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, \
                    __LINE__, err);                               \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

#define CUSPARSE_CHECK(call)                                      \
    do {                                                          \
        cusparseStatus_t err = call;                              \
        if (err != CUSPARSE_STATUS_SUCCESS) {                     \
            fprintf(stderr, "cuSPARSE error at %s:%d: %d\n", __FILE__, \
                    __LINE__, err);                               \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// ============================================================================
// Translator implementation
// ============================================================================

Translator::Translator(int n_shared_layers,
                       int n_species_layers,
                       const std::vector<int>& shared_dims,
                       const std::vector<int>& species_dims,
                       std::mt19937& rng)
    : species_dims_(species_dims), shared_dims_(shared_dims), rng_(rng),
      cublas_handle_(nullptr), cusparse_handle_(nullptr) {
    
    // Validate dimensions
    if (static_cast<int>(shared_dims.size()) != n_shared_layers) {
        throw std::invalid_argument(
            "shared_dims.size() does not match n_shared_layers");
    }
    if (static_cast<int>(species_dims.size()) != n_species_layers) {
        throw std::invalid_argument(
            "species_dims.size() does not match n_species_layers");
    }
    
    // Create cublas and cusparse handles
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CUSPARSE_CHECK(cusparseCreate(&cusparse_handle_));
    
    // Build shared layers
    build_shared_layers_(n_shared_layers, shared_dims);
}

Translator::~Translator() {
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
    if (cusparse_handle_) {
        cusparseDestroy(cusparse_handle_);
    }
}

void Translator::build_shared_layers_(int n_shared_layers,
                                       const std::vector<int>& shared_dims) {
    // Dimension flow — sandwich architecture:
    //
    // The Translator constructs a shared encoder-decoder pair that bridges
    // species-specific layers to a bottleneck. For each species, the full
    // layer stack is:
    //
    //   species_encoder: m → species_dims[0] → ... → species_dims[n-1]
    //   shared_encoder:  species_dims[n-1] → shared_dims[0] → ... → shared_dims[m-1] (bottleneck)
    //   shared_decoder:  shared_dims[m-1] → ... → shared_dims[0]
    //   species_decoder: species_dims[n-1] → ... → species_dims[0] → m
    //
    // CONSTRAINT: The caller must ensure shared_dims[0] == species_dims[n-1]
    // for dimensional compatibility between shared_decoder output and species_decoder input.
    //
    // Shared encoder: n layers
    //   Layer i: (i == 0) ? species_dims.back() → shared_dims[0]
    //            : shared_dims[i-1] → shared_dims[i]
    //
    // Shared decoder: n layers (mirror of encoder)
    //   Layer i: shared_dims[n-1-i] → shared_dims[n-2-i]  for i < n-1
    //            shared_dims[1] → shared_dims[0]           for i == n-1
    
    shared_encoder_.clear();
    shared_decoder_.clear();
    
    // Build shared encoder
    for (int l = 0; l < n_shared_layers; ++l) {
        int in_dim = (l == 0) ? species_dims_.back() : shared_dims[l - 1];
        int out_dim = shared_dims[l];
        
        // All shared layers use ReLU (they are hidden layers within the sandwich)
        Layer::Activation activation = Layer::Activation::ReLU;
        bool sparse_input = false;  // Only the first layer of the whole network is sparse
        unsigned long seed = rng_.get()();
        
        auto layer = std::make_shared<Layer>(in_dim, out_dim, activation, sparse_input,
                                              seed, cublas_handle_, cusparse_handle_);
        shared_encoder_.push_back(layer);
    }
    
    // Build shared decoder (mirror of encoder, so reverse shared_dims)
    for (int l = 0; l < n_shared_layers; ++l) {
        int in_dim = shared_dims[n_shared_layers - 1 - l];
        int out_dim = (l == n_shared_layers - 1) ? shared_dims[0] : shared_dims[n_shared_layers - 2 - l];
        
        // All shared decoder layers use ReLU (they are hidden layers)
        Layer::Activation activation = Layer::Activation::ReLU;
        bool sparse_input = false;
        unsigned long seed = rng_.get()();
        
        auto layer = std::make_shared<Layer>(in_dim, out_dim, activation, sparse_input,
                                              seed, cublas_handle_, cusparse_handle_);
        shared_decoder_.push_back(layer);
    }
}

void Translator::species(const std::string& name, int feature_count) {
    // Check that this species name is not already registered
    if (models_.count(name) > 0) {
        throw std::invalid_argument("Species '" + name + "' is already registered");
    }
    
    int n_species_layers = static_cast<int>(species_dims_.size());
    int n_shared_layers = static_cast<int>(shared_encoder_.size());
    
    // Build species encoder: feature_count → species_dims[0] → ... → species_dims[n-1]
    std::vector<std::shared_ptr<Layer>> species_encoder;
    for (int l = 0; l < n_species_layers; ++l) {
        int in_dim = (l == 0) ? feature_count : species_dims_[l - 1];
        int out_dim = species_dims_[l];
        
        // Hidden species layers use ReLU; but the last output of species encoder
        // is an intermediate, so it should be ReLU (it feeds into shared encoder)
        Layer::Activation activation = Layer::Activation::ReLU;
        bool sparse_input = (l == 0);  // Only the first layer overall is sparse
        unsigned long seed = rng_.get()();
        
        auto layer = std::make_shared<Layer>(in_dim, out_dim, activation, sparse_input,
                                              seed, cublas_handle_, cusparse_handle_);
        species_encoder.push_back(layer);
    }
    
    // Build species decoder: species_dims[n-1] → ... → species_dims[0] → feature_count
    std::vector<std::shared_ptr<Layer>> species_decoder;
    for (int l = 0; l < n_species_layers; ++l) {
        int in_dim = species_dims_[n_species_layers - 1 - l];
        int out_dim = (l == n_species_layers - 1) ? feature_count : species_dims_[n_species_layers - 2 - l];
        
        // All species decoder layers except the last use ReLU.
        // The last (output) layer uses None activation.
        Layer::Activation activation = (l == n_species_layers - 1)
                                        ? Layer::Activation::None
                                        : Layer::Activation::ReLU;
        bool sparse_input = false;
        unsigned long seed = rng_.get()();
        
        auto layer = std::make_shared<Layer>(in_dim, out_dim, activation, sparse_input,
                                              seed, cublas_handle_, cusparse_handle_);
        species_decoder.push_back(layer);
    }
    
    // Concatenate: [species_enc] + [shared_enc] + [shared_dec] + [species_dec]
    std::vector<std::shared_ptr<Layer>> all_layers;
    all_layers.insert(all_layers.end(), species_encoder.begin(), species_encoder.end());
    all_layers.insert(all_layers.end(), shared_encoder_.begin(), shared_encoder_.end());
    all_layers.insert(all_layers.end(), shared_decoder_.begin(), shared_decoder_.end());
    all_layers.insert(all_layers.end(), species_decoder.begin(), species_decoder.end());
    
    // Build layer_dims for the full sandwich topology:
    // Dimension flow: feature_count → species_dims[0] → ... → species_dims[n-1]
    //                 → shared_dims[0] → ... → shared_dims[m-1] (bottleneck)
    //                 → shared_dims[m-2] → ... → shared_dims[0]
    //                 → species_dims[n-2] → ... → species_dims[0] → feature_count
    std::vector<int> layer_dims;
    layer_dims.push_back(feature_count);
    
    for (int d : species_dims_) {
        layer_dims.push_back(d);
    }
    
    for (int d : shared_dims_) {
        layer_dims.push_back(d);
    }
    
    // Shared decoder in reverse (excluding the final bottleneck dimension)
    for (int i = static_cast<int>(shared_dims_.size()) - 2; i >= 0; --i) {
        layer_dims.push_back(shared_dims_[i]);
    }
    
    // Species decoder in reverse (excluding the first dimension after shared decoder)
    for (int i = static_cast<int>(species_dims_.size()) - 2; i >= 0; --i) {
        layer_dims.push_back(species_dims_[i]);
    }
    
    layer_dims.push_back(feature_count);
    
    // Construct the autoencoder with the pre-built layer stack
    auto autoencoder = std::make_unique<Autoencoder>();
    autoencoder->init(layer_dims, all_layers, rng_.get());
    
    // Store in the models map
    models_[name] = std::move(autoencoder);
}

void Translator::forward(const Batch& b, cudaStream_t stream) {
    Autoencoder& ae = model(b);
    ae.forward(b.sparse_view(), stream);
}

void Translator::backward_and_step(const Batch& b, float lr, cudaStream_t stream) {
    Autoencoder& ae = model(b);
    ae.backward_and_step(b.sparse_view(), lr, stream);
}

Autoencoder& Translator::model(const std::string& name) {
    return *models_.at(name);
}

Autoencoder& Translator::model(const Batch& b) {
    return model(b.species_name());
}
