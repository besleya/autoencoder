// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

// Forward declarations
class Batch;
class Autoencoder;
class Layer;

// Translator: Master model coordinator for multi-species autoencoders with shared layers.
//
// Holds one shared layer stack (encoder + decoder) that is aliased into every
// per-species autoencoder. When training, the Translator routes forward/backward
// calls to the correct species' model based on batch species identity.
//
// Layer layout (sandwich):
//   [species encoder] → [shared encoder] → [shared decoder] → [species decoder]
//
// The shared middle is one shared_ptr<Layer> sequence, referenced by every
// species' autoencoder. The species-specific ends are private to each species.
class Translator {
public:
    // Constructor: builds the shared encoder and decoder layers immediately.
    // The shared encoder takes species_dims.back() as input and outputs
    // shared_dims.back() (the bottleneck). The shared decoder mirrors this.
    // Parameters:
    //   n_shared_layers: number of layers in shared encoder/decoder pair
    //   n_species_layers: number of layers in species encoder/decoder pair (per species)
    //   shared_dims: inner dimensions of shared encoder; size must be n_shared_layers
    //   species_dims: inner dimensions of species encoder; size must be n_species_layers
    //   rng: random number generator for weight initialization (stored as reference)
    Translator(int n_shared_layers,
               int n_species_layers,
               const std::vector<int>& shared_dims,
               const std::vector<int>& species_dims,
               std::mt19937& rng);

    ~Translator();

    // Non-copyable; default move semantics.
    Translator(const Translator&) = delete;
    Translator& operator=(const Translator&) = delete;
    Translator(Translator&&) = default;
    Translator& operator=(Translator&&) = default;

    // Register a new species with the given name and feature count.
    // Constructs species-specific encoder and decoder layers, then builds
    // an Autoencoder with the full sandwich: [species_enc] + [shared_enc] +
    // [shared_dec] + [species_dec].
    // Throws std::invalid_argument if the species name is already registered.
    void species(const std::string& name, int feature_count);

    // Forward pass for the batch. Dispatches to the autoencoder corresponding
    // to b.species_name(). Launches kernels on the provided stream.
    // Throws std::out_of_range if the species is unknown.
    void forward(const Batch& b, cudaStream_t stream);

    // Backward pass with Adam update for the batch. Dispatches to the autoencoder
    // corresponding to b.species_name(). Launches kernels on the provided stream.
    // Throws std::out_of_range if the species is unknown.
    void backward_and_step(const Batch& b, float lr, cudaStream_t stream);

    // Accessors: retrieve the autoencoder for a given species name or batch.
    // Throws std::out_of_range if the species is unknown.
    Autoencoder& model(const std::string& name);
    Autoencoder& model(const Batch& b);

private:
    // Shared layer stacks (one copy, referenced by all species' autoencoders).
    std::vector<std::shared_ptr<Layer>> shared_encoder_;
    std::vector<std::shared_ptr<Layer>> shared_decoder_;

    // Constructor parameters (stored for species() method).
    std::vector<int> species_dims_;
    std::vector<int> shared_dims_;
    std::reference_wrapper<std::mt19937> rng_;

    // Per-species autoencoders.
    std::unordered_map<std::string, std::unique_ptr<Autoencoder>> models_;

    // cublas and cusparse handles (owned by Translator, passed to Layer constructors).
    cublasHandle_t cublas_handle_;
    cusparseHandle_t cusparse_handle_;

    // Helper: build shared encoder and decoder layers.
    void build_shared_layers_(int n_shared_layers,
                              const std::vector<int>& shared_dims);
};
