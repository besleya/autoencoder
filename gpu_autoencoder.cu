// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <nvtx3/nvToolsExt.h>

#include "gpu_autoencoder.h"
#include "gpu_timer.h"


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
// Custom CUDA kernels
// ============================================================================

// Compute loss and initialize loss gradient:
//   loss_acc = ||a_L||_F^2
//   d_grad_loss = 2 * a_L / (B * d0) (for all entries)
//   For each sparse (r, j, v): d_grad_loss(r,j) = 2 * (a_L(r,j) - v) / (B * d0)
//                              loss_acc += v^2 - 2*a_L(r,j)*v
// Then loss = loss_acc / (d0 * B)
//
// This kernel iterates through columns; each thread block handles one column.
__global__ void kernel_sparse_loss_and_grad(
    int d0, int B, float* d_grad_loss, float* loss_acc, 
    const float* a_L, const int32_t* col_ptr, const int32_t* row_idx,
    const float* values) {
    
    int j = blockIdx.x;
    if (j >= B) return;
    
    int start = col_ptr[j];
    int end = col_ptr[j + 1];
    
    for (int k = start + threadIdx.x; k < end; k += blockDim.x) {
        int r = row_idx[k];
        float v = values[k];
        float a_val = a_L[r + (size_t)j * d0];
        
        // d_grad_loss(r,j) = 2 * (a_L(r,j) - v) / (B * d0)
        d_grad_loss[r + (size_t)j * d0] = 2.0f * (a_val - v) / (static_cast<float>(B) * static_cast<float>(d0));
        
        // Accumulate loss correction
        float correction = v * v - 2.0f * a_val * v;
        atomicAdd(loss_acc, correction);
    }
}

// Accumulate per-batch loss into a device-side epoch sum.
// Computes: *d_epoch_sum += (*d_dot + *d_loss) / (d0 * B)
// Single-thread kernel (writing one scalar); launched once per batch.
__global__ void kernel_accumulate_loss(const float* d_dot,
                                       const float* d_loss,
                                       float* d_epoch_sum,
                                       int d0, int B) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float denom = static_cast<float>(d0) * static_cast<float>(B);
        *d_epoch_sum += (*d_dot + *d_loss) / denom;
    }
}

// ============================================================================
// GpuAutoencoder implementation
// ============================================================================

GpuAutoencoder::GpuAutoencoder()
    : num_l_(0), batch_size_(0), initialized_buffers_(false),
      d_grad_loss_(nullptr), d_loss_(nullptr),
      d_dot_(nullptr), d_epoch_loss_sum_(nullptr),
      cublas_handle_(nullptr), cusparse_handle_(nullptr) {}

GpuAutoencoder::~GpuAutoencoder() {
    deallocate_buffers_();
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
    if (cusparse_handle_) {
        cusparseDestroy(cusparse_handle_);
    }
}

void GpuAutoencoder::init(const std::vector<int>& layer_dims,
                          std::mt19937& rng) {
    dims_ = layer_dims;
    num_l_ = static_cast<int>(dims_.size()) - 1;
    
    if (num_l_ <= 0) {
        fprintf(stderr, "Invalid layer dims: must have at least 2 dims\n");
        exit(EXIT_FAILURE);
    }
    
    // Create handles if not yet created
    if (!cublas_handle_) {
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    }
    if (!cusparse_handle_) {
        CUSPARSE_CHECK(cusparseCreate(&cusparse_handle_));
    }
    
    // Construct all layers. First layer has sparse input, others dense.
    // Hidden layers (all except last) use ReLU, output layer uses None.
    layers_.resize(num_l_);
    for (int l = 0; l < num_l_; ++l) {
        int in_dim = dims_[l];
        int out_dim = dims_[l + 1];
        bool sparse_input = (l == 0);
        Layer::Activation activation = (l < num_l_ - 1) 
                                        ? Layer::Activation::ReLU 
                                        : Layer::Activation::None;
        
        // Generate seed from rng to match CPU autoencoder's RNG consumption order
        unsigned long seed = rng();
        
        layers_[l] = construct_layer_(in_dim, out_dim, activation, sparse_input, seed);
    }
    
    // Allocate loss scalars
    CUDA_CHECK(cudaMalloc(&d_loss_, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dot_, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_epoch_loss_sum_, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_epoch_loss_sum_, 0, sizeof(float)));
}

void GpuAutoencoder::init(const std::vector<int>& layer_dims,
                          const std::vector<std::shared_ptr<Layer>>& layers_in,
                          std::mt19937& rng) {
    dims_ = layer_dims;
    num_l_ = static_cast<int>(dims_.size()) - 1;
    
    if (num_l_ <= 0) {
        fprintf(stderr, "Invalid layer dims: must have at least 2 dims\n");
        exit(EXIT_FAILURE);
    }
    
    if (layers_in.size() != static_cast<size_t>(num_l_)) {
        fprintf(stderr, "layers_in.size() (%zu) must equal dims.size() - 1 (%d)\n",
                layers_in.size(), num_l_);
        exit(EXIT_FAILURE);
    }
    
    // Create handles if not yet created
    if (!cublas_handle_) {
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    }
    if (!cusparse_handle_) {
        CUSPARSE_CHECK(cusparseCreate(&cusparse_handle_));
    }
    
    // Validate and adopt caller-supplied layers, or construct missing ones
    layers_.resize(num_l_);
    for (int l = 0; l < num_l_; ++l) {
        int in_dim = dims_[l];
        int out_dim = dims_[l + 1];
        bool expected_sparse_input = (l == 0);
        Layer::Activation expected_activation = (l < num_l_ - 1)
                                                 ? Layer::Activation::ReLU
                                                 : Layer::Activation::None;
        
        if (layers_in[l]) {
            // Validate the supplied layer
            auto& layer = layers_in[l];
            if (layer->in_dim() != in_dim) {
                fprintf(stderr, "Layer %d: in_dim mismatch: expected %d, got %d\n",
                        l, in_dim, layer->in_dim());
                exit(EXIT_FAILURE);
            }
            if (layer->out_dim() != out_dim) {
                fprintf(stderr, "Layer %d: out_dim mismatch: expected %d, got %d\n",
                        l, out_dim, layer->out_dim());
                exit(EXIT_FAILURE);
            }
            if (layer->sparse_input() != expected_sparse_input) {
                fprintf(stderr, "Layer %d: sparse_input mismatch: expected %d, got %d\n",
                        l, expected_sparse_input, layer->sparse_input());
                exit(EXIT_FAILURE);
            }
            layers_[l] = layer;
        } else {
            // Construct a new layer
            unsigned long seed = rng();
            layers_[l] = construct_layer_(in_dim, out_dim, expected_activation,
                                          expected_sparse_input, seed);
        }
    }
    
    // Allocate loss scalars
    CUDA_CHECK(cudaMalloc(&d_loss_, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dot_, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_epoch_loss_sum_, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_epoch_loss_sum_, 0, sizeof(float)));
}

std::shared_ptr<Layer> GpuAutoencoder::construct_layer_(
    int in_dim, int out_dim, Layer::Activation activation,
    bool sparse_input, unsigned long seed) {
    
    return std::make_shared<Layer>(in_dim, out_dim, activation, sparse_input,
                                    seed, cublas_handle_, cusparse_handle_);
}

void GpuAutoencoder::allocate_buffers_() {
    if (initialized_buffers_) return;
    
    // Allocate loss gradient buffer: (dims_[num_l_] × batch_size_) column-major
    int out_dim = dims_[num_l_];
    CUDA_CHECK(cudaMalloc(&d_grad_loss_, out_dim * batch_size_ * sizeof(float)));
    
    initialized_buffers_ = true;
}

void GpuAutoencoder::deallocate_buffers_() {
    if (d_grad_loss_) {
        CUDA_CHECK(cudaFree(d_grad_loss_));
        d_grad_loss_ = nullptr;
    }
    if (d_loss_) {
        CUDA_CHECK(cudaFree(d_loss_));
        d_loss_ = nullptr;
    }
    if (d_dot_) {
        CUDA_CHECK(cudaFree(d_dot_));
        d_dot_ = nullptr;
    }
    if (d_epoch_loss_sum_) {
        CUDA_CHECK(cudaFree(d_epoch_loss_sum_));
        d_epoch_loss_sum_ = nullptr;
    }
    initialized_buffers_ = false;
}

void GpuAutoencoder::forward(const SparseView& x, cudaStream_t stream) {
    nvtxRangePushA("GpuAutoencoder::forward");
    GpuScopedTimer timer_total("ae.forward.total", stream);
    
    // Set batch size on first forward
    if (batch_size_ == 0) {
        batch_size_ = x.B;
        allocate_buffers_();
    }
    
    // Layer 0: sparse input
    {
        char layer_name[64];
        snprintf(layer_name, sizeof(layer_name), "ae.forward.layer[0]");
        GpuScopedTimer timer_layer(layer_name, stream);
        const float* output = layers_[0]->forward(x, stream);
        (void)output;  // Use output to avoid unused variable warning
    }
    
    // Layers 1..L-1: dense input, using previous layer's output
    const float* output = layers_[0]->output();
    for (int i = 1; i < num_l_; ++i) {
        char layer_name[64];
        snprintf(layer_name, sizeof(layer_name), "ae.forward.layer[%d]", i);
        GpuScopedTimer timer_layer(layer_name, stream);
        output = layers_[i]->forward(output, x.B, stream);
    }
    nvtxRangePop();
}

void GpuAutoencoder::backward_and_step(const SparseView& x, float lr, cudaStream_t stream) {
    nvtxRangePushA("GpuAutoencoder::backward_and_step");
    GpuScopedTimer timer_total("ae.backward.total", stream);
    
    int d0 = dims_[0];
    int dL = dims_[num_l_];
    int B = x.B;
    
    // Get final layer output (reconstruction)
    const float* a_L = layers_[num_l_ - 1]->output();
    
    // Compute loss: ||a_L - x||_F^2 / (d0 * B), accumulated on device.
    // Strategy: d_dot_ = ||a_L||_F^2 (via cublasSdot in DEVICE pointer mode),
    //           d_loss_ += sum over sparse (r, j, v) of v^2 - 2*a_L(r,j)*v
    //           d_epoch_loss_sum_ += (d_dot_ + d_loss_) / (d0 * B)
    // No host sync; loss is read once per epoch via read_epoch_loss().

    // Compute ||a_L||_F^2 into device scalar d_dot_ (DEVICE pointer mode).
    {
        GpuScopedTimer timer_dot("ae.bwd.loss_dot", stream);
        CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream));
        CUBLAS_CHECK(cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE));
        CUBLAS_CHECK(cublasSdot(cublas_handle_, dL * B, a_L, 1, a_L, 1, d_dot_));
        CUBLAS_CHECK(cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_HOST));
    }
    
    // Initialize d_grad_loss_ = 2 * a_L / (B * d0) for all entries
    {
        int size = dL * B;
        {
            GpuScopedTimer timer_copy("ae.bwd.grad_init_copy", stream);
            CUDA_CHECK(cudaMemcpyAsync(d_grad_loss_, a_L, size * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
        }
        
        // Scale in-place: multiply each element by 2 / (B * d0)
        {
            GpuScopedTimer timer_scal("ae.bwd.grad_init_scal", stream);
            float scale = 2.0f / (static_cast<float>(B) * static_cast<float>(d0));
            CUBLAS_CHECK(cublasSscal(cublas_handle_, size, &scale, d_grad_loss_, 1));
        }
    }
    
    // Sparse correction kernel: for each sparse (r, j, v):
    //   d_grad_loss_(r,j) = 2 * (a_L(r,j) - v) / (B * d0)
    //   d_loss_ += v^2 - 2*a_L(r,j)*v
    {
        CUDA_CHECK(cudaMemsetAsync(d_loss_, 0, sizeof(float), stream));
        
        {
            GpuScopedTimer timer_kernel("ae.bwd.loss_grad_kernel", stream);
            // Launch one block per column
            kernel_sparse_loss_and_grad<<<B, 256, 0, stream>>>(
                d0, B, d_grad_loss_, d_loss_,
                a_L, x.d_col_ptr, x.d_row_idx, x.d_values);
        }
        
        // Accumulate this batch's loss into the device-side epoch sum.
        // No D2H, no host sync.
        {
            GpuScopedTimer timer_accum("ae.bwd.loss_accumulate", stream);
            kernel_accumulate_loss<<<1, 1, 0, stream>>>(
                d_dot_, d_loss_, d_epoch_loss_sum_, d0, B);
        }
    }
    
    // Backward pass: layers in reverse order
    const float* grad = d_grad_loss_;
    
    // Layers L-1 down to 1: dense backward
    for (int i = num_l_ - 1; i >= 1; --i) {
        // Get the input to this layer (output of previous layer)
        const float* layer_input = layers_[i - 1]->output();
        
        // Compute gradient with respect to input (needed for previous layer)
        bool compute_grad_input = true;
        {
            char timer_name[64];
            snprintf(timer_name, sizeof(timer_name), "ae.bwd.layer[%d].back", i);
            GpuScopedTimer timer_back(timer_name, stream);
            grad = layers_[i]->backward(layer_input, grad, compute_grad_input, stream);
        }
    }
    
    // Layer 0: sparse backward (no grad_input needed/returned)
    {
        GpuScopedTimer timer_layer0("ae.bwd.layer[0].back", stream);
        layers_[0]->backward(x, grad, stream);
    }
    
    // Adam update on all layers
    for (size_t i = 0; i < layers_.size(); ++i) {
        char timer_name[64];
        snprintf(timer_name, sizeof(timer_name), "ae.bwd.layer[%zu].update", i);
        GpuScopedTimer timer_update(timer_name, stream);
        layers_[i]->update(lr, stream);
    }
    nvtxRangePop();
}

void GpuAutoencoder::reset_epoch_loss() {
    CUDA_CHECK(cudaMemset(d_epoch_loss_sum_, 0, sizeof(float)));
}

float GpuAutoencoder::read_epoch_loss(int num_batches) {
    float epoch_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&epoch_sum, d_epoch_loss_sum_, sizeof(float),
                          cudaMemcpyDeviceToHost));
    return (num_batches > 0) ? epoch_sum / static_cast<float>(num_batches) : 0.0f;
}

// --- Public accessors ---

int GpuAutoencoder::num_layers() const noexcept {
    return num_l_;
}

std::shared_ptr<Layer> GpuAutoencoder::layer(int i) const {
    if (i < 0 || i >= num_l_) {
        fprintf(stderr, "layer(%d): index out of range [0, %d)\n", i, num_l_);
        exit(EXIT_FAILURE);
    }
    return layers_[i];
}

const std::vector<std::shared_ptr<Layer>>& GpuAutoencoder::layers() const noexcept {
    return layers_;
}

