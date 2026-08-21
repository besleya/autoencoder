// SPDX-License-Identifier: MIT
#include "layer.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <nvtx3/nvToolsExt.h>
#include "gpu_timer.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <algorithm>

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
// Utility: compute grid/block for 1D kernel
// ============================================================================

static void get_grid_block(int n, dim3& grid, dim3& block) {
    block = dim3(256);
    grid = dim3((n + 255) / 256);
}

// ============================================================================
// Custom CUDA kernels
// ============================================================================

// Bias broadcast (column-major): z[i,j] += b[i] for all j
__global__ void kernel_add_bias(int out, int B, float* z, const float* b) {
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    if (bidx >= B) return;

    for (int i = tidx; i < out; i += blockDim.x) {
        z[i + (size_t)bidx * out] += b[i];
    }
}

// ReLU forward: a[i] = max(z[i], 0)
__global__ void kernel_relu_forward(int size, float* a, const float* z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] = fmaxf(z[idx], 0.0f);
    }
}

// ReLU backward: dz[i] *= (z[i] > 0)
__global__ void kernel_relu_backward(int size, float* dz, const float* z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dz[idx] *= (z[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// Fill vector with ones
__global__ void kernel_fill_ones(int n, float* v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        v[idx] = 1.0f;
    }
}

// Adam update: p = p - lr * (m/bc1) / (sqrt(v/bc2) + eps)
__global__ void kernel_adam_update(
    int size, float* p, float* m, float* v, const float* g,
    float lr, float beta1, float beta2, float eps, float bc1, float bc2) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g_val = g[idx];
        float m_val = beta1 * m[idx] + (1.0f - beta1) * g_val;
        float v_val = beta2 * v[idx] + (1.0f - beta2) * g_val * g_val;
        
        m[idx] = m_val;
        v[idx] = v_val;
        
        float m_hat = m_val / bc1;
        float v_hat = v_val / bc2;
        p[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

// ============================================================================
// Layer implementation
// ============================================================================

Layer::Layer(int in_dim,
             int out_dim,
             Activation activation,
             bool sparse_input,
             unsigned long seed,
             cublasHandle_t cublas_handle,
             cusparseHandle_t cusparse_handle)
    : in_dim_(in_dim),
      out_dim_(out_dim),
      activation_(activation),
      sparse_input_(sparse_input),
      cublas_handle_(cublas_handle),
      cusparse_handle_(cusparse_handle),
      d_W_(nullptr),
      d_b_(nullptr),
      d_mW_(nullptr),
      d_vW_(nullptr),
      d_mb_(nullptr),
      d_vb_(nullptr),
      d_z_(nullptr),
      d_y_(nullptr),
      d_dz_(nullptr),
      d_dW_(nullptr),
      d_db_(nullptr),
      d_grad_input_(nullptr),
      d_ones_(nullptr),
      last_batch_size_(0),
      t_(0) {
    
    // Validate inputs
    if (in_dim <= 0 || out_dim <= 0) {
        fprintf(stderr, "Layer: in_dim and out_dim must be > 0\n");
        exit(EXIT_FAILURE);
    }

    // Initialize weights with He initialization (for ReLU) or Xavier (for None)
    std::mt19937 rng(seed);
    float stddev;
    if (activation == Activation::ReLU) {
        // He init
        stddev = std::sqrt(2.0f / static_cast<float>(in_dim));
    } else {
        // Xavier init for None activation
        stddev = std::sqrt(1.0f / static_cast<float>(in_dim));
    }
    std::normal_distribution<float> nd(0.0f, stddev);

    // Generate weights on host
    std::vector<float> W_host(out_dim * in_dim);
    for (int j = 0; j < in_dim; ++j) {
        for (int i = 0; i < out_dim; ++i) {
            W_host[i + (size_t)j * out_dim] = nd(rng);
        }
    }

    // Allocate device parameters
    CUDA_CHECK(cudaMalloc(&d_W_, out_dim * in_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_W_, W_host.data(), out_dim * in_dim * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_b_, out_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b_, 0, out_dim * sizeof(float)));

    // Adam state
    CUDA_CHECK(cudaMalloc(&d_mW_, out_dim * in_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mW_, 0, out_dim * in_dim * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_vW_, out_dim * in_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_vW_, 0, out_dim * in_dim * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_mb_, out_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mb_, 0, out_dim * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_vb_, out_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_vb_, 0, out_dim * sizeof(float)));

    // Gradient buffers (allocated at construction time)
    CUDA_CHECK(cudaMalloc(&d_dW_, out_dim * in_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db_, out_dim * sizeof(float)));
}

Layer::~Layer() {
    if (d_W_) CUDA_CHECK(cudaFree(d_W_));
    if (d_b_) CUDA_CHECK(cudaFree(d_b_));
    if (d_mW_) CUDA_CHECK(cudaFree(d_mW_));
    if (d_vW_) CUDA_CHECK(cudaFree(d_vW_));
    if (d_mb_) CUDA_CHECK(cudaFree(d_mb_));
    if (d_vb_) CUDA_CHECK(cudaFree(d_vb_));
    if (d_z_) CUDA_CHECK(cudaFree(d_z_));
    if (d_y_) CUDA_CHECK(cudaFree(d_y_));
    if (d_dz_) CUDA_CHECK(cudaFree(d_dz_));
    if (d_dW_) CUDA_CHECK(cudaFree(d_dW_));
    if (d_db_) CUDA_CHECK(cudaFree(d_db_));
    if (d_grad_input_) CUDA_CHECK(cudaFree(d_grad_input_));
    if (d_ones_) CUDA_CHECK(cudaFree(d_ones_));
}

// --- Sparse-input overloads ---

const float* Layer::forward(const SparseView& x, cudaStream_t stream) {
    if (!sparse_input_) {
        fprintf(stderr, "Layer::forward(SparseView): layer is not configured for sparse input\n");
        exit(EXIT_FAILURE);
    }
    _ensure_batch_buffers(x.B, /*need_grad_input=*/false);
    _sp_forward(x, stream);
    _apply_activation_forward(x.B, stream);
    return d_y_;
}

void Layer::backward(const SparseView& x, const float* d_grad_output, cudaStream_t stream) {
    if (!sparse_input_) {
        fprintf(stderr, "Layer::backward(SparseView): layer is not configured for sparse input\n");
        exit(EXIT_FAILURE);
    }
    _sp_backward(x, d_grad_output, stream);
}

// --- Dense-input overloads ---

const float* Layer::forward(const float* d_in, int batch_size, cudaStream_t stream) {
    if (sparse_input_) {
        fprintf(stderr, "Layer::forward(dense): layer is configured for sparse input\n");
        exit(EXIT_FAILURE);
    }
    _ensure_batch_buffers(batch_size, /*need_grad_input=*/false);
    _dn_forward(d_in, batch_size, stream);
    _apply_activation_forward(batch_size, stream);
    return d_y_;
}

const float* Layer::backward(const float* d_in,
                              const float* d_grad_output,
                              bool compute_grad_input,
                              cudaStream_t stream) {
    if (sparse_input_) {
        fprintf(stderr, "Layer::backward(dense): layer is configured for sparse input\n");
        exit(EXIT_FAILURE);
    }
    _ensure_batch_buffers(last_batch_size_, compute_grad_input);
    _dn_backward(d_in, d_grad_output, compute_grad_input, stream);
    return compute_grad_input ? d_grad_input_ : nullptr;
}

// --- Optimization ---

void Layer::update(float lr, cudaStream_t stream) {
    nvtxRangePushA("Layer::update");
    GpuScopedTimer timer_total("layer.update.total", stream);
    
    ++t_;
    float bc1 = 1.0f - std::pow(0.9f, static_cast<float>(t_));
    float bc2 = 1.0f - std::pow(0.999f, static_cast<float>(t_));

    int size_w = out_dim_ * in_dim_;
    int size_b = out_dim_;

    // Update W
    {
        GpuScopedTimer timer_w("layer.update.W", stream);
        dim3 grid, block;
        get_grid_block(size_w, grid, block);
        kernel_adam_update<<<grid, block, 0, stream>>>(
            size_w, d_W_, d_mW_, d_vW_, d_dW_,
            lr, 0.9f, 0.999f, 1e-8f, bc1, bc2);
    }

    // Update b
    {
        GpuScopedTimer timer_b("layer.update.b", stream);
        dim3 grid, block;
        get_grid_block(size_b, grid, block);
        kernel_adam_update<<<grid, block, 0, stream>>>(
            size_b, d_b_, d_mb_, d_vb_, d_db_,
            lr, 0.9f, 0.999f, 1e-8f, bc1, bc2);
    }
    
    nvtxRangePop();
}

// --- Weight / bias init setters ---

void Layer::set_weights_from_host(const float* host_W) {
    // host_W is row-major: host_W[i * in_dim_ + j] = W[i,j].
    // d_W_ is column-major: d_W_[i + j * out_dim_] = W[i,j].
    // Transpose to column-major before copying to device.
    std::vector<float> col_major((size_t)out_dim_ * in_dim_);
    for (int i = 0; i < out_dim_; ++i) {
        for (int j = 0; j < in_dim_; ++j) {
            col_major[i + (size_t)j * out_dim_] = host_W[(size_t)i * in_dim_ + j];
        }
    }
    CUDA_CHECK(cudaMemcpy(d_W_, col_major.data(),
                          (size_t)out_dim_ * in_dim_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    reset_optimizer_state();
}

void Layer::set_biases_from_host(const float* host_b) {
    CUDA_CHECK(cudaMemcpy(d_b_, host_b, (size_t)out_dim_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    reset_optimizer_state();
}

void Layer::reset_optimizer_state() {
    size_t bytes_w = (size_t)out_dim_ * in_dim_ * sizeof(float);
    size_t bytes_b = (size_t)out_dim_ * sizeof(float);
    CUDA_CHECK(cudaMemset(d_mW_, 0, bytes_w));
    CUDA_CHECK(cudaMemset(d_vW_, 0, bytes_w));
    CUDA_CHECK(cudaMemset(d_mb_, 0, bytes_b));
    CUDA_CHECK(cudaMemset(d_vb_, 0, bytes_b));
    t_ = 0;
}

// --- Accessors ---

int Layer::in_dim() const noexcept { return in_dim_; }
int Layer::out_dim() const noexcept { return out_dim_; }
Layer::Activation Layer::activation() const noexcept { return activation_; }
bool Layer::sparse_input() const noexcept { return sparse_input_; }
const float* Layer::output() const noexcept { return d_y_; }
int Layer::last_batch_size() const noexcept { return last_batch_size_; }
const float* Layer::weights() const noexcept { return d_W_; }
const float* Layer::bias() const noexcept { return d_b_; }
int Layer::timestep() const noexcept { return t_; }

// --- Private helpers ---

void Layer::_ensure_batch_buffers(int batch_size, bool need_grad_input) {
    if (batch_size == last_batch_size_ && (!need_grad_input || d_grad_input_ != nullptr)) {
        return;  // No-op: buffers already sized appropriately
    }

    // Free existing per-batch buffers
    if (d_z_) CUDA_CHECK(cudaFree(d_z_));
    if (d_y_) CUDA_CHECK(cudaFree(d_y_));
    if (d_dz_) CUDA_CHECK(cudaFree(d_dz_));
    if (d_ones_) CUDA_CHECK(cudaFree(d_ones_));
    if (d_grad_input_) CUDA_CHECK(cudaFree(d_grad_input_));

    // Allocate new buffers
    int z_size = out_dim_ * batch_size;
    CUDA_CHECK(cudaMalloc(&d_z_, z_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_, z_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dz_, z_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_ones_, batch_size * sizeof(float)));
    dim3 grid, block;
    get_grid_block(batch_size, grid, block);
    kernel_fill_ones<<<grid, block>>>(batch_size, d_ones_);

    // Allocate grad_input only if needed
    if (need_grad_input) {
        int in_size = in_dim_ * batch_size;
        CUDA_CHECK(cudaMalloc(&d_grad_input_, in_size * sizeof(float)));
    } else {
        d_grad_input_ = nullptr;
    }

    last_batch_size_ = batch_size;
}

void Layer::_sp_forward(const SparseView& x, cudaStream_t stream) {
    nvtxRangePushA("Layer::_sp_forward");
    GpuScopedTimer timer_total("layer.sp_fwd.total", stream);
    
    cublasHandle_t blas_h = cublas_handle_;
    cusparseHandle_t sparse_h = cusparse_handle_;

    CUBLAS_CHECK(cublasSetStream(blas_h, stream));
    CUSPARSE_CHECK(cusparseSetStream(sparse_h, stream));

    float one_f = 1.0f, zero_f = 0.0f;

    // Create sparse descriptor for x (CSC format)
    cusparseSpMatDescr_t matA;
    CUSPARSE_CHECK(cusparseCreateCsc(
        &matA, in_dim_, x.B, x.nnz,
        (void*)x.d_col_ptr, (void*)x.d_row_idx, (void*)x.d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));

    // Create dense descriptor for W (out x in, col-major)
    cusparseDnMatDescr_t matB;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &matB, out_dim_, in_dim_, out_dim_,
        d_W_, CUDA_R_32F, CUSPARSE_ORDER_COL));

    // Create dense descriptor for z (B x out, row-major result)
    cusparseDnMatDescr_t matC;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &matC, x.B, out_dim_, out_dim_,
        d_z_, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // z = W @ x (with appropriate transposes)
    // matA is CSC(in_dim, B), with OpA=TRANSPOSE → (B, in_dim)
    // matB is (out, in) with OpB=TRANSPOSE → (in, out)
    // matC = (B, out)
    size_t workspace_size = 0;
    void* workspace = nullptr;
    {
        GpuScopedTimer timer_ws("layer.sp_fwd.workspace_alloc", stream);
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            sparse_h, CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE, &one_f, matA, matB,
            &zero_f, matC, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1,
            &workspace_size));

        if (workspace_size > 0) {
            CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
        }
    }

    {
        GpuScopedTimer timer_spmm("layer.sp_fwd.spmm", stream);
        CUSPARSE_CHECK(cusparseSpMM(
            sparse_h, CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE, &one_f, matA, matB,
            &zero_f, matC, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, workspace));
    }

    if (workspace) {
        GpuScopedTimer timer_free("layer.sp_fwd.workspace_free", stream);
        CUDA_CHECK(cudaFree(workspace));
    }
    
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);

    // Add bias
    {
        GpuScopedTimer timer_bias("layer.sp_fwd.bias", stream);
        dim3 grid, block;
        get_grid_block(x.B, grid, block);
        kernel_add_bias<<<grid, block, 0, stream>>>(out_dim_, x.B, d_z_, d_b_);
    }
    
    nvtxRangePop();
}

void Layer::_dn_forward(const float* d_in, int batch_size, cudaStream_t stream) {
    nvtxRangePushA("Layer::_dn_forward");
    GpuScopedTimer timer_total("layer.dn_fwd.total", stream);
    
    cublasHandle_t blas_h = cublas_handle_;
    CUBLAS_CHECK(cublasSetStream(blas_h, stream));

    // z = W @ in + b
    // W: (out, in) col-major
    // in: (in, batch_size) col-major
    // z: (out, batch_size) col-major
    float one_f = 1.0f, zero_f = 0.0f;
    {
        GpuScopedTimer timer_gemm("layer.dn_fwd.gemm", stream);
        CUBLAS_CHECK(cublasSgemm(
            blas_h, CUBLAS_OP_N, CUBLAS_OP_N,
            out_dim_, batch_size, in_dim_,
            &one_f,
            d_W_, out_dim_,
            d_in, in_dim_,
            &zero_f,
            d_z_, out_dim_));
    }

    // Add bias
    {
        GpuScopedTimer timer_bias("layer.dn_fwd.bias", stream);
        dim3 grid, block;
        get_grid_block(batch_size, grid, block);
        kernel_add_bias<<<grid, block, 0, stream>>>(out_dim_, batch_size, d_z_, d_b_);
    }
    
    nvtxRangePop();
}

void Layer::_sp_backward(const SparseView& x, const float* d_grad_output, cudaStream_t stream) {
    nvtxRangePushA("Layer::_sp_backward");
    GpuScopedTimer timer_total("layer.sp_bwd.total", stream);
    
    cublasHandle_t blas_h = cublas_handle_;
    cusparseHandle_t sparse_h = cusparse_handle_;

    CUBLAS_CHECK(cublasSetStream(blas_h, stream));
    CUSPARSE_CHECK(cusparseSetStream(sparse_h, stream));

    // Apply activation backward: dz = d_grad_output * activation'(z)
    _apply_activation_backward(x.B, stream, d_grad_output);

    float one_f = 1.0f, zero_f = 0.0f;

    // Compute dW = x @ dz^T (sparse input, dense gradient)
    // x: CSC(in_dim, B), with OpA=NON_TRANSPOSE → (in_dim, B)
    // dz: (B, out_dim) row-major, with OpB=NON_TRANSPOSE → (B, out_dim)
    // dW: (in_dim, out_dim) result (col-major compatible)
    cusparseSpMatDescr_t matA;
    CUSPARSE_CHECK(cusparseCreateCsc(
        &matA, in_dim_, x.B, x.nnz,
        (void*)x.d_col_ptr, (void*)x.d_row_idx, (void*)x.d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));

    cusparseDnMatDescr_t matB;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &matB, x.B, out_dim_, out_dim_,
        d_dz_, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    cusparseDnMatDescr_t matC;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &matC, in_dim_, out_dim_, out_dim_,
        d_dW_, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    size_t workspace_size = 0;
    void* workspace = nullptr;
    {
        GpuScopedTimer timer_ws("layer.sp_bwd.workspace_alloc", stream);
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            sparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &one_f, matA, matB,
            &zero_f, matC, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1,
            &workspace_size));

        if (workspace_size > 0) {
            CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
        }
    }

    {
        GpuScopedTimer timer_spmm("layer.sp_bwd.spmm", stream);
        CUSPARSE_CHECK(cusparseSpMM(
            sparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &one_f, matA, matB,
            &zero_f, matC, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, workspace));
    }

    if (workspace) CUDA_CHECK(cudaFree(workspace));
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);

    // Compute db = dz @ ones (row-wise sum)
    // dz: (B, out_dim) row-major, but we need to transpose it for GEMV
    // Actually, we need to sum over rows, so: db = dz^T @ ones = (out, B) @ (B,) = (out,)
    // But d_dz is stored as (out, B) col-major after activation backward
    {
        GpuScopedTimer timer_dgemv("layer.sp_bwd.dgemv_bias", stream);
        CUBLAS_CHECK(cublasSgemv(
            blas_h, CUBLAS_OP_N,
            out_dim_, x.B,
            &one_f,
            d_dz_, out_dim_,
            d_ones_, 1,
            &zero_f,
            d_db_, 1));
    }
    
    nvtxRangePop();
}

void Layer::_dn_backward(const float* d_in,
                         const float* d_grad_output,
                         bool compute_grad_input,
                         cudaStream_t stream) {
    nvtxRangePushA("Layer::_dn_backward");
    GpuScopedTimer timer_total("layer.dn_bwd.total", stream);
    
    cublasHandle_t blas_h = cublas_handle_;
    CUBLAS_CHECK(cublasSetStream(blas_h, stream));

    int batch_size = last_batch_size_;

    // Apply activation backward: dz = d_grad_output * activation'(z)
    _apply_activation_backward(batch_size, stream, d_grad_output);

    float one_f = 1.0f, zero_f = 0.0f;

    // Compute dW = dz @ in^T
    // dz: (out, batch_size) col-major
    // in: (in, batch_size) col-major, transposed to (batch_size, in)
    // dW: (out, in) col-major result
    {
        GpuScopedTimer timer_dw("layer.dn_bwd.gemm_dW", stream);
        CUBLAS_CHECK(cublasSgemm(
            blas_h, CUBLAS_OP_N, CUBLAS_OP_T,
            out_dim_, in_dim_, batch_size,
            &one_f,
            d_dz_, out_dim_,
            d_in, in_dim_,
            &zero_f,
            d_dW_, out_dim_));
    }

    // Compute db = dz @ ones (row-wise sum)
    {
        GpuScopedTimer timer_db("layer.dn_bwd.gemv_db", stream);
        CUBLAS_CHECK(cublasSgemv(
            blas_h, CUBLAS_OP_N,
            out_dim_, batch_size,
            &one_f,
            d_dz_, out_dim_,
            d_ones_, 1,
            &zero_f,
            d_db_, 1));
    }

    // Compute grad_input = W^T @ dz
    if (compute_grad_input) {
        // W^T: (in, out), dz: (out, batch_size) col-major
        // grad_input: (in, batch_size) col-major result
        GpuScopedTimer timer_dinput("layer.dn_bwd.gemm_dInput", stream);
        CUBLAS_CHECK(cublasSgemm(
            blas_h, CUBLAS_OP_T, CUBLAS_OP_N,
            in_dim_, batch_size, out_dim_,
            &one_f,
            d_W_, out_dim_,
            d_dz_, out_dim_,
            &zero_f,
            d_grad_input_, in_dim_));
    }
    
    nvtxRangePop();
}

void Layer::_apply_activation_forward(int batch_size, cudaStream_t stream) {
    if (activation_ == Activation::None) {
        // y = z (copy)
        GpuScopedTimer timer("layer.dn_fwd.activation", stream);
        int size = out_dim_ * batch_size;
        CUDA_CHECK(cudaMemcpyAsync(d_y_, d_z_, size * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
    } else {  // ReLU
        // y = max(z, 0)
        GpuScopedTimer timer("layer.dn_fwd.activation", stream);
        int size = out_dim_ * batch_size;
        dim3 grid, block;
        get_grid_block(size, grid, block);
        kernel_relu_forward<<<grid, block, 0, stream>>>(size, d_y_, d_z_);
    }
}

void Layer::_apply_activation_backward(int batch_size, cudaStream_t stream, const float* d_grad_output) {
    if (activation_ == Activation::None) {
        // dz = d_grad_output (copy)
        GpuScopedTimer timer("layer.dn_bwd.activation", stream);
        int size = out_dim_ * batch_size;
        CUDA_CHECK(cudaMemcpyAsync(d_dz_, d_grad_output, size * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
    } else {  // ReLU
        // dz = d_grad_output * (z > 0 ? 1 : 0)
        GpuScopedTimer timer("layer.dn_bwd.activation", stream);
        // First copy
        int size = out_dim_ * batch_size;
        CUDA_CHECK(cudaMemcpyAsync(d_dz_, d_grad_output, size * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
        // Then gate
        dim3 grid, block;
        get_grid_block(size, grid, block);
        kernel_relu_backward<<<grid, block, 0, stream>>>(size, d_dz_, d_z_);
    }
}
