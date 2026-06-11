// SPDX-License-Identifier: MIT
#include "gpu_autoencoder.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cmath>
#include <cstdio>
#include <cstring>

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

// Bias broadcast: z[i,j] += b[i] for all j (column-major, ld=out)
__global__ void kernel_add_bias(int out, int B, float* z, const float* b) {
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    if (bidx >= B) return;

    for (int i = tidx; i < out; i += blockDim.x) {
        z[i + (size_t)bidx * out] += b[i];
    }
}

// Bias broadcast for row-major: z[bidx, i] += b[i] where z is (B, out) row-major with ld=out
__global__ void kernel_add_bias_rowmajor(int out, int B, float* z, const float* b) {
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    if (bidx >= B) return;

    for (int i = tidx; i < out; i += blockDim.x) {
        z[bidx * out + i] += b[i];
    }
}

// ReLU forward: a[i,j] = max(z[i,j], 0)
__global__ void kernel_relu_forward(int size, float* a, const float* z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] = fmaxf(z[idx], 0.0f);
    }
}

// ReLU backward: dz[i,j] *= (z[i,j] > 0)
__global__ void kernel_relu_backward(int size, float* dz, const float* z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dz[idx] *= (z[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// Initialize ones vector
__global__ void kernel_fill_ones(int n, float* v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        v[idx] = 1.0f;
    }
}

// Compute loss and initialize dz[L-1]:
//   loss_acc += ||a_L||_F^2
//   For each sparse (r, j, v): dz[L-1](r,j) = (a_L(r,j) - v) / B
//                              loss_acc -= 2*a_L(r,j)*v + v^2
// Then loss = loss_acc / (d0 * B)
//
// This kernel iterates through columns; each thread block handles one column.
__global__ void kernel_sparse_loss_and_dz_update(
    int d0, int B, float* dz_L, float* loss_acc, 
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
        
        // dz[L-1](r,j) = (a_L(r,j) - v) / B
        dz_L[r + (size_t)j * d0] = (a_val - v) / B;
        
        // Accumulate loss correction
        float correction = v * v - 2.0f * a_val * v;
        atomicAdd(loss_acc, correction);
    }
}

// Adam update for a single parameter:
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g * g
//   p = p - lr_t * m / (sqrt(v) + eps)
__global__ void kernel_adam_update(
    int size, float* p, float* m, float* v, const float* g,
    float lr_t, float beta1, float beta2, float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g_val = g[idx];
        float m_val = beta1 * m[idx] + (1.0f - beta1) * g_val;
        float v_val = beta2 * v[idx] + (1.0f - beta2) * g_val * g_val;
        
        m[idx] = m_val;
        v[idx] = v_val;
        
        p[idx] -= lr_t * m_val / (sqrtf(v_val) + eps);
    }
}

// ============================================================================
// GpuAutoencoder implementation
// ============================================================================

GpuAutoencoder::GpuAutoencoder()
    : num_l(0), batch_size(0), initialized_buffers(false),
      d_loss(nullptr), d_ones(nullptr), t(0),
      cublas_handle(nullptr), cusparse_handle(nullptr) {}

GpuAutoencoder::~GpuAutoencoder() {
    deallocate_buffers();
    if (cublas_handle) {
        cublasDestroy((cublasHandle_t)cublas_handle);
    }
    if (cusparse_handle) {
        cusparseDestroy((cusparseHandle_t)cusparse_handle);
    }
}

void GpuAutoencoder::init(const std::vector<int>& layer_dims,
                          std::mt19937& rng) {
    dims = layer_dims;
    num_l = static_cast<int>(dims.size()) - 1;
    
    if (num_l <= 0) {
        fprintf(stderr, "Invalid layer dims: must have at least 2 dims\n");
        exit(EXIT_FAILURE);
    }
    
    // Create handles if not yet created
    if (!cublas_handle) {
        CUBLAS_CHECK(cublasCreate((cublasHandle_t*)&cublas_handle));
    }
    if (!cusparse_handle) {
        CUSPARSE_CHECK(cusparseCreate((cusparseHandle_t*)&cusparse_handle));
    }
    
    // Resize parameter vectors
    d_W.resize(num_l);
    d_b.resize(num_l);
    d_mW.resize(num_l);
    d_vW.resize(num_l);
    d_mb.resize(num_l);
    d_vb.resize(num_l);
    d_dW.resize(num_l);
    d_db.resize(num_l);
    d_dz.resize(num_l);
    
    d_a.resize(num_l + 1);
    d_z.resize(num_l);
    
    // Initialize parameters on host, then copy to device
    for (int l = 0; l < num_l; ++l) {
        int in = dims[l];
        int out = dims[l + 1];
        
        float stddev = std::sqrt(2.0f / static_cast<float>(in));
        std::normal_distribution<float> nd(0.0f, stddev);
        
        // Allocate and initialize W[l] on host
        std::vector<float> W_host(out * in);
        for (int j = 0; j < in; ++j) {
            for (int i = 0; i < out; ++i) {
                W_host[i + (size_t)j * out] = nd(rng);
            }
        }
        
        // Allocate on device and copy
        CUDA_CHECK(cudaMalloc(&d_W[l], out * in * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_W[l], W_host.data(), out * in * sizeof(float),
                              cudaMemcpyHostToDevice));
        
        // Allocate and initialize b[l] = zeros
        CUDA_CHECK(cudaMalloc(&d_b[l], out * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_b[l], 0, out * sizeof(float)));
        
        // Allocate and initialize moment buffers (zeros)
        CUDA_CHECK(cudaMalloc(&d_mW[l], out * in * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_mW[l], 0, out * in * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_vW[l], out * in * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_vW[l], 0, out * in * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_mb[l], out * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_mb[l], 0, out * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_vb[l], out * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_vb[l], 0, out * sizeof(float)));
        
        // Allocate gradient buffers (will be filled during backward)
        CUDA_CHECK(cudaMalloc(&d_dW[l], out * in * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_db[l], out * sizeof(float)));
        // d_dz[l] will be allocated in allocate_buffers() once batch_size is known
    }
    
    // Allocate activation buffers (will be resized if batch_size changes)
    // For now, just set pointers to nullptr; they'll be allocated in allocate_buffers()
    
    // Allocate loss scalar
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    
    t = 0;
}

void GpuAutoencoder::allocate_buffers() {
    if (initialized_buffers) return;
    
    // Allocate activation buffers based on current batch_size and dims
    for (int l = 0; l <= num_l; ++l) {
        int size = dims[l] * batch_size;
        CUDA_CHECK(cudaMalloc(&d_a[l], size * sizeof(float)));
    }
    
    // Allocate pre-activation buffers
    for (int l = 0; l < num_l; ++l) {
        int size = dims[l + 1] * batch_size;
        CUDA_CHECK(cudaMalloc(&d_z[l], size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dz[l], size * sizeof(float)));
    }
    
    // Allocate ones vector for bias sum
    CUDA_CHECK(cudaMalloc(&d_ones, batch_size * sizeof(float)));
    dim3 grid, block;
    get_grid_block(batch_size, grid, block);
    kernel_fill_ones<<<grid, block>>>(batch_size, d_ones);
    
    initialized_buffers = true;
}

void GpuAutoencoder::deallocate_buffers() {
    for (int l = 0; l <= num_l; ++l) {
        if (d_a[l]) {
            cudaFree(d_a[l]);
            d_a[l] = nullptr;
        }
    }
    
    for (int l = 0; l < num_l; ++l) {
        if (d_z[l]) {
            cudaFree(d_z[l]);
            d_z[l] = nullptr;
        }
        if (d_dz[l]) {
            cudaFree(d_dz[l]);
            d_dz[l] = nullptr;
        }
        if (d_W[l]) {
            cudaFree(d_W[l]);
            d_W[l] = nullptr;
        }
        if (d_b[l]) {
            cudaFree(d_b[l]);
            d_b[l] = nullptr;
        }
        if (d_mW[l]) {
            cudaFree(d_mW[l]);
            d_mW[l] = nullptr;
        }
        if (d_vW[l]) {
            cudaFree(d_vW[l]);
            d_vW[l] = nullptr;
        }
        if (d_mb[l]) {
            cudaFree(d_mb[l]);
            d_mb[l] = nullptr;
        }
        if (d_vb[l]) {
            cudaFree(d_vb[l]);
            d_vb[l] = nullptr;
        }
        if (d_dW[l]) {
            cudaFree(d_dW[l]);
            d_dW[l] = nullptr;
        }
        if (d_db[l]) {
            cudaFree(d_db[l]);
            d_db[l] = nullptr;
        }
    }
    
    if (d_loss) {
        cudaFree(d_loss);
        d_loss = nullptr;
    }
    if (d_ones) {
        cudaFree(d_ones);
        d_ones = nullptr;
    }
    
    initialized_buffers = false;
}

void GpuAutoencoder::forward(const SparseBatch& x) {
    // Set batch size on first forward
    if (batch_size == 0) {
        batch_size = x.B;
        allocate_buffers();
    }
    
    cublasHandle_t blas_h = (cublasHandle_t)cublas_handle;
    cusparseHandle_t sparse_h = (cusparseHandle_t)cusparse_handle;
    
    CUBLAS_CHECK(cublasSetStream(blas_h, x.stream));
    CUSPARSE_CHECK(cusparseSetStream(sparse_h, x.stream));
    
    int d0 = dims[0];
    int d1 = dims[1];
    int B = x.B;
    
    // Layer 0: sparse input
    // z[0] = W[0] * x + b[0]
    // Use cuSPARSE SpMM: treat x (m × B sparse CSC) as CSR(B × m) with transpose
    {
        float one_f = 1.0f, zero_f = 0.0f;
        
        // Create sparse descriptor for x as CSC (CSC format data)
        cusparseSpMatDescr_t matA;
        CUSPARSE_CHECK(cusparseCreateCsc(
            &matA, d0, B, x.nnz,
            (void*)x.d_col_ptr, (void*)x.d_row_idx, (void*)x.d_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F));
        
        // Create dense descriptor for W[0] (d1 × d0 column-major, ld=d1)
        // Need to apply TRANSPOSE in SpMM to get (d0, d1) for multiplication
        cusparseDnMatDescr_t matB;
        CUSPARSE_CHECK(cusparseCreateDnMat(
            &matB, d1, d0, d1,
            d_W[0], CUDA_R_32F, CUSPARSE_ORDER_COL));
        
        // Create dense descriptor for z[0]^T (B × out row-major, ld=d1)
        cusparseDnMatDescr_t matC;
        CUSPARSE_CHECK(cusparseCreateDnMat(
            &matC, B, d1, d1,
            d_z[0], CUDA_R_32F, CUSPARSE_ORDER_ROW));
        
        // Forward SpMM: z[0] = W[0] @ x
        // matA: x as CSC(d0, B), with OpA=TRANSPOSE → (B, d0)
        // matB: W[0] as (d1, d0, COL), with OpB=TRANSPOSE → (d0, d1)
        // matC: z[0] result (B, d1)
        // Computation: (B, d0) @ (d0, d1) = (B, d1) → d_z[0]
        size_t workspace_size = 0;
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            sparse_h, CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE, &one_f, matA, matB,
            &zero_f, matC, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1,
            &workspace_size));
        
        void* workspace = nullptr;
        if (workspace_size > 0) {
            CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
        }
        
        // SpMM
        CUSPARSE_CHECK(cusparseSpMM(
            sparse_h, CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE, &one_f, matA, matB,
            &zero_f, matC, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, workspace));
        
        if (workspace) CUDA_CHECK(cudaFree(workspace));
        cusparseDestroySpMat(matA);
        cusparseDestroyDnMat(matB);
        cusparseDestroyDnMat(matC);
    }
    
    // Add bias to z[0]: for (B, d1) row-major, z[bidx, i] += b[i]
    {
        dim3 grid, block;
        get_grid_block(B, grid, block);
        kernel_add_bias_rowmajor<<<grid, block, 0, x.stream>>>(d1, B, d_z[0], d_b[0]);
    }
    
    // Copy z[0] to a[1] (or apply ReLU): for (B, d1) row-major
    {
        int size = d1 * B;
        dim3 grid, block;
        get_grid_block(size, grid, block);
        kernel_relu_forward<<<grid, block, 0, x.stream>>>(size, d_a[1], d_z[0]);
    }
    
    // Layers 1..L-1: dense
    for (int l = 1; l < num_l; ++l) {
        int in = dims[l];
        int out = dims[l + 1];
        
        // Forward dense layer: z[l] = W[l] @ a[l] + b[l]
        // W[l]: (out, in) column-major (ld=out)
        // a[l]: (in, B) column-major (ld=in)
        // z[l]: (out, B) column-major (ld=out) - result
        // Computation: (out, in) @ (in, B) = (out, B) → d_z[l]
        float one_f = 1.0f, zero_f = 0.0f;
        CUBLAS_CHECK(cublasSgemm(
            blas_h, CUBLAS_OP_N, CUBLAS_OP_N,
            out, B, in,
            &one_f,
            d_W[l], out,
            d_a[l], in,
            &zero_f,
            d_z[l], out));
        
        // Add bias
        {
            dim3 grid, block;
            get_grid_block(B, grid, block);
            kernel_add_bias<<<grid, block, 0, x.stream>>>(out, B, d_z[l], d_b[l]);
        }
        
        // Apply ReLU if not last layer
        if (l < num_l - 1) {
            int size = out * B;
            dim3 grid, block;
            get_grid_block(size, grid, block);
            kernel_relu_forward<<<grid, block, 0, x.stream>>>(size, d_a[l+1], d_z[l]);
        } else {
            // Last layer: linear (copy z to a)
            int size = out * B;
            CUDA_CHECK(cudaMemcpyAsync(d_a[num_l], d_z[num_l - 1], size * sizeof(float),
                                       cudaMemcpyDeviceToDevice, x.stream));
        }
    }
}

float GpuAutoencoder::backward_and_step(const SparseBatch& x, float lr) {
    cublasHandle_t blas_h = (cublasHandle_t)cublas_handle;
    cusparseHandle_t sparse_h = (cusparseHandle_t)cusparse_handle;
    
    CUBLAS_CHECK(cublasSetStream(blas_h, x.stream));
    CUSPARSE_CHECK(cusparseSetStream(sparse_h, x.stream));
    
    int d0 = dims[0];
    int dL = dims[num_l];
    int B = x.B;
    
    // Compute loss: ||a[L] - x||_F^2 / (d0 * B)
    // Strategy: loss_sum = ||a[L]||_F^2
    //           then for each sparse (r, j, v): loss_sum += v^2 - 2*a[L](r,j)*v
    float loss_h = 0.0f;
    
    // Compute ||a[L]||_F^2: dot product a[L] · a[L]
    // a[L]: (d0, B) = (dL, B) column-major (ld=dL)
    // Result: scalar norm_sq = sum of squares → d_loss (device) or norm_sq (host)
    float norm_sq = 0.0f;
    CUBLAS_CHECK(cublasSdot(blas_h, dL * B, d_a[num_l], 1, d_a[num_l], 1, &norm_sq));
    loss_h = norm_sq;
    
    // Initialize dz[L-1] = a[L] / B for all entries
    {
        int size = dL * B;
        CUDA_CHECK(cudaMemcpyAsync(d_dz[num_l - 1], d_a[num_l], size * sizeof(float),
                                   cudaMemcpyDeviceToDevice, x.stream));
        
        // Scale dz[L-1] in-place: multiply each element by 1/B
        // dz[L-1]: (dL, B) column-major → scaled in-place
        float scale = 1.0f / B;
        CUBLAS_CHECK(cublasSscal(blas_h, size, &scale, d_dz[num_l - 1], 1));
    }
    
    // Sparse correction kernel: for each sparse (r, j, v):
    //   dz[L-1](r,j) = (a[L](r,j) - v) / B
    //   loss_h += v^2 - 2*a[L](r,j)*v
    {
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
        
        // Launch one block per column
        kernel_sparse_loss_and_dz_update<<<B, 256, 0, x.stream>>>(
            d0, B, d_dz[num_l - 1], d_loss,
            d_a[num_l],
            x.d_col_ptr, x.d_row_idx, x.d_values);
        
        float loss_correction = 0.0f;
        CUDA_CHECK(cudaMemcpyAsync(&loss_correction, d_loss, sizeof(float),
                                   cudaMemcpyDeviceToHost, x.stream));
        CUDA_CHECK(cudaStreamSynchronize(x.stream));
        
        loss_h += loss_correction;
    }
    
    // Finalize loss: divide by (d0 * B)
    float loss_final = loss_h / (static_cast<float>(d0) * static_cast<float>(B));
    
    // Backward pass: l = L-1 down to 0
    for (int l = num_l - 1; l >= 0; --l) {
        int in = dims[l];
        int out = dims[l + 1];
        
        // ReLU backward for hidden layers
        if (l < num_l - 1) {
            int size = out * B;
            dim3 grid, block;
            get_grid_block(size, grid, block);
            kernel_relu_backward<<<grid, block, 0, x.stream>>>(size, d_dz[l], d_z[l]);
        }
        // For l == L-1, dz[l] is already set (output layer)
        
        // Compute dW[l] = dz[l] * a[l]^T
        if (l > 0) {
            // Dense case: compute gradient dW[l] = dz[l] @ a[l]^T
            // dz[l]: (out, B) column-major (ld=out)
            // a[l]: (in, B) column-major (ld=in), transposed to (B, in)
            // dW[l]: (out, in) column-major (ld=out) - result
            // Computation: (out, B) @ (B, in) = (out, in) → d_dW[l]
            float one_f = 1.0f, zero_f = 0.0f;
            CUBLAS_CHECK(cublasSgemm(
                blas_h, CUBLAS_OP_N, CUBLAS_OP_T,
                out, in, B,
                &one_f,
                d_dz[l], out,
                d_a[l], in,
                &zero_f,
                d_dW[l], out));
        } else {
            // Sparse case (l=0): compute gradient dW[0] = dz[0] @ x^T
            // x: CSC(d0, B) with OpA=NON_TRANSPOSE → (d0, B)
            // dz[0]: (out, B) as (B, out) row-major, transposed to (out, B) col-major
            // dW[0]: (d0, out) result
            // Computation: (d0, B) @ (B, out) = (d0, out) → d_dW[0]
            
            float one_f = 1.0f, zero_f = 0.0f;
            
            cusparseSpMatDescr_t matA;
            CUSPARSE_CHECK(cusparseCreateCsc(
                &matA, d0, B, x.nnz,
                (void*)x.d_col_ptr, (void*)x.d_row_idx, (void*)x.d_values,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                CUDA_R_32F));
            
            cusparseDnMatDescr_t matB;
            CUSPARSE_CHECK(cusparseCreateDnMat(
                &matB, B, out, out,
                d_dz[0], CUDA_R_32F, CUSPARSE_ORDER_ROW));
            
            cusparseDnMatDescr_t matC;
            CUSPARSE_CHECK(cusparseCreateDnMat(
                &matC, d0, out, out,
                d_dW[0], CUDA_R_32F, CUSPARSE_ORDER_ROW));
            
            size_t workspace_size = 0;
            CUSPARSE_CHECK(cusparseSpMM_bufferSize(
                sparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE, &one_f, matA, matB,
                &zero_f, matC, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1,
                &workspace_size));
            
            void* workspace = nullptr;
            if (workspace_size > 0) {
                CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
            }
            
            CUSPARSE_CHECK(cusparseSpMM(
                sparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE, &one_f, matA, matB,
                &zero_f, matC, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, workspace));
            
            if (workspace) CUDA_CHECK(cudaFree(workspace));
            cusparseDestroySpMat(matA);
            cusparseDestroyDnMat(matB);
            cusparseDestroyDnMat(matC);
        }
        
        // db[l] = rowwise sum of dz[l] (dz[l] @ ones_B)
        // dz[l]: (out, B) column-major (ld=out)
        // ones_B: (B, 1) vector of ones
        // db[l]: (out, 1) result - gradient for bias
        // Computation: (out, B) @ (B, 1) = (out, 1) → d_db[l]
        {
            float one_f = 1.0f, zero_f = 0.0f;
            CUBLAS_CHECK(cublasSgemv(
                blas_h, CUBLAS_OP_N,
                out, B,
                &one_f,
                d_dz[l], out,
                d_ones, 1,
                &zero_f,
                d_db[l], 1));
        }
        
        // Propagate to previous layer: dz[l-1] = W[l]^T * dz[l]
        // W[l]: (out, in) column-major (ld=out), transposed to (in, out)
        // dz[l]: (out, B) column-major (ld=out)
        // dz[l-1]: (in, B) column-major (ld=in) - result
        // Computation: (in, out) @ (out, B) = (in, B) → d_dz[l-1]
        if (l > 0) {
            float one_f = 1.0f, zero_f = 0.0f;
            CUBLAS_CHECK(cublasSgemm(
                blas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                in, B, out,
                &one_f,
                d_W[l], out,
                d_dz[l], out,
                &zero_f,
                d_dz[l - 1], in));
        }
    }
    
    // Adam update
    ++t;
    float bc1 = 1.0f - std::pow(beta1, static_cast<float>(t));
    float bc2 = 1.0f - std::pow(beta2, static_cast<float>(t));
    float lr_t = lr * std::sqrt(bc2) / bc1;
    
    for (int l = 0; l < num_l; ++l) {
        int in = dims[l];
        int out = dims[l + 1];
        
        // Update W[l]
        {
            int size = out * in;
            dim3 grid, block;
            get_grid_block(size, grid, block);
            kernel_adam_update<<<grid, block, 0, x.stream>>>(
                size, d_W[l], d_mW[l], d_vW[l], d_dW[l],
                lr_t, beta1, beta2, eps);
        }
        
        // Update b[l]
        {
            int size = out;
            dim3 grid, block;
            get_grid_block(size, grid, block);
            kernel_adam_update<<<grid, block, 0, x.stream>>>(
                size, d_b[l], d_mb[l], d_vb[l], d_db[l],
                lr_t, beta1, beta2, eps);
        }
    }
    
    return loss_final;
}

int GpuAutoencoder::num_layers() const {
    return num_l;
}
