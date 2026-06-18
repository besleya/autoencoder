// Benchmark original vs replacement kernel
// Times both kernels over multiple iterations, reports median and speedup

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>

// Forward declarations
extern "C" void launch_kernel_sparse_loss_and_grad(
    int d0, int B, float* d_grad_loss, float* loss_acc,
    const float* a_L, const int32_t* col_ptr, const int32_t* row_idx,
    const float* values, cudaStream_t stream);

extern "C" void launch_kernel_sparse_loss_and_grad_replacement(
    int d0, int B, float* d_grad_loss, float* loss_acc,
    const float* a_L, const int32_t* col_ptr, const int32_t* row_idx,
    const float* values, cudaStream_t stream);

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, \
                    __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Load binary file
void load_binary(const char* filename, void* buffer, size_t size) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        exit(EXIT_FAILURE);
    }
    size_t read = fread(buffer, 1, size, f);
    if (read != size) {
        fprintf(stderr, "Failed to read %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fclose(f);
}

int main() {
    printf("Benchmark: kernel_sparse_loss_and_grad\n\n");
    
    // Load metadata
    FILE* meta_f = fopen("meta.txt", "r");
    if (!meta_f) {
        fprintf(stderr, "Failed to open meta.txt\n");
        return 1;
    }
    
    int d0, B, nnz;
    fscanf(meta_f, "d0=%d\nB=%d\nnnz=%d\n", &d0, &B, &nnz);
    fclose(meta_f);
    
    printf("Parameters: d0=%d, B=%d, nnz=%d\n\n", d0, B, nnz);
    
    // Load inputs
    float* h_a_L = (float*)malloc(d0 * B * sizeof(float));
    float* h_pre_grad = (float*)malloc(d0 * B * sizeof(float));
    int32_t* h_col_ptr = (int32_t*)malloc((B + 1) * sizeof(int32_t));
    int32_t* h_row_idx = (int32_t*)malloc(nnz * sizeof(int32_t));
    float* h_values = (float*)malloc(nnz * sizeof(float));
    
    load_binary("a_L.bin", h_a_L, d0 * B * sizeof(float));
    load_binary("pre_grad.bin", h_pre_grad, d0 * B * sizeof(float));
    load_binary("col_ptr.bin", h_col_ptr, (B + 1) * sizeof(int32_t));
    load_binary("row_idx.bin", h_row_idx, nnz * sizeof(int32_t));
    load_binary("values.bin", h_values, nnz * sizeof(float));
    
    // Allocate device memory
    float* d_a_L;
    int32_t* d_col_ptr;
    int32_t* d_row_idx;
    float* d_values;
    float* d_grad_loss;
    float* d_loss;
    
    CUDA_CHECK(cudaMalloc(&d_a_L, d0 * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_col_ptr, (B + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_row_idx, nnz * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_loss, d0 * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_a_L, h_a_L, d0 * B * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ptr, h_col_ptr, (B + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_idx, h_row_idx, nnz * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    
    // Benchmark parameters
    const int num_warmup = 10;
    const int num_timed = 100;
    
    // ========================================================================
    // Benchmark ORIGINAL kernel
    // ========================================================================
    printf("Benchmarking ORIGINAL kernel...\n");
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    std::vector<float> times_original;
    
    // Warmup
    for (int i = 0; i < num_warmup; ++i) {
        CUDA_CHECK(cudaMemcpy(d_grad_loss, h_pre_grad, d0 * B * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
        launch_kernel_sparse_loss_and_grad(d0, B, d_grad_loss, d_loss, d_a_L, d_col_ptr, d_row_idx, d_values, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Timed runs
    for (int i = 0; i < num_timed; ++i) {
        CUDA_CHECK(cudaMemcpy(d_grad_loss, h_pre_grad, d0 * B * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
        
        CUDA_CHECK(cudaEventRecord(start));
        launch_kernel_sparse_loss_and_grad(d0, B, d_grad_loss, d_loss, d_a_L, d_col_ptr, d_row_idx, d_values, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_original.push_back(ms);
    }
    
    std::sort(times_original.begin(), times_original.end());
    float median_original = times_original[num_timed / 2];
    printf("ORIGINAL median time: %.4f ms\n\n", median_original);
    
    // ========================================================================
    // Benchmark REPLACEMENT kernel
    // ========================================================================
    printf("Benchmarking REPLACEMENT kernel...\n");
    
    std::vector<float> times_replacement;
    
    // Warmup
    for (int i = 0; i < num_warmup; ++i) {
        CUDA_CHECK(cudaMemcpy(d_grad_loss, h_pre_grad, d0 * B * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
        launch_kernel_sparse_loss_and_grad_replacement(d0, B, d_grad_loss, d_loss, d_a_L, d_col_ptr, d_row_idx, d_values, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Timed runs
    for (int i = 0; i < num_timed; ++i) {
        CUDA_CHECK(cudaMemcpy(d_grad_loss, h_pre_grad, d0 * B * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
        
        CUDA_CHECK(cudaEventRecord(start));
        launch_kernel_sparse_loss_and_grad_replacement(d0, B, d_grad_loss, d_loss, d_a_L, d_col_ptr, d_row_idx, d_values, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_replacement.push_back(ms);
    }
    
    std::sort(times_replacement.begin(), times_replacement.end());
    float median_replacement = times_replacement[num_timed / 2];
    printf("REPLACEMENT median time: %.4f ms\n\n", median_replacement);
    
    // ========================================================================
    // Report speedup
    // ========================================================================
    float speedup = median_original / median_replacement;
    printf("=== SUMMARY ===\n");
    printf("ORIGINAL:     %.4f ms\n", median_original);
    printf("REPLACEMENT:  %.4f ms\n", median_replacement);
    printf("Speedup:      %.2f x\n", speedup);
    
    // Cleanup
    free(h_a_L);
    free(h_pre_grad);
    free(h_col_ptr);
    free(h_row_idx);
    free(h_values);
    
    cudaFree(d_a_L);
    cudaFree(d_col_ptr);
    cudaFree(d_row_idx);
    cudaFree(d_values);
    cudaFree(d_grad_loss);
    cudaFree(d_loss);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
