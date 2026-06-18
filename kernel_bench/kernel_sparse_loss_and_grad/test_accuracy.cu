// Test accuracy of original vs replacement kernel
// Loads .bin files, runs both kernels, compares to expected outputs

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

// Forward declarations from original.cu and replacement.cu
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

// Load binary file into host buffer
void load_binary(const char* filename, void* buffer, size_t size) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        exit(EXIT_FAILURE);
    }
    size_t read = fread(buffer, 1, size, f);
    if (read != size) {
        fprintf(stderr, "Failed to read %s (got %zu, expected %zu)\n", filename, read, size);
        exit(EXIT_FAILURE);
    }
    fclose(f);
}

// Compare float arrays with relative and absolute tolerance
bool compare_float(const float* a, const float* b, int n, float rel_tol, float abs_tol, const char* name) {
    bool all_pass = true;
    for (int i = 0; i < n; ++i) {
        float diff = fabs(a[i] - b[i]);
        float rel_diff = (fabs(b[i]) > 1e-7) ? diff / fabs(b[i]) : diff;
        
        if (diff > abs_tol && rel_diff > rel_tol) {
            fprintf(stderr, "%s[%d]: got %f, expected %f, rel_diff=%f, abs_diff=%f\n",
                    name, i, a[i], b[i], rel_diff, diff);
            all_pass = false;
            if (i >= 10) {
                fprintf(stderr, "... (stopping after 10 errors)\n");
                break;
            }
        }
    }
    return all_pass;
}

int main() {
    printf("Test accuracy: kernel_sparse_loss_and_grad\n\n");
    
    // ========================================================================
    // Step 1: Load metadata
    // ========================================================================
    FILE* meta_f = fopen("meta.txt", "r");
    if (!meta_f) {
        fprintf(stderr, "Failed to open meta.txt\n");
        return 1;
    }
    
    int d0, B, nnz;
    if (fscanf(meta_f, "d0=%d\nB=%d\nnnz=%d\n", &d0, &B, &nnz) != 3) {
        fprintf(stderr, "Failed to parse meta.txt\n");
        return 1;
    }
    fclose(meta_f);
    
    printf("Parameters: d0=%d, B=%d, nnz=%d\n\n", d0, B, nnz);
    
    // ========================================================================
    // Step 2: Load inputs from host memory
    // ========================================================================
    float* h_a_L = (float*)malloc(d0 * B * sizeof(float));
    float* h_pre_grad = (float*)malloc(d0 * B * sizeof(float));
    int32_t* h_col_ptr = (int32_t*)malloc((B + 1) * sizeof(int32_t));
    int32_t* h_row_idx = (int32_t*)malloc(nnz * sizeof(int32_t));
    float* h_values = (float*)malloc(nnz * sizeof(float));
    float* h_expected_grad = (float*)malloc(d0 * B * sizeof(float));
    float* h_expected_loss = (float*)malloc(1 * sizeof(float));
    
    load_binary("a_L.bin", h_a_L, d0 * B * sizeof(float));
    load_binary("pre_grad.bin", h_pre_grad, d0 * B * sizeof(float));
    load_binary("col_ptr.bin", h_col_ptr, (B + 1) * sizeof(int32_t));
    load_binary("row_idx.bin", h_row_idx, nnz * sizeof(int32_t));
    load_binary("values.bin", h_values, nnz * sizeof(float));
    load_binary("expected_grad.bin", h_expected_grad, d0 * B * sizeof(float));
    load_binary("expected_loss.bin", h_expected_loss, 1 * sizeof(float));
    
    printf("Loaded all inputs and expected outputs.\n\n");
    
    // ========================================================================
    // Step 3: Allocate device memory and copy inputs
    // ========================================================================
    float* d_a_L;
    int32_t* d_col_ptr;
    int32_t* d_row_idx;
    float* d_values;
    float* d_grad_loss_original;
    float* d_grad_loss_replacement;
    float* d_loss_original;
    float* d_loss_replacement;
    
    CUDA_CHECK(cudaMalloc(&d_a_L, d0 * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_col_ptr, (B + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_row_idx, nnz * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_loss_original, d0 * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_loss_replacement, d0 * B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss_original, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss_replacement, sizeof(float)));
    
    // Copy inputs
    CUDA_CHECK(cudaMemcpy(d_a_L, h_a_L, d0 * B * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ptr, h_col_ptr, (B + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_idx, h_row_idx, nnz * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    
    // ========================================================================
    // Step 4: Initialize gradient for both runs (pre_grad from file)
    // ========================================================================
    CUDA_CHECK(cudaMemcpy(d_grad_loss_original, h_pre_grad, d0 * B * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_loss_replacement, h_pre_grad, d0 * B * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize loss accumulators to 0
    CUDA_CHECK(cudaMemset(d_loss_original, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss_replacement, 0, sizeof(float)));
    
    printf("Running ORIGINAL kernel...\n");
    launch_kernel_sparse_loss_and_grad(
        d0, B, d_grad_loss_original, d_loss_original,
        d_a_L, d_col_ptr, d_row_idx, d_values, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("Running REPLACEMENT kernel...\n");
    launch_kernel_sparse_loss_and_grad_replacement(
        d0, B, d_grad_loss_replacement, d_loss_replacement,
        d_a_L, d_col_ptr, d_row_idx, d_values, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // ========================================================================
    // Step 4b: Normalize loss by (d0 * B)
    // ========================================================================
    // Create a simple kernel to divide both loss values by (d0 * B)
    float norm_factor = 1.0f / (d0 * B);
    float h_loss_norm[2];
    
    CUDA_CHECK(cudaMemcpy(h_loss_norm, d_loss_original, sizeof(float), cudaMemcpyDeviceToHost));
    h_loss_norm[0] *= norm_factor;
    CUDA_CHECK(cudaMemcpy(d_loss_original, h_loss_norm, sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpy(h_loss_norm, d_loss_replacement, sizeof(float), cudaMemcpyDeviceToHost));
    h_loss_norm[0] *= norm_factor;
    CUDA_CHECK(cudaMemcpy(d_loss_replacement, h_loss_norm, sizeof(float), cudaMemcpyHostToDevice));
    
    // ========================================================================
    // Step 5: Copy results back to host
    // ========================================================================
    float* h_grad_original = (float*)malloc(d0 * B * sizeof(float));
    float* h_grad_replacement = (float*)malloc(d0 * B * sizeof(float));
    float* h_loss_original = (float*)malloc(sizeof(float));
    float* h_loss_replacement = (float*)malloc(sizeof(float));
    
    CUDA_CHECK(cudaMemcpy(h_grad_original, d_grad_loss_original, d0 * B * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_replacement, d_grad_loss_replacement, d0 * B * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_loss_original, d_loss_original, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_loss_replacement, d_loss_replacement, sizeof(float), cudaMemcpyDeviceToHost));
    
    // ========================================================================
    // Step 6: Compare results
    // ========================================================================
    const float rel_tol = 1e-5f;
    const float abs_tol = 1e-6f;
    
    printf("\n=== GRADIENT COMPARISON ===\n");
    bool grad_original_pass = compare_float(h_grad_original, h_expected_grad, d0 * B, rel_tol, abs_tol, "ORIGINAL grad");
    printf("ORIGINAL gradient: %s\n\n", grad_original_pass ? "PASS" : "FAIL");
    
    bool grad_replacement_pass = compare_float(h_grad_replacement, h_expected_grad, d0 * B, rel_tol, abs_tol, "REPLACEMENT grad");
    printf("REPLACEMENT gradient: %s\n\n", grad_replacement_pass ? "PASS" : "FAIL");
    
    printf("=== LOSS COMPARISON ===\n");
    printf("Original loss: %f, Expected: %f, Diff: %e\n",
           h_loss_original[0], h_expected_loss[0], fabs(h_loss_original[0] - h_expected_loss[0]));
    bool loss_original_pass = fabs(h_loss_original[0] - h_expected_loss[0]) <= abs_tol ||
                               fabs(h_loss_original[0] - h_expected_loss[0]) / fabs(h_expected_loss[0]) <= rel_tol;
    printf("ORIGINAL loss: %s\n\n", loss_original_pass ? "PASS" : "FAIL");
    
    printf("Replacement loss: %f, Expected: %f, Diff: %e\n",
           h_loss_replacement[0], h_expected_loss[0], fabs(h_loss_replacement[0] - h_expected_loss[0]));
    bool loss_replacement_pass = fabs(h_loss_replacement[0] - h_expected_loss[0]) <= abs_tol ||
                                  fabs(h_loss_replacement[0] - h_expected_loss[0]) / fabs(h_expected_loss[0]) <= rel_tol;
    printf("REPLACEMENT loss: %s\n\n", loss_replacement_pass ? "PASS" : "FAIL");
    
    // ========================================================================
    // Step 7: Print final verdict
    // ========================================================================
    bool original_pass = grad_original_pass && loss_original_pass;
    bool replacement_pass = grad_replacement_pass && loss_replacement_pass;
    
    printf("=== FINAL RESULTS ===\n");
    printf("ORIGINAL: %s\n", original_pass ? "PASS" : "FAIL");
    printf("REPLACEMENT: %s\n", replacement_pass ? "PASS" : "FAIL");
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    free(h_a_L);
    free(h_pre_grad);
    free(h_col_ptr);
    free(h_row_idx);
    free(h_values);
    free(h_expected_grad);
    free(h_expected_loss);
    free(h_grad_original);
    free(h_grad_replacement);
    free(h_loss_original);
    free(h_loss_replacement);
    
    cudaFree(d_a_L);
    cudaFree(d_col_ptr);
    cudaFree(d_row_idx);
    cudaFree(d_values);
    cudaFree(d_grad_loss_original);
    cudaFree(d_grad_loss_replacement);
    cudaFree(d_loss_original);
    cudaFree(d_loss_replacement);
    
    return (original_pass && replacement_pass) ? 0 : 1;
}
