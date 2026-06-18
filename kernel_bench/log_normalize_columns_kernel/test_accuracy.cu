// SPDX-License-Identifier: MIT
// test_accuracy.cu — Accuracy test for log_normalize_columns_kernel implementations

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t err = (expr);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err));                                       \
      exit(1);                                                               \
    }                                                                         \
  } while (0)

// Declarations of launcher functions (defined in original.cu, replacement.cu)
extern void log_normalize_columns_original(int n_cols,
                                          float scaler,
                                          const int32_t* col_ptr,
                                          float* values,
                                          cudaStream_t stream);

extern void log_normalize_columns_lib(int n_cols,
                                     float scaler,
                                     const int32_t* col_ptr,
                                     float* values,
                                     cudaStream_t stream);

// Helper: load binary file
std::vector<uint8_t> load_binary_file(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> data(size);
    if (fread(data.data(), 1, size, f) != (size_t)size) {
        fprintf(stderr, "Failed to read %s\n", filename);
        exit(1);
    }
    fclose(f);
    return data;
}

// Load metadata from meta.txt
void load_meta(int& n_cols, int& nnz, float& scaler) {
    FILE* f = fopen("meta.txt", "r");
    if (!f) {
        fprintf(stderr, "Failed to open meta.txt\n");
        exit(1);
    }
    if (fscanf(f, "%d %d %f", &n_cols, &nnz, &scaler) != 3) {
        fprintf(stderr, "Failed to parse meta.txt\n");
        exit(1);
    }
    fclose(f);
}

// Compare two float arrays with tolerance
bool compare_arrays(const float* expected, const float* actual, int n,
                   float rel_tol = 1e-5f, float abs_tol = 1e-6f) {
    bool all_pass = true;
    int num_errors = 0;
    for (int i = 0; i < n; i++) {
        float e = expected[i];
        float a = actual[i];
        float abs_err = fabsf(e - a);
        float rel_err = (fabsf(e) > 1e-8f) ? (abs_err / fabsf(e)) : 0.0f;
        
        if (abs_err > abs_tol && rel_err > rel_tol) {
            if (num_errors < 10) {
                printf("Mismatch at [%d]: expected=%.6e, actual=%.6e, abs_err=%.6e, rel_err=%.6e\n",
                       i, e, a, abs_err, rel_err);
            }
            all_pass = false;
            num_errors++;
        }
    }
    if (num_errors > 10) {
        printf("... and %d more errors\n", num_errors - 10);
    }
    if (num_errors > 0) {
        printf("Total errors: %d / %d\n", num_errors, n);
    }
    return all_pass;
}

int main() {
    printf("=== log_normalize_columns_kernel Accuracy Test ===\n\n");

    // Load metadata
    int n_cols, nnz;
    float scaler;
    load_meta(n_cols, nnz, scaler);
    printf("n_cols=%d, nnz=%d, scaler=%.1f\n", n_cols, nnz, scaler);

    // Load test data
    auto col_ptr_data = load_binary_file("col_ptr.bin");
    auto values_init_data = load_binary_file("values_init.bin");
    auto expected_data = load_binary_file("expected_values.bin");

    assert(col_ptr_data.size() == (size_t)(n_cols + 1) * sizeof(int32_t));
    assert(values_init_data.size() == (size_t)nnz * sizeof(float));
    assert(expected_data.size() == (size_t)nnz * sizeof(float));

    int32_t* h_col_ptr = (int32_t*)col_ptr_data.data();
    float* h_values_init = (float*)values_init_data.data();
    float* h_expected = (float*)expected_data.data();

    // Verify col_ptr
    printf("col_ptr: [0]=%d, [%d]=%d\n", h_col_ptr[0], n_cols, h_col_ptr[n_cols]);

    // Allocate GPU memory
    int32_t* d_col_ptr;
    float* d_values;
    CUDA_CHECK(cudaMalloc(&d_col_ptr, (n_cols + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));

    // Copy test data to GPU
    CUDA_CHECK(cudaMemcpy(d_col_ptr, h_col_ptr, (n_cols + 1) * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    // Test 1: Original kernel
    printf("\n--- Test 1: Original Kernel ---\n");
    CUDA_CHECK(cudaMemcpy(d_values, h_values_init, nnz * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    log_normalize_columns_original(n_cols, scaler, d_col_ptr, d_values, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_output_orig = (float*)malloc(nnz * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_output_orig, d_values, nnz * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bool orig_pass = compare_arrays(h_expected, h_output_orig, nnz);
    printf("Original kernel: %s\n", orig_pass ? "PASS" : "FAIL");

    // Test 2: Library replacement
    printf("\n--- Test 2: Library Replacement ---\n");
    CUDA_CHECK(cudaMemcpy(d_values, h_values_init, nnz * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    log_normalize_columns_lib(n_cols, scaler, d_col_ptr, d_values, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_output_lib = (float*)malloc(nnz * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_output_lib, d_values, nnz * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bool lib_pass = compare_arrays(h_expected, h_output_lib, nnz);
    printf("Library replacement: %s\n", lib_pass ? "PASS" : "FAIL");

    // Cleanup
    cudaFree(d_col_ptr);
    cudaFree(d_values);
    free(h_output_orig);
    free(h_output_lib);

    printf("\n=== Test Summary ===\n");
    printf("Original:  %s\n", orig_pass ? "PASS" : "FAIL");
    printf("Library:   %s\n", lib_pass ? "PASS" : "FAIL");

    return (orig_pass && lib_pass) ? 0 : 1;
}
