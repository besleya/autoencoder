// SPDX-License-Identifier: MIT
// bench.cu — Benchmark for log_normalize_columns_kernel implementations

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>
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

// Declarations of launcher functions
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

// Benchmark helper
void benchmark_impl(const char* name,
                   void (*impl_fn)(int, float, const int32_t*, float*, cudaStream_t),
                   int n_cols,
                   float scaler,
                   const int32_t* d_col_ptr,
                   float* d_values,
                   const float* h_values_init,
                   int nnz,
                   int num_warmup,
                   int num_timed) {
    printf("  %s:\n", name);

    // Warmup
    for (int i = 0; i < num_warmup; i++) {
        CUDA_CHECK(cudaMemcpy(d_values, h_values_init, nnz * sizeof(float),
                              cudaMemcpyHostToDevice));
        impl_fn(n_cols, scaler, d_col_ptr, d_values, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed iterations
    std::vector<double> times_ms;
    for (int i = 0; i < num_timed; i++) {
        CUDA_CHECK(cudaMemcpy(d_values, h_values_init, nnz * sizeof(float),
                              cudaMemcpyHostToDevice));
        
        auto t0 = std::chrono::high_resolution_clock::now();
        impl_fn(n_cols, scaler, d_col_ptr, d_values, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);
    }

    // Compute statistics
    std::sort(times_ms.begin(), times_ms.end());
    double median_ms = times_ms[num_timed / 2];
    double min_ms = times_ms[0];
    double max_ms = times_ms[num_timed - 1];
    double mean_ms = 0.0;
    for (auto t : times_ms) mean_ms += t;
    mean_ms /= num_timed;

    printf("    Median: %.4f ms\n", median_ms);
    printf("    Min:    %.4f ms\n", min_ms);
    printf("    Max:    %.4f ms\n", max_ms);
    printf("    Mean:   %.4f ms\n", mean_ms);
}

int main() {
    printf("=== log_normalize_columns_kernel Benchmark ===\n\n");

    // Parameters
    int num_warmup = 10;
    int num_timed = 100;

    // Load metadata
    int n_cols, nnz;
    float scaler;
    load_meta(n_cols, nnz, scaler);
    printf("n_cols=%d, nnz=%d, scaler=%.1f\n", n_cols, nnz, scaler);
    printf("Warmup iterations: %d\n", num_warmup);
    printf("Timed iterations:  %d\n\n", num_timed);

    // Load test data
    auto col_ptr_data = load_binary_file("col_ptr.bin");
    auto values_init_data = load_binary_file("values_init.bin");

    assert(col_ptr_data.size() == (size_t)(n_cols + 1) * sizeof(int32_t));
    assert(values_init_data.size() == (size_t)nnz * sizeof(float));

    int32_t* h_col_ptr = (int32_t*)col_ptr_data.data();
    float* h_values_init = (float*)values_init_data.data();

    // Allocate GPU memory
    int32_t* d_col_ptr;
    float* d_values;
    CUDA_CHECK(cudaMalloc(&d_col_ptr, (n_cols + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_col_ptr, h_col_ptr, (n_cols + 1) * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    // Benchmark original
    printf("--- Original Kernel ---\n");
    double time_orig_ms;
    {
        std::vector<double> times_ms;
        for (int i = 0; i < num_warmup; i++) {
            CUDA_CHECK(cudaMemcpy(d_values, h_values_init, nnz * sizeof(float),
                                  cudaMemcpyHostToDevice));
            log_normalize_columns_original(n_cols, scaler, d_col_ptr, d_values, 0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int i = 0; i < num_timed; i++) {
            CUDA_CHECK(cudaMemcpy(d_values, h_values_init, nnz * sizeof(float),
                                  cudaMemcpyHostToDevice));
            auto t0 = std::chrono::high_resolution_clock::now();
            log_normalize_columns_original(n_cols, scaler, d_col_ptr, d_values, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();
            times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(times_ms.begin(), times_ms.end());
        time_orig_ms = times_ms[num_timed / 2];
        printf("  Median: %.4f ms\n", time_orig_ms);
        printf("  Min:    %.4f ms\n", times_ms[0]);
        printf("  Max:    %.4f ms\n", times_ms[num_timed - 1]);
    }

    // Benchmark library
    printf("\n--- Library Replacement ---\n");
    double time_lib_ms;
    {
        std::vector<double> times_ms;
        for (int i = 0; i < num_warmup; i++) {
            CUDA_CHECK(cudaMemcpy(d_values, h_values_init, nnz * sizeof(float),
                                  cudaMemcpyHostToDevice));
            log_normalize_columns_lib(n_cols, scaler, d_col_ptr, d_values, 0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int i = 0; i < num_timed; i++) {
            CUDA_CHECK(cudaMemcpy(d_values, h_values_init, nnz * sizeof(float),
                                  cudaMemcpyHostToDevice));
            auto t0 = std::chrono::high_resolution_clock::now();
            log_normalize_columns_lib(n_cols, scaler, d_col_ptr, d_values, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();
            times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(times_ms.begin(), times_ms.end());
        time_lib_ms = times_ms[num_timed / 2];
        printf("  Median: %.4f ms\n", time_lib_ms);
        printf("  Min:    %.4f ms\n", times_ms[0]);
        printf("  Max:    %.4f ms\n", times_ms[num_timed - 1]);
    }

    // Compute speedup
    double speedup = time_orig_ms / time_lib_ms;
    printf("\n=== Speedup ===\n");
    printf("Original:  %.4f ms\n", time_orig_ms);
    printf("Library:   %.4f ms\n", time_lib_ms);
    printf("Speedup:   %.2f×\n", speedup);

    // Cleanup
    cudaFree(d_col_ptr);
    cudaFree(d_values);

    return 0;
}
