// Benchmark: kernel_accumulate_loss replacement
// Compares original kernel vs replacement with detailed timing

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

// Forward declarations
extern void kernel_accumulate_loss_launch_original(const float* d_dot,
                                                    const float* d_loss,
                                                    float* d_epoch_sum,
                                                    int d0, int B,
                                                    cudaStream_t stream);

extern void kernel_accumulate_loss_replacement(const float* d_dot,
                                                const float* d_loss,
                                                float* d_epoch_sum,
                                                int d0, int B,
                                                cublasHandle_t handle,
                                                cudaStream_t stream);

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
} while(0)

// Read float from binary file
float read_float_bin(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        exit(1);
    }
    float val;
    if (fread(&val, sizeof(float), 1, f) != 1) {
        fprintf(stderr, "Failed to read from %s\n", filename);
        exit(1);
    }
    fclose(f);
    return val;
}

int main() {
    const int d0 = 64;
    const int B = 32;
    const int NUM_WARMUP = 10;
    const int NUM_TIMED = 1000;

    // Read input data
    float h_d_dot = read_float_bin("d_dot.bin");
    float h_d_loss = read_float_bin("d_loss.bin");
    float h_epoch_sum_init = read_float_bin("epoch_sum_init.bin");

    printf("=== Benchmark: kernel_accumulate_loss ===\n");
    printf("Parameters: d0=%d, B=%d\n", d0, B);
    printf("Warmup iterations: %d\n", NUM_WARMUP);
    printf("Timed iterations: %d\n\n", NUM_TIMED);

    // Allocate device memory
    float *d_dot, *d_loss, *d_epoch_sum;
    CUDA_CHECK(cudaMalloc(&d_dot, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_epoch_sum, sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_dot, &h_d_dot, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_loss, &h_d_loss, sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // Create CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Benchmark original kernel
    printf("Benchmarking Original Kernel...\n");
    {
        // Warmup
        for (int i = 0; i < NUM_WARMUP; ++i) {
            CUDA_CHECK(cudaMemcpy(d_epoch_sum, &h_epoch_sum_init, sizeof(float), cudaMemcpyHostToDevice));
            kernel_accumulate_loss_launch_original(d_dot, d_loss, d_epoch_sum, d0, B, 0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed iterations
        std::vector<float> times;
        for (int i = 0; i < NUM_TIMED; ++i) {
            CUDA_CHECK(cudaMemcpy(d_epoch_sum, &h_epoch_sum_init, sizeof(float), cudaMemcpyHostToDevice));
            
            CUDA_CHECK(cudaEventRecord(start, 0));
            kernel_accumulate_loss_launch_original(d_dot, d_loss, d_epoch_sum, d0, B, 0);
            CUDA_CHECK(cudaEventRecord(stop, 0));
            
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            times.push_back(ms);
        }

        // Compute statistics
        std::sort(times.begin(), times.end());
        float median = times[NUM_TIMED / 2];
        float mean = 0;
        for (float t : times) mean += t;
        mean /= NUM_TIMED;

        printf("  Median: %.6f ms\n", median);
        printf("  Mean:   %.6f ms\n", mean);
        printf("  Min:    %.6f ms\n", times[0]);
        printf("  Max:    %.6f ms\n\n", times[NUM_TIMED-1]);
    }

    // Benchmark replacement
    printf("Benchmarking Replacement (Host Accumulate)...\n");
    {
        float median_original = 0;
        
        // First, re-measure original for fair comparison
        std::vector<float> times_orig;
        for (int i = 0; i < NUM_WARMUP; ++i) {
            CUDA_CHECK(cudaMemcpy(d_epoch_sum, &h_epoch_sum_init, sizeof(float), cudaMemcpyHostToDevice));
            kernel_accumulate_loss_launch_original(d_dot, d_loss, d_epoch_sum, d0, B, 0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        for (int i = 0; i < NUM_TIMED; ++i) {
            CUDA_CHECK(cudaMemcpy(d_epoch_sum, &h_epoch_sum_init, sizeof(float), cudaMemcpyHostToDevice));
            
            CUDA_CHECK(cudaEventRecord(start, 0));
            kernel_accumulate_loss_launch_original(d_dot, d_loss, d_epoch_sum, d0, B, 0);
            CUDA_CHECK(cudaEventRecord(stop, 0));
            
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            times_orig.push_back(ms);
        }
        std::sort(times_orig.begin(), times_orig.end());
        median_original = times_orig[NUM_TIMED / 2];

        // Warmup replacement
        for (int i = 0; i < NUM_WARMUP; ++i) {
            CUDA_CHECK(cudaMemcpy(d_epoch_sum, &h_epoch_sum_init, sizeof(float), cudaMemcpyHostToDevice));
            kernel_accumulate_loss_replacement(d_dot, d_loss, d_epoch_sum, d0, B, cublas_handle, 0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed iterations
        std::vector<float> times_repl;
        for (int i = 0; i < NUM_TIMED; ++i) {
            CUDA_CHECK(cudaMemcpy(d_epoch_sum, &h_epoch_sum_init, sizeof(float), cudaMemcpyHostToDevice));
            
            CUDA_CHECK(cudaEventRecord(start, 0));
            kernel_accumulate_loss_replacement(d_dot, d_loss, d_epoch_sum, d0, B, cublas_handle, 0);
            CUDA_CHECK(cudaEventRecord(stop, 0));
            
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            times_repl.push_back(ms);
        }

        // Compute statistics
        std::sort(times_repl.begin(), times_repl.end());
        float median_repl = times_repl[NUM_TIMED / 2];
        float mean_repl = 0;
        for (float t : times_repl) mean_repl += t;
        mean_repl /= NUM_TIMED;

        printf("  Median: %.6f ms\n", median_repl);
        printf("  Mean:   %.6f ms\n", mean_repl);
        printf("  Min:    %.6f ms\n", times_repl[0]);
        printf("  Max:    %.6f ms\n\n", times_repl[NUM_TIMED-1]);

        // Compute speedup
        float speedup = median_original / median_repl;
        printf("=== Comparison ===\n");
        printf("Original median:     %.6f ms\n", median_original);
        printf("Replacement median:  %.6f ms\n", median_repl);
        printf("Speedup:             %.2f x\n", speedup);
        if (speedup > 1.0f) {
            printf("Result: Replacement is FASTER\n");
        } else {
            printf("Result: Original is faster\n");
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_dot));
    CUDA_CHECK(cudaFree(d_loss));
    CUDA_CHECK(cudaFree(d_epoch_sum));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));

    return 0;
}
