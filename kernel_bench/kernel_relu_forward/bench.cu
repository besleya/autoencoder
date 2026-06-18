/**
 * Benchmark for kernel_relu_forward replacement.
 * 
 * Runs 10 warmup iterations + 100 timed iterations for both original and replacement.
 * Reports median execution time and speedup.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <vector>
#include <algorithm>

// Forward declarations
void relu_forward_original(int size, float* a, const float* z, cudaStream_t stream);
void relu_forward_thrust(int size, float* a, const float* z, cudaStream_t stream);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

float get_elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

int main() {
    // Read metadata
    int size;
    {
        std::ifstream meta("meta.txt");
        meta >> size;
        printf("Benchmark size: %d\n", size);
    }

    // Load z.bin
    float* z_cpu = (float*)malloc(size * sizeof(float));
    {
        std::ifstream zfile("z.bin", std::ios::binary);
        zfile.read((char*)z_cpu, size * sizeof(float));
    }

    // Allocate device memory
    float* d_z;
    float* d_a_original;
    float* d_a_replacement;

    CUDA_CHECK(cudaMalloc(&d_z, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a_original, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a_replacement, size * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_z, z_cpu, size * sizeof(float), cudaMemcpyHostToDevice));

    // Create stream and events
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ===== Original Kernel =====
    printf("\n=== Original Kernel (grid-based) ===\n");

    // Warmup
    printf("Warmup (10 iterations)...\n");
    for (int i = 0; i < 10; i++) {
        relu_forward_original(size, d_a_original, d_z, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timed iterations
    std::vector<float> times_original;
    printf("Timing (100 iterations)...\n");
    for (int i = 0; i < 100; i++) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        relu_forward_original(size, d_a_original, d_z, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed = get_elapsed_ms(start, stop);
        times_original.push_back(elapsed);
    }

    // ===== Replacement Kernel (Thrust) =====
    printf("\n=== Replacement Kernel (thrust) ===\n");

    // Warmup
    printf("Warmup (10 iterations)...\n");
    for (int i = 0; i < 10; i++) {
        relu_forward_thrust(size, d_a_replacement, d_z, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timed iterations
    std::vector<float> times_replacement;
    printf("Timing (100 iterations)...\n");
    for (int i = 0; i < 100; i++) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        relu_forward_thrust(size, d_a_replacement, d_z, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed = get_elapsed_ms(start, stop);
        times_replacement.push_back(elapsed);
    }

    // ===== Analysis =====
    printf("\n=== Results ===\n");

    // Compute medians
    std::sort(times_original.begin(), times_original.end());
    std::sort(times_replacement.begin(), times_replacement.end());

    float median_original = times_original[50];
    float median_replacement = times_replacement[50];

    printf("Original median:     %.6f ms\n", median_original);
    printf("Replacement median:  %.6f ms\n", median_replacement);

    float speedup = median_original / median_replacement;
    printf("Speedup:             %.2f x\n", speedup);

    if (speedup > 1.0f) {
        printf("  -> Replacement is %.2f%% FASTER\n", (speedup - 1.0f) * 100.0f);
    } else {
        printf("  -> Replacement is %.2f%% SLOWER\n", (1.0f - speedup) * 100.0f);
    }

    // Cleanup
    free(z_cpu);
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_a_original));
    CUDA_CHECK(cudaFree(d_a_replacement));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
