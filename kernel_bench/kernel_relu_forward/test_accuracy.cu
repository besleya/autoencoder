/**
 * Accuracy test for kernel_relu_forward replacement.
 * 
 * Loads z.bin and expected_a.bin, runs both original and replacement kernels,
 * and verifies they match to within specified tolerances.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>

// Forward declarations (defined in original.cu and replacement.cu)
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

int main() {
    // Read metadata
    int size;
    {
        std::ifstream meta("meta.txt");
        meta >> size;
        printf("Test size: %d\n", size);
    }

    // Load z.bin and expected_a.bin from disk
    float* z_cpu = (float*)malloc(size * sizeof(float));
    float* expected_a_cpu = (float*)malloc(size * sizeof(float));

    {
        std::ifstream zfile("z.bin", std::ios::binary);
        zfile.read((char*)z_cpu, size * sizeof(float));
    }

    {
        std::ifstream afile("expected_a.bin", std::ios::binary);
        afile.read((char*)expected_a_cpu, size * sizeof(float));
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

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Run original kernel
    printf("Running original kernel...\n");
    relu_forward_original(size, d_a_original, d_z, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Run replacement kernel
    printf("Running replacement kernel...\n");
    relu_forward_thrust(size, d_a_replacement, d_z, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy results back
    float* a_original_cpu = (float*)malloc(size * sizeof(float));
    float* a_replacement_cpu = (float*)malloc(size * sizeof(float));

    CUDA_CHECK(cudaMemcpy(a_original_cpu, d_a_original, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_replacement_cpu, d_a_replacement, size * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare original vs replacement
    printf("Comparing original vs replacement...\n");
    int errors = 0;
    for (int i = 0; i < size; i++) {
        float diff = fabs(a_original_cpu[i] - a_replacement_cpu[i]);
        if (diff > 1e-5) {
            errors++;
            if (errors <= 5) {
                printf("  Error at idx %d: original=%f, replacement=%f, diff=%e\n",
                       i, a_original_cpu[i], a_replacement_cpu[i], diff);
            }
        }
    }
    printf("Original vs Replacement: %d mismatches\n", errors);

    // Compare replacement vs expected
    printf("Comparing replacement vs expected...\n");
    float rel_tol = 1e-5f;
    float abs_tol = 1e-6f;
    int mismatch = 0;
    for (int i = 0; i < size; i++) {
        float diff = fabs(a_replacement_cpu[i] - expected_a_cpu[i]);
        float rel_diff = diff / (fabs(expected_a_cpu[i]) + 1e-10f);
        if (diff > abs_tol && rel_diff > rel_tol) {
            mismatch++;
            if (mismatch <= 5) {
                printf("  Error at idx %d: got=%f, expected=%f, diff=%e, rel_diff=%e\n",
                       i, a_replacement_cpu[i], expected_a_cpu[i], diff, rel_diff);
            }
        }
    }

    if (errors == 0 && mismatch == 0) {
        printf("\n=== PASS: All tests passed ===\n");
    } else {
        printf("\n=== FAIL: %d mismatches with expected ===\n", mismatch);
        exit(1);
    }

    // Cleanup
    free(z_cpu);
    free(expected_a_cpu);
    free(a_original_cpu);
    free(a_replacement_cpu);
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_a_original));
    CUDA_CHECK(cudaFree(d_a_replacement));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}
