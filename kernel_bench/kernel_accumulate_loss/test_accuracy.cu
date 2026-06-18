// Test accuracy of kernel_accumulate_loss replacement
// Compares original kernel vs replacement against expected result

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

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

// Helper: read float from binary file
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

// Helper: compare floats
bool compare_floats(float computed, float expected, float rel_tol=1e-5f, float abs_tol=1e-6f) {
    float diff = fabsf(computed - expected);
    float rel_err = fabsf(expected) > 1e-10f ? diff / fabsf(expected) : diff;
    return (diff <= abs_tol) || (rel_err <= rel_tol);
}

int main() {
    const int d0 = 64;
    const int B = 32;

    // Read input data
    float h_d_dot = read_float_bin("d_dot.bin");
    float h_d_loss = read_float_bin("d_loss.bin");
    float h_epoch_sum_init = read_float_bin("epoch_sum_init.bin");
    float h_expected = read_float_bin("expected_epoch_sum.bin");

    printf("=== Test Accuracy: kernel_accumulate_loss ===\n");
    printf("Parameters: d0=%d, B=%d, denom=%d\n", d0, B, d0*B);
    printf("Inputs: d_dot=%.6f, d_loss=%.6f, epoch_sum_init=%.6f\n", h_d_dot, h_d_loss, h_epoch_sum_init);
    printf("Expected: %.6f\n\n", h_expected);

    // Allocate device memory
    float *d_dot, *d_loss, *d_epoch_sum;
    CUDA_CHECK(cudaMalloc(&d_dot, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_epoch_sum, sizeof(float)));

    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // Test 1: Original kernel
    {
        printf("Test 1: Original kernel\n");
        
        // Copy inputs to device
        CUDA_CHECK(cudaMemcpy(d_dot, &h_d_dot, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_loss, &h_d_loss, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_epoch_sum, &h_epoch_sum_init, sizeof(float), cudaMemcpyHostToDevice));

        // Run kernel
        kernel_accumulate_loss_launch_original(d_dot, d_loss, d_epoch_sum, d0, B, 0);

        // Copy result back
        float h_result_original;
        CUDA_CHECK(cudaMemcpy(&h_result_original, d_epoch_sum, sizeof(float), cudaMemcpyDeviceToHost));

        printf("  Result: %.6f\n", h_result_original);
        
        if (compare_floats(h_result_original, h_expected)) {
            printf("  Status: PASS\n\n");
        } else {
            float diff = fabsf(h_result_original - h_expected);
            float rel_err = fabsf(h_expected) > 1e-10f ? diff / fabsf(h_expected) : diff;
            printf("  Status: FAIL (diff=%.2e, rel_err=%.2e)\n\n", diff, rel_err);
        }
    }

    // Test 2: Replacement (host accumulate)
    {
        printf("Test 2: Replacement (host accumulate)\n");
        
        // Copy inputs to device
        CUDA_CHECK(cudaMemcpy(d_dot, &h_d_dot, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_loss, &h_d_loss, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_epoch_sum, &h_epoch_sum_init, sizeof(float), cudaMemcpyHostToDevice));

        // Run replacement
        kernel_accumulate_loss_replacement(d_dot, d_loss, d_epoch_sum, d0, B, cublas_handle, 0);

        // Copy result back
        float h_result_replacement;
        CUDA_CHECK(cudaMemcpy(&h_result_replacement, d_epoch_sum, sizeof(float), cudaMemcpyDeviceToHost));

        printf("  Result: %.6f\n", h_result_replacement);
        
        if (compare_floats(h_result_replacement, h_expected)) {
            printf("  Status: PASS\n\n");
        } else {
            float diff = fabsf(h_result_replacement - h_expected);
            float rel_err = fabsf(h_expected) > 1e-10f ? diff / fabsf(h_expected) : diff;
            printf("  Status: FAIL (diff=%.2e, rel_err=%.2e)\n\n", diff, rel_err);
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_dot));
    CUDA_CHECK(cudaFree(d_loss));
    CUDA_CHECK(cudaFree(d_epoch_sum));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));

    printf("=== All tests completed ===\n");
    return 0;
}
