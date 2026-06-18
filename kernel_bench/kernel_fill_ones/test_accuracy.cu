/*
   TEST ACCURACY
   Load expected_v.bin, allocate device buffer, fill with garbage,
   run fill_ones_thrust, compare against expected.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <cstring>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

// Declare the implementation
void fill_ones_thrust(int n, float* v, cudaStream_t stream);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

int main(int argc, char** argv) {
    int n = 4096;
    
    // Read metadata
    FILE* meta_fp = fopen("meta.txt", "r");
    if (meta_fp) {
        fscanf(meta_fp, "n=%d\n", &n);
        fclose(meta_fp);
    }
    printf("Test size: n = %d\n", n);
    
    // Read expected from file
    float* expected = new float[n];
    FILE* fp = fopen("expected_v.bin", "rb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open expected_v.bin\n");
        return 1;
    }
    size_t nread = fread(expected, sizeof(float), n, fp);
    fclose(fp);
    if (nread != n) {
        fprintf(stderr, "Error: read %zu expected %zu floats\n", nread, (size_t)n);
        return 1;
    }
    
    // Allocate device buffer
    float* d_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_v, n * sizeof(float)));
    
    // Fill with garbage (7.7f using thrust::fill from host)
    // Use thrust::fill to fill with a garbage value first
    thrust::fill(thrust::device, d_v, d_v + n, 7.7f);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Create stream and run implementation
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    fill_ones_thrust(n, d_v, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    // Copy result back to host
    float* result = new float[n];
    CUDA_CHECK(cudaMemcpy(result, d_v, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compare
    bool pass = true;
    int mismatch_count = 0;
    float max_rel_err = 0.0f;
    float max_abs_err = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float expected_val = expected[i];
        float result_val = result[i];
        
        // Check bit-exact (all should be 1.0f)
        if (result_val != expected_val) {
            float abs_err = fabsf(result_val - expected_val);
            float rel_err = (expected_val != 0.0f) ? abs_err / fabsf(expected_val) : abs_err;
            
            if (abs_err > 1e-6f || rel_err > 1e-5f) {
                mismatch_count++;
                if (mismatch_count <= 5) {
                    printf("Mismatch at index %d: expected %f, got %f (abs_err=%e, rel_err=%e)\n",
                           i, expected_val, result_val, abs_err, rel_err);
                }
                pass = false;
            }
            max_abs_err = fmaxf(max_abs_err, abs_err);
            max_rel_err = fmaxf(max_rel_err, rel_err);
        }
    }
    
    if (pass) {
        printf("PASS: All %d values match expected (bit-exact = 1.0f)\n", n);
    } else {
        printf("FAIL: %d mismatches, max_abs_err=%e, max_rel_err=%e\n",
               mismatch_count, max_abs_err, max_rel_err);
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_v));
    delete[] expected;
    delete[] result;
    
    return pass ? 0 : 1;
}
