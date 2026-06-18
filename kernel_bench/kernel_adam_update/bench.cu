// Benchmarking for original vs thrust implementation
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>

// Forward declarations
void adam_update_original(int size, float* p, float* m, float* v, const float* g,
                          float lr_t, float beta1, float beta2, float eps,
                          cudaStream_t stream);

void adam_update_thrust(int size, float* p, float* m, float* v, const float* g,
                        float lr_t, float beta1, float beta2, float eps,
                        cudaStream_t stream);

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// GPU Timer using CUDA events
// ============================================================================
struct GpuTimer {
    cudaEvent_t start, stop;
    
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void tic() {
        CUDA_CHECK(cudaEventRecord(start, 0));
    }
    
    float toc() {
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ============================================================================
// Benchmark a single implementation
// ============================================================================
void benchmark_impl(const char* name,
                   void (*update_func)(int, float*, float*, float*, const float*,
                                      float, float, float, float, cudaStream_t),
                   int size,
                   const float* p_init, const float* m_init, const float* v_init,
                   const float* g,
                   float lr_t, float beta1, float beta2, float eps,
                   int n_warmup, int n_timed) {
    printf("\n=== Benchmarking %s ===\n", name);
    
    // Allocate device memory for actual data
    float *d_p, *d_m, *d_v, *d_g;
    CUDA_CHECK(cudaMalloc(&d_p, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g, size * sizeof(float)));
    
    // Allocate device memory for reset copies (to re-init after each iter)
    float *d_p_init, *d_m_init, *d_v_init;
    CUDA_CHECK(cudaMalloc(&d_p_init, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_init, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_init, size * sizeof(float)));
    
    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_p_init, p_init, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m_init, m_init, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_init, v_init, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g, g, size * sizeof(float), cudaMemcpyHostToDevice));
    
    GpuTimer timer;
    
    // Warmup
    printf("Warmup (%d iterations)...\n", n_warmup);
    for (int iter = 0; iter < n_warmup; ++iter) {
        // Reset p, m, v from init copies
        CUDA_CHECK(cudaMemcpy(d_p, d_p_init, size * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_m, d_m_init, size * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, d_v_init, size * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // Run
        update_func(size, d_p, d_m, d_v, d_g, lr_t, beta1, beta2, eps, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed runs
    printf("Timing (%d iterations)...\n", n_timed);
    std::vector<float> times(n_timed);
    for (int iter = 0; iter < n_timed; ++iter) {
        // Reset p, m, v from init copies
        CUDA_CHECK(cudaMemcpy(d_p, d_p_init, size * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_m, d_m_init, size * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, d_v_init, size * sizeof(float), cudaMemcpyDeviceToDevice));
        
        timer.tic();
        update_func(size, d_p, d_m, d_v, d_g, lr_t, beta1, beta2, eps, 0);
        times[iter] = timer.toc();
    }
    
    // Compute statistics
    std::sort(times.begin(), times.end());
    float median_ms = times[n_timed / 2];
    float min_ms = times[0];
    float max_ms = times[n_timed - 1];
    float mean_ms = 0;
    for (float t : times) mean_ms += t;
    mean_ms /= n_timed;
    
    printf("%s Results:\n", name);
    printf("  Min:    %.3f ms\n", min_ms);
    printf("  Median: %.3f ms\n", median_ms);
    printf("  Mean:   %.3f ms\n", mean_ms);
    printf("  Max:    %.3f ms\n", max_ms);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_p_init));
    CUDA_CHECK(cudaFree(d_m_init));
    CUDA_CHECK(cudaFree(d_v_init));
}

// ============================================================================
// Main benchmark
// ============================================================================
int main() {
    printf("Adam Update Benchmark\n");
    printf("=====================\n");
    
    // Read metadata
    FILE* meta_file = fopen("meta.txt", "r");
    if (!meta_file) {
        fprintf(stderr, "Failed to open meta.txt\n");
        return 1;
    }
    
    int size;
    float lr_t, beta1, beta2, eps;
    fscanf(meta_file, "size=%d\n", &size);
    fscanf(meta_file, "lr_t=%f\n", &lr_t);
    fscanf(meta_file, "beta1=%f\n", &beta1);
    fscanf(meta_file, "beta2=%f\n", &beta2);
    fscanf(meta_file, "eps=%f\n", &eps);
    fclose(meta_file);
    
    printf("Parameters: size=%d, lr_t=%.2e, beta1=%.2f, beta2=%.4f, eps=%.2e\n",
           size, lr_t, beta1, beta2, eps);
    
    // Allocate host arrays
    float* p_init = new float[size];
    float* m_init = new float[size];
    float* v_init = new float[size];
    float* g = new float[size];
    
    // Read binary files
    auto read_file = [](const char* filename, float* data, int size) {
        FILE* f = fopen(filename, "rb");
        if (!f) {
            fprintf(stderr, "Failed to open %s\n", filename);
            exit(1);
        }
        size_t nread = fread(data, sizeof(float), size, f);
        if (nread != (size_t)size) {
            fprintf(stderr, "Error reading %s\n", filename);
            exit(1);
        }
        fclose(f);
    };
    
    read_file("p_init.bin", p_init, size);
    read_file("m_init.bin", m_init, size);
    read_file("v_init.bin", v_init, size);
    read_file("g.bin", g, size);
    
    int n_warmup = 10;
    int n_timed = 100;
    
    // Benchmark both implementations
    benchmark_impl("Original Kernel", adam_update_original,
                  size, p_init, m_init, v_init, g,
                  lr_t, beta1, beta2, eps,
                  n_warmup, n_timed);
    
    benchmark_impl("Thrust Replacement", adam_update_thrust,
                  size, p_init, m_init, v_init, g,
                  lr_t, beta1, beta2, eps,
                  n_warmup, n_timed);
    
    printf("\n=== Done ===\n");
    
    // Cleanup
    delete[] p_init;
    delete[] m_init;
    delete[] v_init;
    delete[] g;
    
    return 0;
}
