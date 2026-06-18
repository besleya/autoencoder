// Original kernel_adam_update implementation (from layer.cu, lines 97-107 and launcher ~lines 274, 286)
#include <cuda_runtime.h>
#include <cmath>

// ============================================================================
// kernel_adam_update: Original CUDA kernel
// ============================================================================
// Per-element update:
//   m[i] = beta1*m[i] + (1-beta1)*g[i]
//   v[i] = beta2*v[i] + (1-beta2)*g[i]*g[i]
//   p[i] -= lr_t * m[i] / (sqrt(v[i]) + eps)
//
__global__ void kernel_adam_update(
    int size, float* p, float* m, float* v, const float* g,
    float lr_t, float beta1, float beta2, float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g_val = g[idx];
        float m_val = beta1 * m[idx] + (1.0f - beta1) * g_val;
        float v_val = beta2 * v[idx] + (1.0f - beta2) * g_val * g_val;
        
        m[idx] = m_val;
        v[idx] = v_val;
        
        p[idx] -= lr_t * m_val / (sqrtf(v_val) + eps);
    }
}

// ============================================================================
// Helper function to launch the original kernel
// ============================================================================
void adam_update_original(int size, float* p, float* m, float* v, const float* g,
                          float lr_t, float beta1, float beta2, float eps,
                          cudaStream_t stream) {
    // Standard grid/block calculation (same as in layer.cu)
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    kernel_adam_update<<<grid_size, block_size, 0, stream>>>(
        size, p, m, v, g, lr_t, beta1, beta2, eps);
}
