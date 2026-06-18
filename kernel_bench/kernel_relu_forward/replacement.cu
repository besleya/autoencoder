/**
 * Thrust-based replacement for kernel_relu_forward.
 * 
 * Uses thrust::transform with a device lambda to compute
 * the same operation as the original kernel.
 */

#include <cuda_runtime.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

/**
 * Thrust-based ReLU forward pass.
 * 
 * Computes a[i] = max(z[i], 0) using thrust::transform.
 * 
 * @param size    Number of elements
 * @param a       Output array (device memory, raw pointer)
 * @param z       Input array (device memory, raw pointer)
 * @param stream  CUDA stream for execution
 */
void relu_forward_thrust(int size, float* a, const float* z, cudaStream_t stream) {
    thrust::transform(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(z),
        thrust::device_pointer_cast(z) + size,
        thrust::device_pointer_cast(a),
        [] __device__ (float v) { return v > 0.0f ? v : 0.0f; }
    );
}
