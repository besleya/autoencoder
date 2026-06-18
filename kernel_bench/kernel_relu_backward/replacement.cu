// Thrust-based replacement for kernel_relu_backward
//
// Algorithm: For each element, apply ReLU backward gate:
//   dz[i] = (z[i] > 0.0) ? dz[i] : 0.0
// Implemented using thrust::transform with element-wise multiplication

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

void relu_backward_thrust(int size, float* dz, const float* z, cudaStream_t stream) {
    // Thrust transform: multiply each dz[i] by gate function result
    // gate(z[i]) = (z[i] > 0.0f) ? 1.0f : 0.0f
    thrust::transform(
        thrust::cuda::par.on(stream),
        dz, dz + size,           // input range (gradient buffer)
        z,                        // secondary input range (activation values)
        dz,                       // output range (in-place)
        [] __device__ (float g, float zv) {
            return zv > 0.0f ? g : 0.0f;
        }
    );
}
