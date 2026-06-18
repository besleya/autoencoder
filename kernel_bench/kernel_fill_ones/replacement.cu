/*
   REPLACEMENT USING THRUST::FILL_N
   Library-based implementation for kernel_fill_ones
*/

#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>

/*
   fill_ones_thrust: Replace kernel_fill_ones with thrust::fill_n
   
   Usage:
     fill_ones_thrust(n, d_v, stream);
   
   Fills d_v[0..n-1] with 1.0f on the given stream.
   (cudaMemset cannot directly set float 1.0f, so thrust::fill_n is the cleanest approach.)
*/
void fill_ones_thrust(int n, float* v, cudaStream_t stream) {
    thrust::fill_n(thrust::cuda::par.on(stream), v, n, 1.0f);
}
