// Replacement kernel_adam_update using thrust::for_each_n
//
// Strategy: Use thrust::for_each_n with a device functor and zip_iterator
// to perform the Adam update in a single fused kernel call.
//
// Rationale for thrust over cuBLAS:
// - Adam update is fundamentally an element-wise operation with mixed arithmetic (scale, add, multiply, sqrt)
// - cuBLAS (Basic Linear Algebra Subroutines) provides matrix-vector operations:
//   * cublasSscal: scales vectors (can do beta1*m, beta2*v)
//   * cublasSaxpy: adds scaled vectors (can do (1-beta1)*g to m)
// - However, cuBLAS lacks element-wise operations like element-wise square (g*g) and sqrt
// - Multi-call cuBLAS approach would require 5+ kernel launches:
//   1. cublasSscal(m, beta1)
//   2. cublasSaxpy(m, (1-beta1)*g)
//   3. cublasSscal(v, beta2)
//   4. thrust for_each for v += (1-beta2)*g*g  [cuBLAS can't do element-wise square]
//   5. thrust for_each for p -= lr_t*m/(sqrt(v)+eps)  [cuBLAS lacks sqrt]
// - This causes excessive memory traffic and kernel launch overhead
// - thrust provides a library-style single-kernel abstraction with the same memory efficiency
//   as a hand-written kernel, achieving both code elegance and performance

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <cmath>

// ============================================================================
// Device functor for Adam update step
// ============================================================================
struct adam_update_functor {
    float lr_t;
    float beta1;
    float beta2;
    float eps;
    
    adam_update_functor(float lr_t_, float beta1_, float beta2_, float eps_)
        : lr_t(lr_t_), beta1(beta1_), beta2(beta2_), eps(eps_) {}
    
    // Called once per element; tuple contains (p, m, v, g)
    __device__ void operator()(thrust::tuple<float&, float&, float&, float> t) const {
        float& p = thrust::get<0>(t);
        float& m = thrust::get<1>(t);
        float& v = thrust::get<2>(t);
        const float g = thrust::get<3>(t);
        
        // Update first moment estimate
        m = beta1 * m + (1.0f - beta1) * g;
        
        // Update second moment estimate
        v = beta2 * v + (1.0f - beta2) * g * g;
        
        // Update parameter
        p -= lr_t * m / (sqrtf(v) + eps);
    }
};

// ============================================================================
// Replacement function using thrust
// ============================================================================
void adam_update_thrust(int size, float* p, float* m, float* v, const float* g,
                        float lr_t, float beta1, float beta2, float eps,
                        cudaStream_t stream) {
    thrust::device_ptr<float> dp(p);
    thrust::device_ptr<float> dm(m);
    thrust::device_ptr<float> dv(v);
    thrust::device_ptr<const float> dg(g);
    
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(dp, dm, dv, dg));
    
    adam_update_functor functor(lr_t, beta1, beta2, eps);
    
    // Execute on the given stream
    thrust::for_each(thrust::cuda::par.on(stream), zip_begin, zip_begin + size, functor);
}
