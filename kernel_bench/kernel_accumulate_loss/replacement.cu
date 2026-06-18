// Library-based replacement for kernel_accumulate_loss
// Approach: Use cudaMemcpyAsync to move values to host, accumulate, move back
// This avoids kernel launch overhead and single-thread serialization

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstring>

// Approach B: Host-based accumulation using cudaMemcpyAsync
// Pros: Avoids kernel launch overhead, minimal device memory usage
// Cons: Host<->Device transfers, but for single floats this is negligible
void kernel_accumulate_loss_replacement_host_accumulate(const float* d_dot,
                                                         const float* d_loss,
                                                         float* d_epoch_sum,
                                                         int d0, int B,
                                                         cublasHandle_t handle,
                                                         cudaStream_t stream = 0) {
    // Allocate pinned host memory for transfers
    float *h_dot = nullptr, *h_loss = nullptr, *h_epoch_sum = nullptr;
    cudaMallocHost(&h_dot, sizeof(float));
    cudaMallocHost(&h_loss, sizeof(float));
    cudaMallocHost(&h_epoch_sum, sizeof(float));

    // D2H: copy dot and loss
    cudaMemcpyAsync(h_dot, d_dot, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_epoch_sum, d_epoch_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Host-side accumulation
    float denom = static_cast<float>(d0) * static_cast<float>(B);
    *h_epoch_sum += (*h_dot + *h_loss) / denom;

    // H2D: copy result back
    cudaMemcpyAsync(d_epoch_sum, h_epoch_sum, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // Cleanup
    cudaFreeHost(h_dot);
    cudaFreeHost(h_loss);
    cudaFreeHost(h_epoch_sum);
}

// Approach A: Two cublasSaxpy_v2 calls
// Pros: Fully GPU-based, predictable
// Cons: Two cublas calls, overhead of saxpy for single elements
void kernel_accumulate_loss_replacement_saxpy(const float* d_dot,
                                               const float* d_loss,
                                               float* d_epoch_sum,
                                               int d0, int B,
                                               cublasHandle_t handle,
                                               cudaStream_t stream = 0) {
    float scale = 1.0f / (static_cast<float>(d0) * static_cast<float>(B));
    int n = 1;

    // d_epoch_sum += scale * d_dot
    cublasSaxpy_v2(handle, n, &scale, d_dot, 1, d_epoch_sum, 1);

    // d_epoch_sum += scale * d_loss
    cublasSaxpy_v2(handle, n, &scale, d_loss, 1, d_epoch_sum, 1);
}

// Default replacement: use host accumulation (typically faster for single-element kernels)
void kernel_accumulate_loss_replacement(const float* d_dot,
                                         const float* d_loss,
                                         float* d_epoch_sum,
                                         int d0, int B,
                                         cublasHandle_t handle,
                                         cudaStream_t stream = 0) {
    kernel_accumulate_loss_replacement_host_accumulate(d_dot, d_loss, d_epoch_sum, d0, B, handle, stream);
}
