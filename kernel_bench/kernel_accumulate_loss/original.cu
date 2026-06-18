// Original kernel_accumulate_loss
// Source: /mnt/home/besleya/autoencoder/gpu_autoencoder.cu, lines 87-97 and 358-361

#include <cuda_runtime.h>

// Accumulate per-batch loss into a device-side epoch sum.
// Computes: *d_epoch_sum += (*d_dot + *d_loss) / (d0 * B)
// Single-thread kernel (writing one scalar); launched once per batch.
__global__ void kernel_accumulate_loss(const float* d_dot,
                                       const float* d_loss,
                                       float* d_epoch_sum,
                                       int d0, int B) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float denom = static_cast<float>(d0) * static_cast<float>(B);
        *d_epoch_sum += (*d_dot + *d_loss) / denom;
    }
}

// Host launcher (original)
void kernel_accumulate_loss_launch_original(const float* d_dot,
                                            const float* d_loss,
                                            float* d_epoch_sum,
                                            int d0, int B,
                                            cudaStream_t stream = 0) {
    kernel_accumulate_loss<<<1, 1, 0, stream>>>(d_dot, d_loss, d_epoch_sum, d0, B);
}
