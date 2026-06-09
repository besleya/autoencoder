// SPDX-License-Identifier: MIT
// load_pz.h — load v3 pileup files to GPU memory.
//
// Implements load_pz_v3: a v3-specific variant of singlet::gpu::io::load_pz
// that uses singlet::pz::read_1pz for CPU-side decoding.

#pragma once

#include <singlet/gpu/io/pz_device_loader.h>
#include <cuda_runtime.h>
#include <string>

namespace singlet::gpu {
namespace io {

// Load a v3 .1pz file (or v1/v4 with identical layout) to GPU device memory.
//
// path: absolute path to the .1pz file
// stream: CUDA stream for cudaMemcpyAsync. If nullptr, a new high-priority
//         stream is created; the caller owns it and must destroy it.
// keep_host_pinned: if true, retain host-side CSC copies in the returned
//                   PzDeviceMatrix via shared_ptr fields (for SVD adapters).
//
// Returns PzDeviceMatrix with populated DeviceCSC, Metadata, and async copies
// in flight on the supplied (or created) stream.
//
// Throws std::runtime_error on file I/O, CUDA, or decode errors.
singlet::gpu::io::PzDeviceMatrix load_pz_v3(
    const std::string& path,
    cudaStream_t stream = nullptr,
    bool keep_host_pinned = false);

}  // namespace io
}  // namespace singlet::gpu
