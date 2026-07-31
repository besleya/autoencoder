// SPDX-License-Identifier: MIT
// batch.h — Transient abstraction of one training batch.

#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <stdexcept>
#include "slot.h"

// Forward declaration
class Slot;

// Chunk — a concatenation of one or more singlet::pz::ReadResult objects along
// columns (cells). DataLoader owns/produces this; Batch only ever reads from
// it and assumes it remains valid for Batch's entire lifetime.
struct Chunk {
    int m = 0;                       // rows/features, same for the whole chunk
    int n = 0;                       // total columns across concatenated files
    std::vector<uint32_t> col_ptr;   // length n+1, cumulative nnz offsets (rebased across files)
    std::vector<uint32_t> row_idx;   // length nnz, row/gene index per nonzero
    std::vector<uint32_t> values;    // length nnz, raw widened counts (from read_1pz)
};

// SparseView — plain aggregate providing device pointers and shape for consumption.
struct SparseView {
    int m = 0, B = 0, nnz = 0;
    const int32_t* d_col_ptr = nullptr;
    const int32_t* d_row_idx = nullptr;
    const float* d_values = nullptr;
};

// Transient abstraction of one training batch. Owns no memory;
// borrows buffers and stream from a Slot. Move-only.
class Batch {
public:
    // Constructor: cheap initialization. Stores references to chunk and slot; does not
    // gather, allocate, or copy data. All heavy lifting deferred to prepare().
    //
    // Arguments:
    //   slot             - non-owning pointer to a Slot; must outlive Batch
    //   species_name     - human-readable species identifier
    //   chunk            - shared_ptr to source Chunk; Batch holds a copy to ensure lifetime safety
    //   column_indices   - length B, column positions into chunk to gather
    //   chunk_end        - true if this is the last batch of the chunk
    //   scale            - scale factor for log-normalization (default 10000.0f)
    Batch(Slot* slot,
          std::string species_name,
          std::shared_ptr<const Chunk> chunk,
          std::vector<int> column_indices,
          bool chunk_end,
          float scale = 10000.0f);

    // Destructor: marks the slot EMPTY if still bound (non-null).
    ~Batch();

    // Sets the consumer stream (stream on which trainer reads device buffers).
    // Call this before Batch is destroyed, so destructor can pass it to mark_empty().
    void set_consumer_stream(cudaStream_t stream);

    // Move-only semantics
    Batch(Batch&& other) noexcept;
    Batch& operator=(Batch&& other) noexcept;
    Batch(const Batch&) = delete;
    Batch& operator=(const Batch&) = delete;

    // Gathers this batch's columns out of the chunk, casts types, CPU log-normalizes.
    // Chunk data is not touched after this returns. Returns total nnz.
    int gather();

    // Copies gathered data to device and records ready event.
    void send_to_device();

    // Convenience wrapper: calls gather() then send_to_device().
    void prepare();

    // Accessors
    const std::string& species_name() const;
    bool               chunk_end() const;
    int                m() const;       // number of rows (features)
    int                B() const;       // number of columns (batch size)
    int                nnz() const;     // number of nonzeros
    SparseView         sparse_view() const;
    cudaEvent_t        ready_event() const;

private:
    Slot* slot_;                       // non-owning, nullable after move
    std::string species_name_;         // human-readable species identifier
    std::shared_ptr<const Chunk> chunk_;  // owns reference to source data
    std::vector<int> column_indices_;  // length B_, column positions into chunk_
    int m_ = 0;                        // number of rows (features)
    int B_ = 0;                        // number of columns (batch size)
    int nnz_ = 0;                      // number of nonzeros (computed in layout())
    bool chunk_end_;                   // true if last batch of chunk
    float scale_;                      // scale factor for log-normalization
    cudaStream_t consumer_stream_ = nullptr;  // stream on which trainer reads buffers (nullptr = default stream)

    // Builds this batch's own col_ptr (prefix sum) into slot's pinned col_ptr buffer;
    // returns total nnz. Does NOT copy row_idx/values yet.
    int layout();

    // Copies col_ptr, row_idx, values to device via cudaMemcpyAsync and records ready event.
    void to_device();
};
