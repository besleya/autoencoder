// SPDX-License-Identifier: MIT
// batch.h — Transient abstraction of one training batch.

#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "slot.h"
#include "ring.h"

// Forward declaration
class Slot;

// Transient abstraction of one training batch. Owns no memory;
// borrows buffers and stream from a Slot. Move-only.
class Batch {
public:
    // Constructor: takes ownership of a FILLING slot, metadata, and host-side CSC data.
    // The caller must have already called slot->mark_filling() before constructing Batch.
    // The CSC data (col_ptr, row_idx, values) is copied from host into the slot's
    // pinned buffers during construction.
    // 
    // Arguments:
    //   slot          - non-owning pointer to a Slot in FILLING state; slot must outlive prepare()
    //   species_name  - human-readable species identifier
    //   col_ptr       - host array of length B+1, CSC column pointers
    //   row_idx       - host array of length nnz, CSC row indices
    //   values        - host array of length nnz, CSC values
    //   m             - number of rows (features) in the batch
    //   B             - number of columns (batch size)
    //   nnz           - number of nonzeros in the batch
    //   chunk_end     - true if this is the last batch of a chunk
    Batch(Slot* slot,
          std::string species_name,
          const int32_t* col_ptr,
          const int32_t* row_idx,
          const float* values,
          int m,
          int B,
          int nnz,
          bool chunk_end);

    // Destructor: marks the slot FREE if still bound (non-null).
    ~Batch();

    // Move-only semantics
    Batch(Batch&& other) noexcept;
    Batch& operator=(Batch&& other) noexcept;
    Batch(const Batch&) = delete;
    Batch& operator=(const Batch&) = delete;

    // Prepare the batch: copy CSC from pinned to device, launch log-normalize kernel,
    // record ready event, and mark slot READY. Idempotent-safe (only call once per Batch).
    void prepare();

    // Accessors
    const std::string& species_name() const;
    bool               chunk_end() const;
    int                m() const;       // number of rows (features)
    int                B() const;       // number of columns (batch size)
    int                nnz() const;     // number of nonzeros
    cudaEvent_t        ready_event() const;

    // Return a SparseBatch struct (from ring.h) populated with device pointers and shape.
    // The SparseBatch::ready_event points to slot_->ready_event().
    SparseBatch sparse_view() const;

private:
    Slot* slot_;                // non-owning, nullable after move
    std::string species_name_;  // human-readable species identifier
    int m_;                     // number of rows (features)
    int B_;                     // number of columns (batch size)
    int nnz_;                   // number of nonzeros
    bool chunk_end_;            // true if last batch of chunk
};
