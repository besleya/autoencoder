#!/usr/bin/env python3
"""
Python reference implementation for kernel_sparse_loss_and_grad.
Generates random test data, computes expected outputs, and saves as .bin files.
"""

import numpy as np
import torch
import struct
import os

def main():
    # Configuration
    d0 = 64
    B = 32
    density = 0.05  # ~5% sparsity
    
    # Create output directory if needed
    os.makedirs(".", exist_ok=True)
    
    # ========================================================================
    # Step 1: Generate random A_L (d0 x B matrix, column-major)
    # ========================================================================
    np.random.seed(42)
    torch.manual_seed(42)
    
    A_L = torch.randn(d0, B, dtype=torch.float32)
    A_L_np = A_L.numpy()
    
    # ========================================================================
    # Step 2: Generate sparse targets Y in CSC format
    # ========================================================================
    # Generate CSC sparse matrix
    nnz = max(1, int(d0 * B * density))
    
    # Random entries
    col_ptr = [0]
    row_idx_list = []
    values_list = []
    
    for j in range(B):
        # Roughly nnz/B entries per column on average
        entries_in_col = max(0, int(np.random.poisson(nnz / B)))
        if entries_in_col == 0 and j == 0:
            entries_in_col = 1  # Ensure at least one entry
        
        used_rows = set()
        for _ in range(entries_in_col):
            # Keep trying until we get a unique row for this column
            max_attempts = 10
            for attempt in range(max_attempts):
                r = np.random.randint(0, d0)
                if r not in used_rows:
                    used_rows.add(r)
                    v = torch.randn(1, dtype=torch.float32).item()
                    row_idx_list.append(r)
                    values_list.append(v)
                    break
        
        col_ptr.append(len(row_idx_list))
    
    nnz = len(values_list)
    col_ptr = np.array(col_ptr, dtype=np.int32)
    row_idx = np.array(row_idx_list, dtype=np.int32)
    values = np.array(values_list, dtype=np.float32)
    
    Y_torch = torch.zeros(d0, B, dtype=torch.float32)
    for k in range(nnz):
        # Find column j: entry k belongs to column j if col_ptr[j] <= k < col_ptr[j+1]
        j = 0
        for i in range(B):
            if col_ptr[i] <= k < col_ptr[i + 1]:
                j = i
                break
        r = row_idx[k]
        Y_torch[r, j] = float(values[k])
    
    # ========================================================================
    # Step 3: Compute expected gradient
    # Expected behavior: caller pre-fills d_grad_loss = A_L / B
    # Then for each sparse entry: d_grad_loss[r, j] = (A_L[r, j] - v) / B
    # ========================================================================
    pre_grad = A_L / B
    expected_grad = pre_grad.clone()
    
    for k in range(nnz):
        # Find column j: entry k belongs to column j if col_ptr[j] <= k < col_ptr[j+1]
        j = 0
        for i in range(B):
            if col_ptr[i] <= k < col_ptr[i + 1]:
                j = i
                break
        r = row_idx[k]
        v = values[k]
        expected_grad[r, j] = (A_L[r, j] - v) / B
    
    # ========================================================================
    # Step 4: Compute expected loss
    # loss_acc = ||a_L||_F^2 + sum over sparse entries of (v^2 - 2*A_L[r,j]*v)
    # Then loss = loss_acc / (d0 * B)
    # ========================================================================
    # Initialize with Frobenius norm of A_L
    frobenius_norm_sq = torch.sum(A_L ** 2).item()
    expected_loss = frobenius_norm_sq
    
    for k in range(nnz):
        # Find column j: entry k belongs to column j if col_ptr[j] <= k < col_ptr[j+1]
        j = 0
        for i in range(B):
            if col_ptr[i] <= k < col_ptr[i + 1]:
                j = i
                break
        r = row_idx[k]
        v = values[k]
        a_val = A_L[r, j].item()
        expected_loss += v * v - 2.0 * a_val * v
    
    # Normalize by d0 * B
    expected_loss = expected_loss / (d0 * B)
    expected_loss_np = np.array([expected_loss], dtype=np.float32)
    
    # ========================================================================
    # Step 5: Write all inputs and outputs as binary files
    # ========================================================================
    
    # a_L.bin: float32, d0*B, row-major (standard PyTorch layout)
    with open("a_L.bin", "wb") as f:
        f.write(A_L_np.astype(np.float32).tobytes())
    
    # pre_grad.bin: float32, d0*B (the initial d_grad_loss)
    with open("pre_grad.bin", "wb") as f:
        f.write(pre_grad.numpy().astype(np.float32).tobytes())
    
    # col_ptr.bin: int32, B+1
    with open("col_ptr.bin", "wb") as f:
        f.write(col_ptr.tobytes())
    
    # row_idx.bin: int32, nnz
    with open("row_idx.bin", "wb") as f:
        f.write(row_idx.tobytes())
    
    # values.bin: float32, nnz
    with open("values.bin", "wb") as f:
        f.write(values.tobytes())
    
    # expected_grad.bin: float32, d0*B
    with open("expected_grad.bin", "wb") as f:
        f.write(expected_grad.numpy().astype(np.float32).tobytes())
    
    # expected_loss.bin: float32, 1
    with open("expected_loss.bin", "wb") as f:
        f.write(expected_loss_np.tobytes())
    
    # meta.txt: metadata
    with open("meta.txt", "w") as f:
        f.write(f"d0={d0}\n")
        f.write(f"B={B}\n")
        f.write(f"nnz={nnz}\n")
    
    print(f"Generated test data: d0={d0}, B={B}, nnz={nnz}, density={nnz/(d0*B):.2%}")
    print(f"Files: a_L.bin, pre_grad.bin, col_ptr.bin, row_idx.bin, values.bin, expected_grad.bin, expected_loss.bin, meta.txt")

if __name__ == "__main__":
    main()
