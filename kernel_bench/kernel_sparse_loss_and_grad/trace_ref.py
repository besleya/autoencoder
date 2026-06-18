#!/usr/bin/env python3
import numpy as np

# Replicate reference.py logic with numpy
d0 = 64
B = 32
np.random.seed(42)

A_L = np.random.randn(d0, B).astype(np.float32)

# Load sparse data
col_ptr = np.fromfile("col_ptr.bin", dtype=np.int32)
row_idx = np.fromfile("row_idx.bin", dtype=np.int32)
values = np.fromfile("values.bin", dtype=np.float32)
nnz = len(values)

# Compute expected_grad exactly as in reference.py
pre_grad = A_L / B
expected_grad = pre_grad.copy()

print(f"Before loop: expected_grad[10, 0] = {expected_grad[10, 0]}")

for k in range(nnz):
    # Find column j
    j = 0
    for i in range(B):
        if col_ptr[i] <= k < col_ptr[i + 1]:
            j = i
            break
    r = row_idx[k]
    v = values[k]
    
    if r == 10 and j == 0 and k < 5:
        print(f"k={k}: setting expected_grad[{r}, {j}] = ({A_L[r, j]:.6f} - {v:.6f}) / {B} = {(A_L[r, j] - v) / B:.6f}")
    
    expected_grad[r, j] = (A_L[r, j] - v) / B

print(f"After loop:  expected_grad[10, 0] = {expected_grad[10, 0]}")

# Now check what's actually in the file
expected_grad_file = np.fromfile("expected_grad.bin", dtype=np.float32).reshape(d0, B)
print(f"From file:   expected_grad_file[10, 0] = {expected_grad_file[10, 0]}")

# Check if they match
if abs(expected_grad[10, 0] - expected_grad_file[10, 0]) > 1e-6:
    print(f"MISMATCH! Expected {expected_grad[10, 0]} but got {expected_grad_file[10, 0]}")
