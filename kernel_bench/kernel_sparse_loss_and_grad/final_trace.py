#!/usr/bin/env python3
import numpy as np

d0 = 64
B = 32

a_L = np.fromfile("a_L.bin", dtype=np.float32).reshape(d0, B)
col_ptr = np.fromfile("col_ptr.bin", dtype=np.int32)
row_idx = np.fromfile("row_idx.bin", dtype=np.int32)
values = np.fromfile("values.bin", dtype=np.float32)
nnz = len(values)

print(f"a_L[10, 0] = {a_L[10, 0]}")
print(f"\nPosition (10, 0) has TWO sparse entries:")

pre_grad = a_L / B
expected_grad = pre_grad.copy()

for k in range(nnz):
    # Find column j
    j = 0
    for i in range(B):
        if col_ptr[i] <= k < col_ptr[i + 1]:
            j = i
            break
    r = row_idx[k]
    
    if r == 10 and j == 0:
        new_val = (a_L[r, j] - values[k]) / B
        print(f"k={k}: expected_grad[10,0] = ({a_L[10,0]} - {values[k]}) / 32 = {new_val}")
        expected_grad[r, j] = new_val

print(f"\nFinal expected_grad[10, 0] = {expected_grad[10, 0]}")

expected_grad_file = np.fromfile("expected_grad.bin", dtype=np.float32).reshape(d0, B)
print(f"Value in file:       expected_grad_file[10, 0] = {expected_grad_file[10, 0]}")
