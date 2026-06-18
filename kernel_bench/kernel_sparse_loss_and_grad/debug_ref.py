#!/usr/bin/env python3
import numpy as np
import struct

# Load metadata
with open("meta.txt", "r") as f:
    lines = f.readlines()
    d0 = int(lines[0].split('=')[1])
    B = int(lines[1].split('=')[1])
    nnz = int(lines[2].split('=')[1])

print(f"d0={d0}, B={B}, nnz={nnz}")

# Load data
a_L = np.fromfile("a_L.bin", dtype=np.float32).reshape(d0, B)
pre_grad = np.fromfile("pre_grad.bin", dtype=np.float32).reshape(d0, B)
col_ptr = np.fromfile("col_ptr.bin", dtype=np.int32)
row_idx = np.fromfile("row_idx.bin", dtype=np.int32)
values = np.fromfile("values.bin", dtype=np.float32)
expected_grad = np.fromfile("expected_grad.bin", dtype=np.float32).reshape(d0, B)

print(f"\nFirst few a_L values: {a_L[0, :5]}")
print(f"First few pre_grad values: {pre_grad[0, :5]}")
print(f"First few expected_grad values: {expected_grad[0, :5]}")

# Check a specific position (row 10, col 0 = index 320)
r, j = 10, 0
idx = r * B + j
print(f"\nPosition ({r}, {j}) = index {idx}:")
print(f"  a_L[{r}, {j}] = {a_L[r, j]}")
print(f"  pre_grad[{r}, {j}] = {pre_grad[r, j]}")
print(f"  expected_grad[{r}, {j}] = {expected_grad[r, j]}")

# Check if this position is sparse
is_sparse = False
for k in range(nnz):
    # Find which column entry k belongs to
    col = 0
    for i in range(B):
        if col_ptr[i] <= k < col_ptr[i + 1]:
            col = i
            break
    if row_idx[k] == r and col == j:
        print(f"  This IS a sparse entry: k={k}, row={row_idx[k]}, col={col}, value={values[k]}")
        expected_val = (a_L[r, j] - values[k]) / B
        print(f"  Expected: ({a_L[r, j]} - {values[k]}) / {B} = {expected_val}")
        is_sparse = True
        break

if not is_sparse:
    print(f"  This is NOT a sparse entry, expected = a_L / B = {a_L[r, j] / B}")

print(f"\n  Pre_grad value:  {pre_grad[r, j]}")
print(f"  Expected value:  {expected_grad[r, j]}")
