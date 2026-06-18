#!/usr/bin/env python3
import numpy as np

# Load data
col_ptr = np.fromfile("col_ptr.bin", dtype=np.int32)
row_idx = np.fromfile("row_idx.bin", dtype=np.int32)
values = np.fromfile("values.bin", dtype=np.float32)

print("col_ptr:", col_ptr[:10])
print(f"\nEntry k=2:")
print(f"  row_idx[2] = {row_idx[2]}")
print(f"  values[2] = {values[2]}")

# Find which column entry k=2 belongs to
k = 2
for i in range(len(col_ptr)-1):
    if col_ptr[i] <= k < col_ptr[i+1]:
        print(f"  col_ptr[{i}] = {col_ptr[i]}, col_ptr[{i+1}] = {col_ptr[i+1]}")
        print(f"  So k={k} is in column {i}")
        break
