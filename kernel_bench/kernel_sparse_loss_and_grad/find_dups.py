#!/usr/bin/env python3
import numpy as np

d0 = 64
B = 32

col_ptr = np.fromfile("col_ptr.bin", dtype=np.int32)
row_idx = np.fromfile("row_idx.bin", dtype=np.int32)
values = np.fromfile("values.bin", dtype=np.float32)
nnz = len(values)

print(f"Looking for position (10, 0) in sparse matrix...")
print()

for k in range(nnz):
    # Find column j
    j = 0
    for i in range(B):
        if col_ptr[i] <= k < col_ptr[i + 1]:
            j = i
            break
    r = row_idx[k]
    
    if r == 10 and j == 0:
        print(f"Entry k={k}: row={r}, col={j}, value={values[k]}")

print()
print("All entries in column 0 (first 20):")
k = 0
for i in range(B):
    if i < 2:  # Print first 2 columns
        for idx in range(col_ptr[i], col_ptr[i+1]):
            print(f"  Entry {idx}: row={row_idx[idx]}, col={i}, value={values[idx]}")
