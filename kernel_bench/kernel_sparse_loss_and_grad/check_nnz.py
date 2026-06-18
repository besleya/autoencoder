#!/usr/bin/env python3
import numpy as np

col_ptr = np.fromfile("col_ptr.bin", dtype=np.int32)
row_idx = np.fromfile("row_idx.bin", dtype=np.int32)
values = np.fromfile("values.bin", dtype=np.float32)

print(f"col_ptr has {len(col_ptr)} elements (should be B+1=33)")
print(f"row_idx has {len(row_idx)} elements")
print(f"values has {len(values)} elements")
print(f"col_ptr[-1] = {col_ptr[-1]} (should equal nnz)")

# Check if they're consistent
if col_ptr[-1] == len(row_idx) and col_ptr[-1] == len(values):
    print(f"\nnnz = {col_ptr[-1]} - all consistent")
else:
    print(f"\nERROR: Mismatch in sizes!")

print(f"\nFirst 15 col_ptr values: {col_ptr[:15]}")
print(f"Column 0 has {col_ptr[1] - col_ptr[0]} entries")
print(f"Column 0 entries: indices {col_ptr[0]} to {col_ptr[1]-1}")
