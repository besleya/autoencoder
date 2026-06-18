#!/usr/bin/env python3
import numpy as np

col_ptr = np.fromfile("col_ptr.bin", dtype=np.int32)
row_idx = np.fromfile("row_idx.bin", dtype=np.int32)
values = np.fromfile("values.bin", dtype=np.float32)
expected_grad = np.fromfile("expected_grad.bin", dtype=np.float32).reshape(64, 32)

nnz = col_ptr[-1]
print(f"nnz = {nnz}")

# Count how many expected_grad values differ from pre_grad
pre_grad = np.fromfile("pre_grad.bin", dtype=np.float32).reshape(64, 32)
modified_count = np.sum(np.abs(expected_grad - pre_grad) > 1e-6)
print(f"Modified gradient positions: {modified_count}")

if modified_count == nnz:
    print(f"✓ All {nnz} sparse entries are represented in expected_grad (no duplicates!)")
else:
    print(f"✗ ERROR: Only {modified_count} out of {nnz} sparse entries are in expected_grad")

# Check for duplicate (row, col) pairs
print("\nChecking for duplicate sparse entries...")
positions = set()
duplicates = 0
for j in range(32):
    for k in range(col_ptr[j], col_ptr[j+1]):
        r = row_idx[k]
        if (r, j) in positions:
            duplicates += 1
        positions.add((r, j))

if duplicates == 0:
    print(f"✓ No duplicate sparse entries found")
else:
    print(f"✗ Found {duplicates} duplicate sparse entries")
