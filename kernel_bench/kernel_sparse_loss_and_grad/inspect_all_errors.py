#!/usr/bin/env python3
import numpy as np

d0 = 64
B = 32

pre_grad = np.fromfile("pre_grad.bin", dtype=np.float32).reshape(d0, B)
expected_grad = np.fromfile("expected_grad.bin", dtype=np.float32).reshape(d0, B)
col_ptr = np.fromfile("col_ptr.bin", dtype=np.int32)
row_idx = np.fromfile("row_idx.bin", dtype=np.int32)
values = np.fromfile("values.bin", dtype=np.float32)

print("Checking for ALL positions where expected_grad != pre_grad...")
print()

diffs = []
for r in range(d0):
    for j in range(B):
        if abs(expected_grad[r, j] - pre_grad[r, j]) > 1e-6:
            idx = r * B + j
            diffs.append((idx, r, j, pre_grad[r, j], expected_grad[r, j]))

print(f"Found {len(diffs)} positions where gradient was modified:")
print()

for idx, (linear_idx, r, j, pre_val, exp_val) in enumerate(diffs[:20]):
    print(f"Index {linear_idx} [{r}, {j}]: pre_grad={pre_val:.6f}, expected_grad={exp_val:.6f}")
    
    # Check if this position is sparse
    for k in range(len(values)):
        col = 0
        for i in range(B):
            if col_ptr[i] <= k < col_ptr[i+1]:
                col = i
                break
        if row_idx[k] == r and col == j:
            print(f"  Sparse entry k={k}: value={values[k]:.6f}")

print()
if len(diffs) == 102:
    print("All 102 sparse entries are present in expected_grad")
else:
    print(f"WARNING: Only {len(diffs)} out of 102 sparse entries are in expected_grad!")
