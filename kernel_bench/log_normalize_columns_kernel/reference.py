#!/mnt/home/besleya/s3/bin/python
# SPDX-License-Identifier: MIT
# reference.py — Generate CSC sparse matrix test data and compute expected output

import numpy as np
import torch
import struct
import random

random.seed(42)
np.random.seed(42)

n_cols = 256
nnz_per_col_min = 0
nnz_per_col_max = 40
scaler = 10000.0

# Generate random nnz per column
nnz_per_col = np.random.randint(nnz_per_col_min, nnz_per_col_max + 1, size=n_cols, dtype=np.int32)
total_nnz = int(np.sum(nnz_per_col))

print(f"n_cols: {n_cols}")
print(f"nnz_per_col: min={nnz_per_col.min()}, max={nnz_per_col.max()}, mean={nnz_per_col.mean():.2f}")
print(f"total_nnz: {total_nnz}")

# Build col_ptr
col_ptr = np.zeros(n_cols + 1, dtype=np.int32)
col_ptr[1:] = np.cumsum(nnz_per_col)

print(f"col_ptr[0]: {col_ptr[0]}, col_ptr[-1]: {col_ptr[-1]}")

# Generate values: float32 in [0, 100]
values_init = np.random.uniform(0, 100, size=total_nnz).astype(np.float32)

print(f"values_init: min={values_init.min():.4f}, max={values_init.max():.4f}, mean={values_init.mean():.4f}")

# Compute expected output on CPU with torch
values_torch = torch.from_numpy(values_init.copy()).float()
expected_values = values_torch.clone()

for col in range(n_cols):
    start = col_ptr[col]
    end = col_ptr[col + 1]
    s = expected_values[start:end].sum().item()
    if s > 0.0:
        inv = scaler / s
        expected_values[start:end] = torch.log1p(expected_values[start:end] * inv)

expected_values = expected_values.numpy().astype(np.float32)

print(f"expected_values: min={expected_values.min():.6f}, max={expected_values.max():.6f}, mean={expected_values.mean():.6f}")

# Write binary files
with open('col_ptr.bin', 'wb') as f:
    f.write(col_ptr.tobytes())
print("Wrote col_ptr.bin")

with open('values_init.bin', 'wb') as f:
    f.write(values_init.tobytes())
print("Wrote values_init.bin")

with open('scaler.bin', 'wb') as f:
    scaler_arr = np.array([scaler], dtype=np.float32)
    f.write(scaler_arr.tobytes())
print("Wrote scaler.bin")

with open('expected_values.bin', 'wb') as f:
    f.write(expected_values.tobytes())
print("Wrote expected_values.bin")

with open('meta.txt', 'w') as f:
    f.write(f"{n_cols}\n")
    f.write(f"{total_nnz}\n")
    f.write(f"{scaler}\n")
print("Wrote meta.txt")

print("\nTest data generation complete.")
