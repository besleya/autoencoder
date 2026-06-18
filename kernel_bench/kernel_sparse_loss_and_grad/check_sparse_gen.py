#!/usr/bin/env python3
import numpy as np

# Check the sparse matrix generation logic
np.random.seed(42)
d0 = 64
B = 32
density = 0.05
nnz = max(1, int(d0 * B * density))

col_ptr = [0]
row_idx_list = []
values_list = []

for j in range(B):
    entries_in_col = max(0, int(np.random.poisson(nnz / B)))
    if entries_in_col == 0 and j == 0:
        entries_in_col = 1

    for _ in range(entries_in_col):
        r = np.random.randint(0, d0)  # <-- CAN GENERATE DUPLICATES!
        v = np.random.randn(1).item()
        row_idx_list.append(r)
        values_list.append(v)

    col_ptr.append(len(row_idx_list))

# Check for duplicates in column 0
print("Column 0 entries:")
for idx in range(col_ptr[0], col_ptr[1]):
    print(f"  Entry {idx}: row={row_idx_list[idx]}")

# Check if there are duplicate rows in column 0
rows_in_col_0 = row_idx_list[col_ptr[0]:col_ptr[1]]
print(f"\nRows in column 0: {rows_in_col_0}")
if len(rows_in_col_0) != len(set(rows_in_col_0)):
    print("WARNING: Duplicate rows in column 0!")
