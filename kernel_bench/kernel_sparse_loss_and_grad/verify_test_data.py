#!/usr/bin/env python3
import numpy as np

d0 = 64
B = 32

a_L = np.fromfile("a_L.bin", dtype=np.float32).reshape(d0, B)
pre_grad = np.fromfile("pre_grad.bin", dtype=np.float32).reshape(d0, B)
expected_grad = np.fromfile("expected_grad.bin", dtype=np.float32).reshape(d0, B)

# Index 320
idx = 320
r = idx // B  # 320 // 32 = 10
j = idx % B   # 320 % 32 = 0

print(f"Index 320 corresponds to position [{r}, {j}]")
print(f"a_L[{r}, {j}] = {a_L[r, j]}")
print(f"pre_grad[{r}, {j}] = {pre_grad[r, j]}")
print(f"expected_grad[{r}, {j}] = {expected_grad[r, j]}")

# According to reference:
# pre_grad = a_L / B
print(f"\nExpected pre_grad[{r}, {j}] = a_L / B = {a_L[r, j] / B}")
print(f"Pre-grad check: {abs(pre_grad[r, j] - a_L[r, j] / B) < 1e-6}")
