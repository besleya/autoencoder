#!/usr/bin/env python3
"""
Generate reference test data for kernel_relu_forward benchmark.

Generates:
  - z.bin: random floats in [-1, 1], size 131072
  - expected_a.bin: ReLU output (max(z, 0))
  - meta.txt: size value
"""

import torch
import struct

SIZE = 512 * 256  # 131072

# Generate random input in [-1, 1]
z = torch.rand(SIZE, dtype=torch.float32) * 2.0 - 1.0

# Compute expected output: ReLU(z) = max(z, 0)
a = torch.clamp(z, min=0.0)

# Save z.bin
with open("z.bin", "wb") as f:
    f.write(z.numpy().tobytes())

# Save expected_a.bin
with open("expected_a.bin", "wb") as f:
    f.write(a.numpy().tobytes())

# Save meta.txt
with open("meta.txt", "w") as f:
    f.write(str(SIZE))

print(f"Generated test data: size={SIZE}")
print(f"  z.bin: {SIZE} random floats in [-1, 1]")
print(f"  expected_a.bin: ReLU(z)")
print(f"  meta.txt: {SIZE}")
