#!/usr/bin/env python3
"""
Generate reference data for kernel_add_bias accuracy test.

Generates:
  z_init.bin      - Initial z matrix (out x B, column-major), float32
  b.bin           - Bias vector (out,), float32
  expected_z.bin  - Expected output (out x B, column-major), float32
  meta.txt        - Metadata: out, B
"""

import numpy as np
import struct

# Parameters
out = 512
B = 256

# Create random data
np.random.seed(42)
z_init = np.random.randn(out, B).astype(np.float32)  # column-major view
b = np.random.randn(out).astype(np.float32)

# Expected: z + b broadcast along columns
# z_expected[i, j] = z_init[i, j] + b[i]
expected_z = z_init + b[:, np.newaxis]

# Save z_init in Fortran order (column-major) so CUDA sees it correctly
with open('z_init.bin', 'wb') as f:
    f.write(z_init.T.tobytes())

# Save b
with open('b.bin', 'wb') as f:
    f.write(b.tobytes())

# Save expected_z in Fortran order (column-major)
with open('expected_z.bin', 'wb') as f:
    f.write(expected_z.T.tobytes())

# Save metadata
with open('meta.txt', 'w') as f:
    f.write(f"out={out}\n")
    f.write(f"B={B}\n")

print(f"Generated reference data: out={out}, B={B}")
print(f"  z_init.bin: {z_init.nbytes} bytes")
print(f"  b.bin: {b.nbytes} bytes")
print(f"  expected_z.bin: {expected_z.nbytes} bytes")
