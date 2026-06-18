#!/usr/bin/env python3
"""
Generate reference data for kernel_accumulate_loss testing.

Kernel computes: d_epoch_sum += (d_dot + d_loss) / (d0 * B)

Parameters:
  d0 = 64
  B = 32
  denom = d0 * B = 2048

Outputs:
  d_dot.bin: random float32
  d_loss.bin: random float32
  epoch_sum_init.bin: 0.5 (float32)
  expected_epoch_sum.bin: 0.5 + (d_dot + d_loss) / 2048 (float32)
  meta.txt: parameter documentation
"""

import struct
import numpy as np

# Parameters
d0 = 64
B = 32
denom = d0 * B

# Set seed for reproducibility
np.random.seed(42)

# Generate random values
d_dot = np.float32(np.random.randn())
d_loss = np.float32(np.random.randn())
epoch_sum_init = np.float32(0.5)

# Compute expected result
expected_epoch_sum = epoch_sum_init + (d_dot + d_loss) / denom

# Write binary files
with open('d_dot.bin', 'wb') as f:
    f.write(struct.pack('f', d_dot))

with open('d_loss.bin', 'wb') as f:
    f.write(struct.pack('f', d_loss))

with open('epoch_sum_init.bin', 'wb') as f:
    f.write(struct.pack('f', epoch_sum_init))

with open('expected_epoch_sum.bin', 'wb') as f:
    f.write(struct.pack('f', expected_epoch_sum))

# Write metadata
with open('meta.txt', 'w') as f:
    f.write(f"d0 = {d0}\n")
    f.write(f"B = {B}\n")
    f.write(f"denom (d0*B) = {denom}\n")
    f.write(f"d_dot = {d_dot}\n")
    f.write(f"d_loss = {d_loss}\n")
    f.write(f"epoch_sum_init = {epoch_sum_init}\n")
    f.write(f"(d_dot + d_loss) / denom = {(d_dot + d_loss) / denom}\n")
    f.write(f"expected_epoch_sum = {expected_epoch_sum}\n")

print(f"Generated reference data:")
print(f"  d0={d0}, B={B}, denom={denom}")
print(f"  d_dot={d_dot:.6f}")
print(f"  d_loss={d_loss:.6f}")
print(f"  epoch_sum_init={epoch_sum_init:.6f}")
print(f"  expected_epoch_sum={expected_epoch_sum:.6f}")
