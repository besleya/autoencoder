#!/usr/bin/env python3
"""
Reference implementation for ReLU backward test data generation.
Uses PyTorch CPU to generate ground truth.

Output files:
  - dz_init.bin: initial gradient (float32, size elements)
  - z.bin: activation values (float32, size elements)
  - expected_dz.bin: expected output gradient (float32, size elements)
  - meta.txt: metadata (size, seed)
"""

import torch
import struct
import os

# Configuration
SIZE = 131072
SEED = 42

def main():
    torch.manual_seed(SEED)
    
    # Generate random test data
    dz_init = torch.FloatTensor(SIZE).uniform_(-1.0, 1.0)
    z = torch.FloatTensor(SIZE).uniform_(-1.0, 1.0)
    
    # Compute expected output: ReLU backward gate
    # dz_out[i] = (z[i] > 0) ? dz[i] : 0
    dz_expected = torch.where(z > 0.0, dz_init, torch.zeros_like(dz_init))
    
    # Write binary files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    dz_init.cpu().numpy().astype('float32').tofile(os.path.join(script_dir, 'dz_init.bin'))
    z.cpu().numpy().astype('float32').tofile(os.path.join(script_dir, 'z.bin'))
    dz_expected.cpu().numpy().astype('float32').tofile(os.path.join(script_dir, 'expected_dz.bin'))
    
    # Write metadata
    with open(os.path.join(script_dir, 'meta.txt'), 'w') as f:
        f.write(f"size={SIZE}\n")
        f.write(f"seed={SEED}\n")
    
    print(f"Generated test data:")
    print(f"  SIZE: {SIZE}")
    print(f"  dz_init range: [{dz_init.min():.4f}, {dz_init.max():.4f}]")
    print(f"  z range: [{z.min():.4f}, {z.max():.4f}]")
    print(f"  expected_dz range: [{dz_expected.min():.4f}, {dz_expected.max():.4f}]")
    print(f"  Files written: dz_init.bin, z.bin, expected_dz.bin, meta.txt")

if __name__ == '__main__':
    main()
