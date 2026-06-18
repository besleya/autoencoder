#!/usr/bin/env python3
"""
Reference implementation: fill ones on CPU with PyTorch.
Generates expected_v.bin (float32) and meta.txt
"""

import torch
import struct
import os

def main():
    # Parameters
    n = 4096
    
    # Create tensor of ones on CPU
    v = torch.ones(n, dtype=torch.float32)
    
    # Save to binary file (float32)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    bin_path = os.path.join(output_dir, "expected_v.bin")
    meta_path = os.path.join(output_dir, "meta.txt")
    
    # Write binary (raw numpy data)
    with open(bin_path, 'wb') as f:
        f.write(v.numpy().tobytes())
    
    # Write metadata
    with open(meta_path, 'w') as f:
        f.write(f"n={n}\n")
        f.write(f"dtype=float32\n")
    
    print(f"Generated {bin_path} ({n} floats)")
    print(f"Generated {meta_path}")
    print(f"Expected: all values = 1.0f")

if __name__ == "__main__":
    main()
