#!/usr/bin/env python3
"""
Reference Python implementation of Adam update.

Generates test data and expected outputs for verification.
Test parameters:
  - size: 8192
  - Random init for p, m, v, g (small magnitudes, seed=42)
  - lr_t = 1e-3
  - beta1 = 0.9
  - beta2 = 0.999
  - eps = 1e-8

Outputs:
  - p_init.bin, m_init.bin, v_init.bin, g.bin (inputs)
  - expected_p.bin, expected_m.bin, expected_v.bin (expected after update)
  - meta.txt (test parameters)
"""

import numpy as np

def main():
    # Test parameters
    size = 8192
    lr_t = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    # Generate random data (small magnitudes for numerical stability)
    rng = np.random.RandomState(42)
    p_init = rng.normal(0, 0.1, size).astype(np.float32)
    m_init = rng.normal(0, 0.01, size).astype(np.float32)
    v_init = rng.normal(0, 0.01, size).astype(np.float32)
    g = rng.normal(0, 0.1, size).astype(np.float32)
    
    # Compute expected outputs (manual implementation matching the kernel exactly)
    p = p_init.copy()
    m = m_init.copy()
    v = v_init.copy()
    
    for i in range(size):
        g_val = g[i]
        m_val = beta1 * m[i] + (1.0 - beta1) * g_val
        v_val = beta2 * v[i] + (1.0 - beta2) * g_val * g_val
        
        m[i] = m_val
        v[i] = v_val
        
        p[i] -= lr_t * m_val / (np.sqrt(v_val) + eps)
    
    # Write binary files
    with open('p_init.bin', 'wb') as f:
        p_init.tofile(f)
    
    with open('m_init.bin', 'wb') as f:
        m_init.tofile(f)
    
    with open('v_init.bin', 'wb') as f:
        v_init.tofile(f)
    
    with open('g.bin', 'wb') as f:
        g.tofile(f)
    
    with open('expected_p.bin', 'wb') as f:
        p.tofile(f)
    
    with open('expected_m.bin', 'wb') as f:
        m.tofile(f)
    
    with open('expected_v.bin', 'wb') as f:
        v.tofile(f)
    
    # Write metadata
    with open('meta.txt', 'w') as f:
        f.write(f"size={size}\n")
        f.write(f"lr_t={lr_t}\n")
        f.write(f"beta1={beta1}\n")
        f.write(f"beta2={beta2}\n")
        f.write(f"eps={eps}\n")
    
    print(f"Generated reference data for size={size}")
    print(f"Files: p_init.bin, m_init.bin, v_init.bin, g.bin, expected_p.bin, expected_m.bin, expected_v.bin, meta.txt")

if __name__ == '__main__':
    main()
