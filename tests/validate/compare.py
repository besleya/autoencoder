#!/usr/bin/env python3
"""
Validation script comparing C++ autoencoder against PyTorch reference.

Data orientation:
  - Input X: shape (n_genes, 256), loaded and normalized as dense float32
  - For nn.Linear, PyTorch expects (batch_size, features), so we transpose to (256, n_genes)
  - Bottleneck embedding: (256, 128) from PyTorch, transposed to (128, 256) for comparison with C++
  - C++ orientation assumed: (128, 256) = (bottleneck_dim, n_samples)
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import os

# Force CPU and float32
torch.set_default_dtype(torch.float32)
device = torch.device("cpu")

def load_input_data(filepath, n_cols=256):
    """
    Load .1pz file using singlet and extract first n_cols columns.
    Returns dense float32 matrix of shape (n_genes, n_cols).
    """
    try:
        import singlet
    except ImportError:
        print("ERROR: singlet library not found. Cannot load .1pz file.")
        sys.exit(1)
    
    try:
        data = singlet.read_1pz(filepath)
    except Exception as e:
        print(f"ERROR: Failed to read .1pz file: {e}")
        sys.exit(1)
    
    # Inspect sparse matrix structure
    # singlet.read_1pz() returns an AnnData object with (n_obs, n_vars) = (samples, genes)
    n_genes = data.n_vars
    n_samples = data.n_obs
    print(f"[Data] Loaded sparse matrix: {n_genes} genes × {n_samples} samples (CSC format)")
    
    if n_samples < n_cols:
        print(f"WARNING: Only {n_samples} samples available, using all of them instead of {n_cols}")
        n_cols = n_samples
    
    # AnnData.X is stored as (n_obs, n_vars) = (samples, genes)
    # We need to transpose and extract columns
    # Convert to CSC if not already for efficient column access
    X_csc = data.X.T.tocsc()  # Transpose to (genes, samples) and convert to CSC
    
    # Extract first n_cols columns from CSC sparse matrix
    # indptr[j:j+2] gives the range in indices/data for column j
    col_slice = []
    for j in range(n_cols):
        start = X_csc.indptr[j]
        end = X_csc.indptr[j + 1]
        row_indices = X_csc.indices[start:end]
        col_values = X_csc.data[start:end]
        col_slice.append((row_indices, col_values))
    
    # Convert to dense (n_genes, n_cols)
    X = np.zeros((n_genes, n_cols), dtype=np.float32)
    for j, (row_idx, values) in enumerate(col_slice):
        X[row_idx, j] = values.astype(np.float32)
    
    print(f"[Data] Extracted first {n_cols} columns; dense shape: {X.shape}")
    return X

def log_normalize_columns(X):
    """
    Apply log-normalization matching gpu_data_loader.cu::log_normalize_columns_kernel.
    
    Per column j:
      col_sum_j = sum of non-zero values in column j
      For each non-zero x in column j: x_new = log1p(x * 10000.0 / col_sum_j)
      Zeros stay zero.
      If col_sum_j == 0, column is unchanged.
    
    Input: X shape (n_genes, n_samples), float32
    Output: normalized X, float32
    """
    X_norm = X.copy()
    n_genes, n_samples = X.shape
    
    for j in range(n_samples):
        col = X[:, j]
        col_sum = np.sum(col[col > 0])  # Sum of non-zero values
        
        if col_sum > 0:
            # Apply log1p normalization to non-zero elements
            X_norm[:, j] = np.log1p(X[:, j] * (10000.0 / col_sum))
        # Else: column unchanged (all zeros or sum is 0)
    
    print(f"[Normalize] Applied log-normalization; output shape: {X_norm.shape}, dtype: {X_norm.dtype}")
    return X_norm

def build_deterministic_autoencoder(input_dim):
    """
    Build nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, input_dim)).
    Override weights using deterministic formula:
      W[i,j] = ((((i * in_dim + j) % 7) - 3) * 0.01)
    All biases set to zero.
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 128, bias=True),
        nn.ReLU(),
        nn.Linear(128, input_dim, bias=True)
    )
    
    # Layer 0: Linear(input_dim, 128)
    layer0 = model[0]
    in_dim0, out_dim0 = input_dim, 128
    for i in range(out_dim0):
        for j in range(in_dim0):
            val = ((((i * in_dim0 + j) % 7) - 3) * 0.01)
            layer0.weight.data[i, j] = val
    layer0.bias.data.zero_()
    
    # Layer 2: Linear(128, input_dim)
    layer2 = model[2]
    in_dim2, out_dim2 = 128, input_dim
    for i in range(out_dim2):
        for j in range(in_dim2):
            val = ((((i * in_dim2 + j) % 7) - 3) * 0.01)
            layer2.weight.data[i, j] = val
    layer2.bias.data.zero_()
    
    print(f"[Model] Built deterministic autoencoder: {input_dim} -> 128 -> {input_dim}")
    return model.to(device).train()

def train_autoencoder(model, X_norm, n_epochs=3):
    """
    Train autoencoder for n_epochs full-batch updates.
    
    Input X_norm: shape (n_genes, n_samples), float32
    PyTorch expects (batch_size, features), so transpose to (n_samples, n_genes).
    
    Returns: list of epoch MSEs, embedding after final epoch (shape 128 x 256)
    """
    X_torch = torch.from_numpy(X_norm.T).to(device)  # (n_samples, n_genes)
    print(f"[Train] Input to model: {X_torch.shape}, dtype: {X_torch.dtype}")
    
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epoch_mses = []
    
    for epoch in range(n_epochs):
        # Forward pass
        output = model(X_torch)
        loss = criterion(output, X_torch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record the loss from the training forward pass (NOT a re-computed loss after step)
        epoch_mses.append(loss.item())
        print(f"[Train] Epoch {epoch + 1}/{n_epochs}: MSE = {loss.item():.8f}")
    
    # Capture ReLU embedding after training
    with torch.no_grad():
        # Forward through encoder layers
        hidden = model[0](X_torch)  # (n_samples, 128)
        hidden = model[1](hidden)   # ReLU, still (n_samples, 128)
        embedding = hidden.cpu().numpy().T  # Transpose to (128, n_samples)
    
    print(f"[Train] Captured embedding shape: {embedding.shape}")
    return epoch_mses, embedding

def test_lognorm(X_norm):
    """
    Compare first 10 columns of log-normalized data.
    
    Reads validate_lognorm_10cols.csv from C++ side.
    Compares against first 10 columns of Python X_norm.
    
    Always exit 0.
    """
    lognorm_csv_path = "/mnt/home/besleya/autoencoder/tests/validate/validate_lognorm_10cols.csv"
    
    if not os.path.exists(lognorm_csv_path):
        print(f"[Lognorm] WARNING: C++ lognorm CSV not found at {lognorm_csv_path}")
        print("[Lognorm] Skipped.\n")
        return
    
    print("[Lognorm comparison]")
    
    # Load C++ lognorm result
    try:
        cpp_lognorm = np.genfromtxt(lognorm_csv_path, delimiter=',', dtype=np.float32)
    except Exception as e:
        print(f"[Lognorm] ERROR reading CSV: {e}\n")
        return
    
    # Extract first 10 columns from Python lognorm
    py_lognorm = X_norm[:, :10]
    
    if cpp_lognorm.shape != py_lognorm.shape:
        print(f"[Lognorm] ERROR: Shape mismatch. PyTorch: {py_lognorm.shape}, C++: {cpp_lognorm.shape}\n")
        return
    
    # Compare element-wise
    abs_diff = np.abs(py_lognorm - cpp_lognorm)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    test_pass = max_abs_diff < 1e-5
    status = "PASS" if test_pass else "FAIL"
    
    print(f"  max_abs_diff={max_abs_diff:.8e}  mean_abs_diff={mean_abs_diff:.8e}  [{status}]")
    print(f"  Threshold: max_abs_diff < 1e-5\n")

def test_forward_pass(model, X_norm):
    """
    Compare pre-training forward pass embedding.
    
    Reads validate_embedding_epoch0.csv from C++ side.
    Compares against Python forward pass (before training).
    
    Always exit 0.
    """
    embedding_epoch0_path = "/mnt/home/besleya/autoencoder/tests/validate/validate_embedding_epoch0.csv"
    
    if not os.path.exists(embedding_epoch0_path):
        print(f"[Forward] WARNING: C++ embedding CSV not found at {embedding_epoch0_path}")
        print("[Forward] Skipped.\n")
        return
    
    print("[Forward pass comparison (epoch 0)]")
    
    # Python: forward pass on initialized (not yet trained) model
    X_torch = torch.from_numpy(X_norm.T).to(device)  # (n_samples, n_genes)
    with torch.no_grad():
        hidden = model[0](X_torch)  # (n_samples, 128)
        hidden = model[1](hidden)   # ReLU, still (n_samples, 128)
        py_embedding = hidden.cpu().numpy().T  # Transpose to (128, n_samples)
    
    # Load C++ embedding
    try:
        cpp_embedding = np.genfromtxt(embedding_epoch0_path, delimiter=',', dtype=np.float32)
    except Exception as e:
        print(f"[Forward] ERROR reading CSV: {e}\n")
        return
    
    if cpp_embedding.shape != py_embedding.shape:
        print(f"[Forward] ERROR: Shape mismatch. PyTorch: {py_embedding.shape}, C++: {cpp_embedding.shape}\n")
        return
    
    # Compare element-wise
    abs_diff = np.abs(py_embedding - cpp_embedding)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    test_pass = max_abs_diff < 1e-3
    status = "PASS" if test_pass else "FAIL"
    
    print(f"  max_abs_diff={max_abs_diff:.8e}  mean_abs_diff={mean_abs_diff:.8e}  [{status}]")
    print(f"  Threshold: max_abs_diff < 1e-3\n")

def compare_results(epoch_mses, embedding, X_norm, model):
    """
    Load C++ outputs and compare.
    
    C++ outputs:
      - validate_loss.csv: epoch,mse (3 rows, 1 data row per epoch)
      - validate_embedding.csv: 128 rows x 256 cols, no header
    
    Always exit 0.
    """
    loss_csv_path = "/mnt/home/besleya/autoencoder/tests/validate/validate_loss.csv"
    embedding_csv_path = "/mnt/home/besleya/autoencoder/tests/validate/validate_embedding.csv"
    
    print("\n[PyTorch Results]")
    for i, mse in enumerate(epoch_mses):
        print(f"  Epoch {i + 1}: MSE = {mse:.8f}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding[0, :5] = {embedding[0, :5]}")
    
    # Check if C++ CSVs exist
    if not os.path.exists(loss_csv_path):
        print(f"\nWARNING: C++ loss CSV not found at {loss_csv_path}")
        print("Comparison skipped. PyTorch reference output above.\n")
        return
    
    if not os.path.exists(embedding_csv_path):
        print(f"\nWARNING: C++ embedding CSV not found at {embedding_csv_path}")
        print("Comparison skipped. PyTorch reference output above.\n")
        return
    
    print("\n[C++ vs PyTorch Comparison]")
    
    # Load C++ loss
    cpp_mses = []
    try:
        with open(loss_csv_path, 'r') as f:
            lines = f.readlines()
            # Skip header
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    cpp_mses.append(float(parts[1]))
    except Exception as e:
        print(f"ERROR reading loss CSV: {e}")
        return
    
    # Compare per-epoch MSE
    print("\n  Per-epoch MSE:")
    mse_pass = True
    for i in range(len(epoch_mses)):
        if i >= len(cpp_mses):
            print(f"    Epoch {i + 1}: PyTorch MSE = {epoch_mses[i]:.8f}, C++ missing")
            mse_pass = False
            continue
        
        pytorch_mse = epoch_mses[i]
        cpp_mse = cpp_mses[i]
        abs_diff = abs(pytorch_mse - cpp_mse)
        rel_diff = abs_diff / (abs(cpp_mse) + 1e-12)
        
        status = "PASS" if rel_diff < 1e-4 else "FAIL"
        print(f"    Epoch {i + 1}: PyTorch={pytorch_mse:.8f}  C++={cpp_mse:.8f}  "
              f"abs_diff={abs_diff:.8f}  rel_diff={rel_diff:.8e}  [{status}]")
        
        if rel_diff >= 1e-4:
            mse_pass = False
    
    # Load C++ embedding
    cpp_embedding = None
    try:
        cpp_embedding = np.genfromtxt(embedding_csv_path, delimiter=',', dtype=np.float32)
    except Exception as e:
        print(f"\nERROR reading embedding CSV: {e}")
        return
    
    if cpp_embedding.shape != embedding.shape:
        print(f"\nERROR: Shape mismatch. PyTorch: {embedding.shape}, C++: {cpp_embedding.shape}")
        return
    
    # Compare embedding
    print(f"\n  Embedding comparison (shape {embedding.shape}):")
    abs_diff = np.abs(embedding - cpp_embedding)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    # Relative difference, avoiding division by zero
    rel_diff = abs_diff / (np.abs(cpp_embedding) + 1e-12)
    max_rel_diff = np.max(rel_diff)
    
    embed_pass = max_abs_diff < 1e-3
    status = "PASS" if embed_pass else "FAIL"
    
    print(f"    max_abs_diff={max_abs_diff:.8e}  mean_abs_diff={mean_abs_diff:.8e}  "
          f"max_rel_diff={max_rel_diff:.8e}  [{status}]")
    
    # Summary
    print("\n[Summary]")
    overall = mse_pass and embed_pass
    print(f"  MSE comparison: {'PASS' if mse_pass else 'FAIL'} (threshold: rel_diff < 1e-4)")
    print(f"  Embedding comparison: {'PASS' if embed_pass else 'FAIL'} (threshold: max_abs_diff < 1e-3)")
    print(f"  Overall: {'PASS' if overall else 'FAIL'}\n")

def main():
    input_file = "/mnt/home/besleya/quant/GSE260931/GSM8128195/counts.1pz"
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)
    
    # Load data
    X = load_input_data(input_file, n_cols=256)
    
    # Normalize
    X_norm = log_normalize_columns(X)
    
    # Build model
    model = build_deterministic_autoencoder(X_norm.shape[0])
    
    # Test 1: Log-normalization (before training)
    test_lognorm(X_norm)
    
    # Test 2: Forward pass embedding (before training)
    test_forward_pass(model, X_norm)
    
    # Train
    epoch_mses, embedding = train_autoencoder(model, X_norm, n_epochs=3)
    
    # Compare with C++ outputs
    compare_results(epoch_mses, embedding, X_norm, model)
    
    # Always exit 0
    sys.exit(0)

if __name__ == "__main__":
    main()
