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
import argparse
import numpy as np
import torch
import torch.nn as nn
import os

# Force CPU and float32
torch.set_default_dtype(torch.float32)
device = torch.device("cpu")

class Autoencoder(nn.Module):
    def __init__(self, input_dim, deterministic=False):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128, bias=True),
            nn.ReLU()
        )
        self.decoder = nn.Linear(128, input_dim, bias=True)
        if deterministic:
            self._init_deterministic_weights(input_dim)
    
    def _init_deterministic_weights(self, input_dim):
        """Initialize weights using deterministic formula: ((i*d + j) % 7 - 3) * 0.01."""
        # Layer 0: Linear(input_dim, 128)
        layer0 = self.encoder[0]
        for i in range(128):
            for j in range(input_dim):
                val = ((((i * input_dim + j) % 7) - 3) * 0.01)
                layer0.weight.data[i, j] = val
        layer0.bias.data.zero_()
        
        # Layer: decoder Linear(128, input_dim)
        for i in range(input_dim):
            for j in range(128):
                val = ((((i * 128 + j) % 7) - 3) * 0.01)
                self.decoder.weight.data[i, j] = val
        self.decoder.bias.data.zero_()
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def mse(self, output, target):
        """Compute MSE for loss: ||output - target||^2 / (d0 * B)."""
        return torch.nn.functional.mse_loss(output, target)

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
    
    print(f"[Data] Extracted first {n_cols} columns; dense shape: {X.shape}", flush=True)
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
    
    print(f"[Normalize] Applied log-normalization; output shape: {X_norm.shape}, dtype: {X_norm.dtype}", flush=True)
    return X_norm

def train_autoencoder(model, X_norm, n_epochs=3):
    """
    Train autoencoder for n_epochs full-batch updates.

    Input X_norm: shape (n_genes, n_samples), float32
    PyTorch expects (batch_size, features), so transpose to (n_samples, n_genes).

    Loss formula: MSE = ||output - target||^2 / (d0 * B)
    This matches the C++ loss and gradient formula.

    Embedding is captured from epoch n_epochs's TRAINING forward pass (before the Adam
    step), matching C++'s layer(0)->output() which is also pre-step.

    Returns: list of (epoch_mse, weights_list, grads_list) tuples, final embedding
    """
    X_torch = torch.from_numpy(X_norm.T).to(device)  # (n_samples, n_genes)
    B = X_torch.shape[0]  # batch size = 256
    print(f"[Train] Input to model: {X_torch.shape}, dtype: {X_torch.dtype}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch_results = []
    embedding = None

    for epoch in range(n_epochs):
        # Forward pass
        encoded, output = model(X_torch)

        # Compute MSE loss (matches C++ formula: sum((a-x)^2) / (d0*B))
        mse = model.mse(output, X_torch)

        # Backward pass
        optimizer.zero_grad()
        mse.backward()

        # Capture gradients BEFORE optimizer.step()
        grads = [model.encoder[0].weight.grad.clone(), model.decoder.weight.grad.clone()]

        # Optimizer step
        optimizer.step()

        # Capture weights AFTER optimizer.step()
        weights = [model.encoder[0].weight.data.clone(), model.decoder.weight.data.clone()]

        # Capture embedding from the LAST epoch's training forward (before step),
        # matching C++ layer(0)->output() which is also pre-step.
        if epoch == n_epochs - 1:
            embedding = encoded.detach().cpu().numpy().T  # (128, n_samples)

        epoch_results.append((mse.item(), weights, grads))
        print(f"[Train] Epoch {epoch + 1}/{n_epochs}: MSE = {mse.item():.8f}")

    print(f"[Train] Captured embedding shape: {embedding.shape}", flush=True)
    return epoch_results, embedding

def test_lognorm(X_norm):
    """
    Compare first 10 columns of log-normalized data.
    
    Reads validate_lognorm_10cols.csv from C++ side.
    Compares against first 10 columns of Python X_norm.
    
    Returns:
        bool: True if test passes.
    """
    lognorm_csv_path = "/mnt/home/besleya/autoencoder/tests/validate/validate_lognorm_10cols.csv"
    
    if not os.path.exists(lognorm_csv_path):
        print(f"[Lognorm] WARNING: C++ lognorm CSV not found at {lognorm_csv_path}")
        print("[Lognorm] Skipped.\n")
        return False
    
    print("[Lognorm comparison]")
    
    # Load C++ lognorm result
    try:
        cpp_lognorm = np.genfromtxt(lognorm_csv_path, delimiter=',', dtype=np.float32)
    except Exception as e:
        print(f"[Lognorm] ERROR reading CSV: {e}\n")
        return False
    
    # Extract first 10 columns from Python lognorm
    py_lognorm = X_norm[:, :10]
    
    if cpp_lognorm.shape != py_lognorm.shape:
        print(f"[Lognorm] ERROR: Shape mismatch. PyTorch: {py_lognorm.shape}, C++: {cpp_lognorm.shape}\n")
        return False
    
    # Compare element-wise
    abs_diff = np.abs(py_lognorm - cpp_lognorm)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    test_pass = max_abs_diff < 1e-5
    status = "PASS" if test_pass else "FAIL"
    
    print(f"  max_abs_diff={max_abs_diff:.8e}  mean_abs_diff={mean_abs_diff:.8e}  [{status}]")
    print(f"  Threshold: max_abs_diff < 1e-5\n", flush=True)
    return test_pass

def save_cache(cache_file, X_norm, embedding_epoch0, epoch_results, embedding_final):
    """
    Save expensive computed values to a compressed .npz file.
    
    Args:
        cache_file: path to .npz file
        X_norm: log-normalized data, shape (n_genes, n_samples)
        embedding_epoch0: pre-train embedding, shape (128, n_samples)
        epoch_results: list of (mse, weights, grads) tuples
        embedding_final: final embedding after training, shape (128, n_samples)
    """
    n_epochs = len(epoch_results)
    
    # Extract MSEs
    mses = np.array([mse for mse, _, _ in epoch_results], dtype=np.float32)
    
    # Stack weights and grads
    weights_enc = np.stack([w[0].cpu().numpy() for _, w, _ in epoch_results], axis=0).astype(np.float32)
    weights_dec = np.stack([w[1].cpu().numpy() for _, w, _ in epoch_results], axis=0).astype(np.float32)
    grads_enc = np.stack([g[0].cpu().numpy() for _, _, g in epoch_results], axis=0).astype(np.float32)
    grads_dec = np.stack([g[1].cpu().numpy() for _, _, g in epoch_results], axis=0).astype(np.float32)
    
    np.savez_compressed(
        cache_file,
        lognorm=X_norm.astype(np.float32),
        embedding_epoch0=embedding_epoch0.astype(np.float32),
        mses=mses,
        weights_enc=weights_enc,
        weights_dec=weights_dec,
        grads_enc=grads_enc,
        grads_dec=grads_dec,
        embedding_final=embedding_final.astype(np.float32)
    )
    print(f"[Cache] Saved to {cache_file}", flush=True)

def load_cache(cache_file):
    """
    Load cached values from .npz file.
    
    Returns:
        (X_norm, embedding_epoch0, epoch_results, embedding_final)
        where epoch_results is list of (mse, weights, grads) tuples
        and weights/grads are already as numpy arrays
    """
    if not os.path.exists(cache_file):
        print(f"ERROR: Cache file not found: {cache_file}")
        sys.exit(1)
    
    with np.load(cache_file) as data:
        X_norm = data['lognorm']
        embedding_epoch0 = data['embedding_epoch0']
        mses = data['mses']
        weights_enc = data['weights_enc']
        weights_dec = data['weights_dec']
        grads_enc = data['grads_enc']
        grads_dec = data['grads_dec']
        embedding_final = data['embedding_final']
    
    # Reconstruct epoch_results as list of (mse, weights, grads)
    # weights and grads remain as numpy arrays (not torch tensors)
    epoch_results = []
    for i in range(len(mses)):
        mse = float(mses[i])
        weights = [weights_enc[i], weights_dec[i]]
        grads = [grads_enc[i], grads_dec[i]]
        epoch_results.append((mse, weights, grads))
    
    print(f"[Cache] Loaded from {cache_file}", flush=True)
    return X_norm, embedding_epoch0, epoch_results, embedding_final

def test_forward_pass(py_embedding):
    """
    Compare pre-training forward pass embedding.
    
    Reads validate_embedding_epoch0.csv from C++ side.
    Compares against provided embedding (pre-train).
    
    Args:
        py_embedding: numpy array, shape (128, n_samples)
    
    Returns:
        bool: True if test passes.
    """
    embedding_epoch0_path = "/mnt/home/besleya/autoencoder/tests/validate/validate_embedding_epoch0.csv"
    
    if not os.path.exists(embedding_epoch0_path):
        print(f"[Forward] WARNING: C++ embedding CSV not found at {embedding_epoch0_path}")
        print("[Forward] Skipped.\n")
        return False
    
    print("[Forward pass comparison (epoch 0)]")
    
    # Load C++ embedding
    try:
        cpp_embedding = np.genfromtxt(embedding_epoch0_path, delimiter=',', dtype=np.float32)
    except Exception as e:
        print(f"[Forward] ERROR reading CSV: {e}\n")
        return False
    
    if cpp_embedding.shape != py_embedding.shape:
        print(f"[Forward] ERROR: Shape mismatch. PyTorch: {py_embedding.shape}, C++: {cpp_embedding.shape}\n")
        return False
    
    # Compare element-wise
    abs_diff = np.abs(py_embedding - cpp_embedding)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    test_pass = max_abs_diff < 1e-3
    status = "PASS" if test_pass else "FAIL"
    
    print(f"  max_abs_diff={max_abs_diff:.8e}  mean_abs_diff={mean_abs_diff:.8e}  [{status}]")
    print(f"  Threshold: max_abs_diff < 1e-3\n", flush=True)
    return test_pass

def test_weights(epoch, py_weights):
    """
    Compare weight matrices for a given epoch.
    
    Args:
        epoch: epoch number (1-indexed)
        py_weights: list of 2 weight arrays [layer0, layer2], either torch tensors or numpy arrays
                   shape (out_dim, in_dim) each
    
    Reads C++ weight CSVs and compares.
    Always returns True/False for pass/fail.
    """
    base_path = "/mnt/home/besleya/autoencoder/tests/validate"
    
    print(f"[Weights Epoch {epoch}]")
    all_pass = True
    
    for layer_idx in range(2):
        csv_path = f"{base_path}/validate_weights_layer{layer_idx}_epoch{epoch}.csv"
        
        if not os.path.exists(csv_path):
            print(f"  Layer {layer_idx}: WARNING: CSV not found at {csv_path}")
            continue
        
        try:
            cpp_weights = np.genfromtxt(csv_path, delimiter=',', dtype=np.float32)
            w = py_weights[layer_idx]
            py_w = w.cpu().numpy() if torch.is_tensor(w) else w
            
            if cpp_weights.shape != py_w.shape:
                print(f"  Layer {layer_idx}: ERROR: Shape mismatch. PyTorch: {py_w.shape}, C++: {cpp_weights.shape}")
                all_pass = False
                continue
            
            abs_diff = np.abs(py_w - cpp_weights)
            max_abs_diff = np.max(abs_diff)
            mean_abs_diff = np.mean(abs_diff)
            
            test_pass = max_abs_diff < 1e-3
            status = "PASS" if test_pass else "FAIL"
            
            print(f"  Layer {layer_idx}: max_abs_diff={max_abs_diff:.8e}  mean_abs_diff={mean_abs_diff:.8e}  [{status}]  (threshold 1e-3)")
            
            if not test_pass:
                all_pass = False
        
        except Exception as e:
            print(f"  Layer {layer_idx}: ERROR reading CSV: {e}")
            all_pass = False
    
    print()
    return all_pass

def test_gradients(epoch, py_grads):
    """
    Compare weight gradients for a given epoch.
    
    Args:
        epoch: epoch number (1-indexed)
        py_grads: list of 2 gradient arrays [layer0, layer2], either torch tensors or numpy arrays
                 shape (out_dim, in_dim) each
    
    Reads C++ gradient CSVs and compares.
    Always returns True/False for pass/fail.
    """
    base_path = "/mnt/home/besleya/autoencoder/tests/validate"
    
    print(f"[Gradients Epoch {epoch}]")
    all_pass = True
    
    for layer_idx in range(2):
        csv_path = f"{base_path}/validate_grads_layer{layer_idx}_epoch{epoch}.csv"
        
        if not os.path.exists(csv_path):
            print(f"  Layer {layer_idx}: WARNING: CSV not found at {csv_path}")
            continue
        
        try:
            cpp_grads = np.genfromtxt(csv_path, delimiter=',', dtype=np.float32)
            g = py_grads[layer_idx]
            py_g = g.cpu().numpy() if torch.is_tensor(g) else g
            
            if cpp_grads.shape != py_g.shape:
                print(f"  Layer {layer_idx}: ERROR: Shape mismatch. PyTorch: {py_g.shape}, C++: {cpp_grads.shape}")
                all_pass = False
                continue
            
            abs_diff = np.abs(py_g - cpp_grads)
            max_abs_diff = np.max(abs_diff)
            mean_abs_diff = np.mean(abs_diff)
            
            test_pass = max_abs_diff < 1e-4
            status = "PASS" if test_pass else "FAIL"
            
            print(f"  Layer {layer_idx}: max_abs_diff={max_abs_diff:.8e}  mean_abs_diff={mean_abs_diff:.8e}  [{status}]  (threshold 1e-4)")
            
            if not test_pass:
                all_pass = False
        
        except Exception as e:
            print(f"  Layer {layer_idx}: ERROR reading CSV: {e}")
            all_pass = False
    
    print()
    return all_pass

def test_loss(mses):
    """
    Compare per-epoch MSE against C++ validate_loss.csv.
    
    Args:
       mses: iterable of per-epoch Python MSE floats.
    
    Returns:
       bool: True if all epochs pass.
    """
    loss_csv_path = "/mnt/home/besleya/autoencoder/tests/validate/validate_loss.csv"
    
    if not os.path.exists(loss_csv_path):
       print("[MSE comparison]")
       print(f"  WARNING: C++ loss CSV not found at {loss_csv_path}")
       print("  Skipped.\n")
       return False
    
    print("[MSE comparison]")
    
    # Load C++ MSEs
    try:
       cpp_data = np.genfromtxt(loss_csv_path, delimiter=',', dtype=np.float64, skip_header=1)
       if cpp_data.ndim == 1:
           cpp_mses = [cpp_data[1]] if len(cpp_data) > 1 else []
       else:
           cpp_mses = cpp_data[:, 1]
    except Exception as e:
       print(f"  ERROR reading CSV: {e}\n")
       return False
    
    mses_list = list(mses)
    all_pass = True
    
    for epoch_num, (pytorch_mse, cpp_mse) in enumerate(zip(mses_list, cpp_mses), start=1):
       abs_diff = abs(float(pytorch_mse) - float(cpp_mse))
       rel_diff = abs_diff / (abs(float(cpp_mse)) + 1e-12)
       test_pass = rel_diff < 1e-4
       status = "PASS" if test_pass else "FAIL"
        
       print(f"  Epoch {epoch_num}: PyTorch={float(pytorch_mse):.8f}  C++={float(cpp_mse):.8f}  "
             f"abs_diff={abs_diff:.2e}  rel_diff={rel_diff:.2e}  [{status}]")
        
       if not test_pass:
           all_pass = False
    
    print()
    return all_pass

def test_embedding_final(embedding):
    """
    Compare final-epoch embedding against C++ validate_embedding.csv.
    
    Args:
       embedding: numpy array, shape (128, n_samples)
    
    Returns:
       bool: True if max_abs_diff < 1e-3.
    """
    embedding_csv_path = "/mnt/home/besleya/autoencoder/tests/validate/validate_embedding.csv"
    
    if not os.path.exists(embedding_csv_path):
       print("[Embedding comparison (final)]")
       print(f"  WARNING: C++ embedding CSV not found at {embedding_csv_path}")
       print("  Skipped.\n")
       return False
    
    print("[Embedding comparison (final)]")
    
    # Load C++ embedding
    try:
       cpp_embedding = np.genfromtxt(embedding_csv_path, delimiter=',', dtype=np.float32)
    except Exception as e:
       print(f"  ERROR reading CSV: {e}\n")
       return False
    
    if cpp_embedding.shape != embedding.shape:
       print(f"  ERROR: Shape mismatch. PyTorch: {embedding.shape}, C++: {cpp_embedding.shape}\n")
       return False
    
    # Compare element-wise
    abs_diff = np.abs(embedding - cpp_embedding)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    test_pass = max_abs_diff < 1e-3
    status = "PASS" if test_pass else "FAIL"
    
    print(f"  max_abs_diff={max_abs_diff:.8e}  mean_abs_diff={mean_abs_diff:.8e}  [{status}]")
    print(f"  Threshold: max_abs_diff < 1e-3\n", flush=True)
    return test_pass

def main():
    parser = argparse.ArgumentParser(
        description="Validate PyTorch autoencoder against C++ implementation with optional caching."
    )
    
    cache_group = parser.add_mutually_exclusive_group(required=True)
    cache_group.add_argument(
        '--recompute-cache',
        action='store_true',
        help="Run the Python pipeline fresh, save results to cache file, then run comparisons."
    )
    cache_group.add_argument(
        '--use-cache',
        action='store_true',
        help="Load saved results from cache file, skip the Python pipeline, run comparisons against cached values."
    )
    
    parser.add_argument(
        '--cache-file',
        type=str,
        default='tests/validate/.compare_cache.npz',
        help="Path to the .npz cache file (default: tests/validate/.compare_cache.npz)."
    )
    
    args = parser.parse_args()
    
    # Resolve cache file path
    cache_file = args.cache_file
    if not os.path.isabs(cache_file):
        # Make it relative to the current working directory
        cache_file = os.path.join(os.getcwd(), cache_file)
    
    input_file = "/mnt/home/besleya/quant/GSE260931/GSM8128195/counts.1pz"
    
    if args.recompute_cache:
        if not os.path.exists(input_file):
            print(f"ERROR: Input file not found: {input_file}")
            sys.exit(1)
        
        # Load data
        X = load_input_data(input_file, n_cols=256)
        
        # Normalize
        X_norm = log_normalize_columns(X)
        
        # Build model
        model = Autoencoder(X_norm.shape[0], deterministic=True).to(device).train()
        print(f"[Model] Built deterministic autoencoder: {X_norm.shape[0]} -> 128 -> {X_norm.shape[0]}", flush=True)
        
        # Compute pre-train embedding
        X_torch = torch.from_numpy(X_norm.T).to(device)
        with torch.no_grad():
            encoded, _ = model(X_torch)
            embedding_epoch0 = encoded.cpu().numpy().T  # (128, n_samples)
        
        # Train
        epoch_results, embedding_final = train_autoencoder(model, X_norm, n_epochs=3)
        
        # Save cache
        save_cache(cache_file, X_norm, embedding_epoch0, epoch_results, embedding_final)
        
    else:  # use_cache
        X_norm, embedding_epoch0, epoch_results, embedding_final = load_cache(cache_file)
    
    # Run comparisons (identical in both modes)
    results = []
    
    results.append(("Lognorm", test_lognorm(X_norm)))
    results.append(("Forward pass (epoch 0)", test_forward_pass(embedding_epoch0)))
    
    # Compare per-epoch results
    for epoch_num, (mse, weights, grads) in enumerate(epoch_results, start=1):
       results.append((f"Weights epoch {epoch_num}", test_weights(epoch_num, weights)))
       results.append((f"Gradients epoch {epoch_num}", test_gradients(epoch_num, grads)))
    
    # Compare per-epoch MSE
    mses = [mse for mse, _, _ in epoch_results]
    results.append(("Per-epoch MSE", test_loss(mses)))
    
    # Compare final embedding
    results.append(("Final embedding", test_embedding_final(embedding_final)))
    
    # Print summary
    print("[Test Summary]")
    passed = sum(1 for _, status in results if status)
    failed = sum(1 for _, status in results if not status)
    total = len(results)
    
    for name, status in results:
       status_str = "PASS" if status else "FAIL"
       print(f"  {status_str:4s}  {name}")
    
    print("-" * 48)
    print(f"  {passed} passed, {failed} failed ({total} total)\n", flush=True)
    
    # Always exit 0
    sys.exit(0)

if __name__ == "__main__":
    main()
