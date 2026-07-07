"""Log-normalizes a 1pz matrix


"""

import scanpy as sc
import singlet

def lognorm(matrix, target_sum=1e4):
    """Log-normalizes a 1pz matrix

    Args:
        matrix (np.ndarray): 1pz matrix to log-normalize
        target_sum (float): target sum for normalization

    Returns:
        np.ndarray: log-normalized matrix
    """

    # Add a small constant to avoid log(0)
    sc.pp.normalize_total(matrix, target_sum=target_sum, inplace=True)
    sc.pp.log1p(matrix)

    return matrix
