"""Metrics for measuring feature convergence and rank collapse.

Key metrics:
- Pairwise cosine similarity (cone effect indicator)
- Effective rank via singular values
- Residual norm (token uniformity indicator, per Dong et al. 2021)
"""

import numpy as np
from typing import Dict


def pairwise_cosine_similarity(X: np.ndarray) -> np.ndarray:
    """Compute all pairwise cosine similarities between rows of X.

    Args:
        X: (N, d) feature matrix.

    Returns:
        (N*(N-1)/2,) array of pairwise cosine similarities.
    """
    # L2 normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_norm = X / norms

    # Full cosine similarity matrix
    sim_matrix = X_norm @ X_norm.T

    # Extract upper triangle (exclude diagonal)
    idx = np.triu_indices(X.shape[0], k=1)
    return sim_matrix[idx]


def cosine_similarity_stats(X: np.ndarray) -> Dict[str, float]:
    """Compute statistics of pairwise cosine similarity.

    Returns dict with: mean, std, min, max, median.
    """
    sims = pairwise_cosine_similarity(X)
    return {
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
        "median": float(np.median(sims)),
    }


def effective_rank(X: np.ndarray) -> float:
    """Compute the effective rank of X via Shannon entropy of singular values.

    effective_rank = exp( -sum_i p_i log p_i )
    where p_i = sigma_i / sum(sigma)

    A higher value means the representation is more spread out (less collapsed).
    """
    sv = np.linalg.svd(X, compute_uv=False)
    sv = sv[sv > 1e-12]
    p = sv / sv.sum()
    entropy = -np.sum(p * np.log(p + 1e-30))
    return float(np.exp(entropy))


def residual_norm(X: np.ndarray, norm_type: str = "1_inf") -> float:
    """Compute residual norm measuring deviation from rank-1.

    res(X) = X - 1 * x^T,  where x = mean row of X.
    This is the metric from Dong et al. (2021), Theorem 2.2.

    Args:
        X: (N, d) feature matrix.
        norm_type: '1_inf' for the composite norm, 'fro' for Frobenius.
    """
    x_mean = X.mean(axis=0, keepdims=True)  # (1, d)
    R = X - x_mean  # Residual

    if norm_type == "1_inf":
        # ||R||_{1,inf} = sqrt(||R||_1 * ||R||_inf)
        norm_1 = np.max(np.sum(np.abs(R), axis=0))   # max column abs sum
        norm_inf = np.max(np.sum(np.abs(R), axis=1))  # max row abs sum
        return float(np.sqrt(norm_1 * norm_inf))
    elif norm_type == "fro":
        return float(np.linalg.norm(R, "fro"))
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


def relative_residual(X: np.ndarray) -> float:
    """Relative residual: ||res(X)||_{1,inf} / ||X||_{1,inf}.

    This is the y-axis metric used in Fig.2 of Dong et al. (2021).
    Values close to 0 indicate rank collapse (token uniformity).
    """
    x_mean = X.mean(axis=0, keepdims=True)
    R = X - x_mean

    def composite_norm(M):
        n1 = np.max(np.sum(np.abs(M), axis=0))
        ninf = np.max(np.sum(np.abs(M), axis=1))
        return np.sqrt(n1 * ninf)

    r_norm = composite_norm(R)
    x_norm = composite_norm(X)
    if x_norm < 1e-12:
        return 0.0
    return float(r_norm / x_norm)
