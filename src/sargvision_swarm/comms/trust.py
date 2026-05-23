"""PageRank-style damped trust propagation over comm graph, weighted by loyalty."""
from __future__ import annotations
import numpy as np


def pagerank_trust(
    adjacency: np.ndarray,
    loyalty: np.ndarray,
    damping: float = 0.85,
    iters: int = 12,
) -> np.ndarray:
    """Damped iteration on the loyalty-weighted graph.

    T_i^{k+1} = (1 - d) + d * sum_{j in N_i} ell_j * T_j^k / |N_j|
    """
    n = adjacency.shape[0]
    T = np.ones(n, dtype=np.float64)
    # outbound normaliser per node
    out_deg = adjacency.sum(axis=1).astype(np.float64)
    out_deg = np.where(out_deg > 0, out_deg, 1.0)
    for _ in range(iters):
        contrib = (loyalty * T) / out_deg
        T = (1.0 - damping) + damping * (adjacency.T @ contrib)
    # normalise to (0, 1] so we can use as a multiplicative weight
    T = T / max(T.max(), 1e-9)
    return T
