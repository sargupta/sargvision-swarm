"""TWSL — Trust-Weighted Sheaf Laplacian.

SARGVISION-original mathematical object (provisional; see prior-art audit
33_PRIOR_ART_AUDIT.md in AI_Workspace/drone_swarm_research/).

Reference docs:
  34_NOVEL_OBJECTS_DEFINITIONS.md — formal definition (Def. 1)
  35_THEOREMS_AND_PROOFS.md       — Theorems 1-4

In matrix form on a graph with N scalar stalks (the simplest case used by SHIELD):
    L_T = D_T - A_T
    A_T[i,j] = T_i * T_j * A[i,j]
    D_T[i,i] = Sum_j A_T[i,j]

The full sheaf form generalises to vector stalks F(v) ⊂ R^d via the conformal-block
formula in Definition 1; for now we implement the scalar-stalk case, which is what
SHIELD and the live counter-swarm scenario actually use. Vector-stalk extension is
trivial via the block-Kronecker form.
"""

from __future__ import annotations

import numpy as np


def trust_weighted_laplacian(
    adjacency: np.ndarray,
    trust: np.ndarray,
) -> np.ndarray:
    """Compute L_T = D_T − A_T with A_T = (T T^T) ⊙ A.

    Args
    ----
    adjacency : (N, N) symmetric, non-negative, zero diagonal.
    trust     : (N,) in (0, 1].

    Returns
    -------
    L_T : (N, N) symmetric PSD.
    """
    A = np.asarray(adjacency, dtype=np.float64)
    T = np.asarray(trust, dtype=np.float64)
    if A.shape[0] != A.shape[1] or T.shape[0] != A.shape[0]:
        raise ValueError("adjacency must be NxN and trust must be (N,)")
    A_T = (T[:, None] * T[None, :]) * A
    D_T = np.diag(A_T.sum(axis=1))
    return D_T - A_T


def fragmentation_index(
    adjacency: np.ndarray,
    trust: np.ndarray,
) -> float:
    """Φ_T = λ₂(L_T) / λ₂(L_F)  ∈ (0, 1].

    Returns 1.0 when trust is uniform (TWSL reduces to vanilla sheaf Laplacian).
    Theorem 4 guarantees Φ_T ≥ T_min² for CONNECTED graphs.

    Returns NaN if the underlying graph is disconnected (λ₂(L_F) = 0 and the
    ratio is undefined). Use `connected_components(adjacency)` from
    `orchestrator/vajra.py` to apply Theorem 4 per-component on a disconnected graph.
    """
    L_T = trust_weighted_laplacian(adjacency, trust)
    L_F = trust_weighted_laplacian(adjacency, np.ones_like(trust))
    eig_T = np.linalg.eigvalsh(L_T)
    eig_F = np.linalg.eigvalsh(L_F)
    # second-smallest (first is ~0)
    lam2_T = float(eig_T[1])
    lam2_F = float(eig_F[1])
    # Connectivity guard: λ₂(L_F) ≈ 0 means the graph is disconnected.
    if lam2_F < 1e-9:
        return float("nan")
    return lam2_T / lam2_F


def twsl_self_consistent_iteration(
    adjacency: np.ndarray,
    test_cochain: np.ndarray,
    damping: float = 0.85,
    iters: int = 30,
    sigma: float = 1.0,
    T0: float = 1e-3,
) -> np.ndarray:
    """The TRUE nonlinear TWSL self-consistent iteration (Theorem 1).

    Updates BOTH the loyalty and the trust vector each step:
      ℓ_k     = exp(-r(T_k)² / 2σ²)  where r(T) = (L_T x)_v per vertex
      T_{k+1} = (1-d) ℓ_k + d ℓ_k ⊙ (M T_k)

    The iteration is clamped to [T0, 1]^N to avoid the 1/√T_min Lipschitz blow-up
    flagged by PhD review (35_THEOREMS_AND_PROOFS.md §1 caveat).

    Returns the converged trust vector T*.
    """
    A = np.asarray(adjacency, dtype=np.float64)
    x = np.asarray(test_cochain, dtype=np.float64)
    n = A.shape[0]
    out_deg = A.sum(axis=1)
    out_deg = np.where(out_deg > 0, out_deg, 1.0)
    M = (A / out_deg[:, None]).T  # column-stochastic
    T = np.ones(n, dtype=np.float64)
    for _ in range(iters):
        L_T = trust_weighted_laplacian(A, T)
        # per-vertex residual r_i = |(L_T x)_i|
        residuals = np.abs(L_T @ x)
        ell = np.exp(-(residuals**2) / (2 * sigma**2))
        T = (1 - damping) * ell + damping * ell * (M @ T)
        T = np.clip(T, T0, 1.0)
    return T


def twsl_iteration(
    adjacency: np.ndarray,
    residuals: np.ndarray,
    damping: float = 0.85,
    iters: int = 20,
    sigma: float = 1.0,
) -> np.ndarray:
    """The TWSL self-consistent iteration of Definition 1.

    T_{k+1} = (1-d) * ell(T_k) + d * ell(T_k) ⊙ (M T_k)
    where ell_i(T) = exp(-r_i(T)² / 2σ²) and M is the column-stochastic
    propagation matrix induced by A.

    For Theorem 1 verification: we pass in pre-computed residuals (one per
    vertex) treated as constants — i.e., we hold ell fixed and check the
    linear-part contraction. The full nonlinear case requires sigma chosen
    so the Lipschitz constant L_r / sigma² < 1 - d.
    """
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    out_deg = A.sum(axis=1)
    out_deg = np.where(out_deg > 0, out_deg, 1.0)
    M = (A / out_deg[:, None]).T  # column-stochastic
    ell = np.exp(-(residuals**2) / (2 * sigma**2))
    T = np.ones(n, dtype=np.float64)
    for _ in range(iters):
        T = (1 - damping) * ell + damping * ell * (M @ T)
    return T


def is_psd(matrix: np.ndarray, tol: float = 1e-9) -> bool:
    """Numerical PSD check via min eigenvalue."""
    return bool(np.linalg.eigvalsh((matrix + matrix.T) / 2).min() >= -tol)
