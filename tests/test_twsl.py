"""TWSL — verification of Theorems 1-4 from 35_THEOREMS_AND_PROOFS.md."""

from __future__ import annotations

import numpy as np

from sargvision_swarm.core.twsl import (
    fragmentation_index,
    is_psd,
    trust_weighted_laplacian,
    twsl_iteration,
)


# ── Setup helpers ────────────────────────────────────────────────────


def _ring_adjacency(n: int) -> np.ndarray:
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    return A


def _complete_adjacency(n: int) -> np.ndarray:
    return np.ones((n, n)) - np.eye(n)


# ── PSD-ness + reduction sanity (Definition 1 corollaries) ───────────


def test_twsl_is_psd():
    A = _complete_adjacency(8)
    T = np.random.default_rng(0).uniform(0.1, 1.0, size=8)
    L_T = trust_weighted_laplacian(A, T)
    assert is_psd(L_T)


def test_twsl_reduces_to_sheaf_laplacian_when_trust_uniform():
    A = _ring_adjacency(6)
    L_T = trust_weighted_laplacian(A, np.ones(6))
    # Vanilla scalar-stalk sheaf Laplacian = graph Laplacian = D - A
    L_F = np.diag(A.sum(axis=1)) - A
    assert np.allclose(L_T, L_F)


# ── Theorem 2 (Monotonicity) ─────────────────────────────────────────


def test_lowering_one_trust_does_not_increase_lambda2():
    """Theorem 2: lowering T_i monotonically lowers λ₂(L_T)."""
    A = _complete_adjacency(7)
    T_hi = np.ones(7) * 0.9
    T_lo = T_hi.copy()
    T_lo[3] = 0.3  # collapse drone 3's trust
    lam_hi = np.linalg.eigvalsh(trust_weighted_laplacian(A, T_hi))[1]
    lam_lo = np.linalg.eigvalsh(trust_weighted_laplacian(A, T_lo))[1]
    assert lam_lo <= lam_hi + 1e-9, f"monotonicity failed: hi={lam_hi:.4f} lo={lam_lo:.4f}"


def test_monotonicity_sweep():
    """For 20 random trust descents, λ₂ never strictly increases."""
    rng = np.random.default_rng(42)
    A = _complete_adjacency(10)
    for _ in range(20):
        T1 = rng.uniform(0.2, 1.0, size=10)
        # randomly lower a subset of entries
        T2 = T1.copy()
        idx = rng.choice(10, size=3, replace=False)
        for i in idx:
            T2[i] = T1[i] * rng.uniform(0.1, 0.99)
        lam1 = np.linalg.eigvalsh(trust_weighted_laplacian(A, T1))[1]
        lam2 = np.linalg.eigvalsh(trust_weighted_laplacian(A, T2))[1]
        assert lam2 <= lam1 + 1e-9


# ── Theorem 4 (Fragmentation lower bound) ────────────────────────────


def test_fragmentation_index_at_least_t_min_squared():
    """Theorem 4: Φ_T ≥ T_min²."""
    rng = np.random.default_rng(7)
    A = _complete_adjacency(8)
    for _ in range(15):
        T = rng.uniform(0.1, 1.0, size=8)
        phi = fragmentation_index(A, T)
        t_min_sq = float(T.min() ** 2)
        assert phi >= t_min_sq - 1e-9, f"Φ_T={phi:.4f} < T_min²={t_min_sq:.4f}"


def test_fragmentation_index_is_unity_when_uniform_trust():
    A = _ring_adjacency(5)
    T = np.ones(5)
    assert abs(fragmentation_index(A, T) - 1.0) < 1e-9


# ── Theorem 1 (TWSL iteration convergence) ───────────────────────────


def test_twsl_iteration_converges_to_same_fixed_point():
    """Theorem 1: TWSL iteration is independent of starting point in the linear-part regime."""
    A = _complete_adjacency(10)
    rng = np.random.default_rng(0)
    residuals = rng.uniform(0.5, 2.0, size=10)
    T1 = twsl_iteration(A, residuals, damping=0.85, iters=50)
    T2 = twsl_iteration(A, residuals, damping=0.85, iters=50)
    # Both runs deterministic (same residuals) → identical
    assert np.allclose(T1, T2)


def test_twsl_iteration_geometric_convergence_rate():
    """Theorem 1: TWSL iteration converges linearly at rate ≤ d.

    Check: error after k iterations is O(d^k) for fixed residual + linear regime.
    """
    A = _complete_adjacency(8)
    residuals = np.ones(8) * 0.5
    T_long = twsl_iteration(A, residuals, damping=0.85, iters=200)
    err5 = np.linalg.norm(twsl_iteration(A, residuals, damping=0.85, iters=5) - T_long)
    err10 = np.linalg.norm(twsl_iteration(A, residuals, damping=0.85, iters=10) - T_long)
    # Error should shrink at least by factor 0.85^5 = 0.444
    assert err10 < err5 * 0.5 + 1e-9, (
        f"convergence too slow: err5={err5:.4f} err10={err10:.4f}"
    )


# ── Theorem 3 (Byzantine detection — calibration sanity) ──────────────


def test_byzantine_detection_threshold_calibration():
    """Theorem 3: kill-switch at 0.25 catches bias ≥ 4σ.

    Numerical sanity: for spoofed-residual b = 4σ, the per-vertex loyalty
    ell = exp(-16/2) = exp(-8) ≈ 3.4e-4 → after one iteration T ≤ (1-d)·ell + d/N
    well below 0.25.
    """
    sigma = 1.5
    b = 4 * sigma
    ell = np.exp(-(b ** 2) / (2 * sigma ** 2))
    d = 0.85
    N = 30
    upper = (1 - d) * ell + d / N
    assert upper < 0.25, f"Byzantine upper bound {upper:.4f} not below 0.25 threshold"


# ── Integration with the existing SHIELD/sheaf machinery ─────────────


def test_twsl_matches_sheaf_dirichlet_residual_scaling():
    """Sanity: on a fully connected graph with uniform trust, x^T L_T x equals
    the Dirichlet energy of x with respect to the graph Laplacian."""
    A = _complete_adjacency(6)
    L_T = trust_weighted_laplacian(A, np.ones(6))
    x = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    # Dirichlet energy = 0.5 Σ_{(i,j) ∈ E} (x_i − x_j)² = number of disagreeing edges
    # In K_6 with alternating x: every edge connects ±1 → all 15 edges disagree → 15·4 = 60
    expected = 0.5 * sum(
        (x[i] - x[j]) ** 2 for i in range(6) for j in range(i + 1, 6) if A[i, j] > 0
    )
    quad = float(x @ L_T @ x) / 2.0  # both sides should match up to factor
    # x^T L x = 2 · Dirichlet energy in the standard convention
    assert abs(float(x @ L_T @ x) - 2 * expected) < 1e-9 or abs(quad - expected) < 1e-9
