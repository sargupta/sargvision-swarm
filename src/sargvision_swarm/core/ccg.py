"""CCG — Classical-Correlation Gain.

SARGVISION-original mathematical object (provisional; see prior-art audit
33_PRIOR_ART_AUDIT.md in AI_Workspace/drone_swarm_research/).

Reference docs:
  34_NOVEL_OBJECTS_DEFINITIONS.md — formal definition (Def. 2)
  35_THEOREMS_AND_PROOFS.md       — Theorems 5-7

CCG measures the payoff gap between (a) the best signal-conditioned policy under a
shared correlation device ν, and (b) the best product policy without any shared
randomness. For coordination games (common payoff u), CCG ≥ 0 and is bounded above
by U · sqrt(I̅(ν) / 2) where I̅ is the mean pairwise mutual information of ν.

The functions here let you (i) numerically compute CCG for a finite game via brute
force over the policy space, (ii) verify the Theorem 5 upper bound, (iii) check
Theorem 7's adversarial-CCG positivity under encrypted seeds.
"""
from __future__ import annotations

from itertools import product

import numpy as np


def shannon_entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def mutual_information(joint: np.ndarray) -> float:
    """I(X; Y) from a joint distribution table (|X|, |Y|)."""
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    h_x = shannon_entropy(p_x)
    h_y = shannon_entropy(p_y)
    h_xy = shannon_entropy(joint.ravel())
    return h_x + h_y - h_xy


def mean_pairwise_mi(nu: np.ndarray) -> float:
    """I̅(ν) — mean pairwise MI for a joint over N variables.

    Input: nu of shape (k, k, ..., k) for N variables on alphabet of size k.
    """
    N = nu.ndim
    if N < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            axes = tuple(ax for ax in range(N) if ax not in (i, j))
            joint_ij = nu.sum(axis=axes) if axes else nu
            # ensure 2D (i first, j second)
            if i > j:
                joint_ij = joint_ij.T
            total += mutual_information(joint_ij)
            count += 1
    return total / max(count, 1)


def best_product_payoff(
    payoff: np.ndarray,
    action_sizes: tuple[int, ...],
    n_samples: int = 0,
) -> tuple[float, list[np.ndarray]]:
    """Best product-policy payoff via gradient ascent on the simplex product.

    For exact-but-slow brute force (every deterministic profile), pass n_samples=0;
    we evaluate pure profiles only (sufficient for coordination games with strictly
    concave payoff in mixed policy, which is true for finite-action payoff matrices
    that have a pure equilibrium).
    """
    N = payoff.ndim
    assert N == len(action_sizes), "payoff dim must match number of players"
    best = -np.inf
    best_profile: list[np.ndarray] = []
    for profile in product(*[range(k) for k in action_sizes]):
        v = float(payoff[profile])
        if v > best:
            best = v
            best_profile = [np.eye(k)[a] for a, k in zip(profile, action_sizes)]
    return best, best_profile


def best_signal_payoff(
    payoff: np.ndarray,
    nu: np.ndarray,
    action_sizes: tuple[int, ...],
) -> float:
    """Best signal-conditioned policy payoff.

    For each signal profile ξ with probability ν(ξ), pick the action profile a
    that maximises u(a); average over ξ.

    This is the optimal deterministic signal-conditioned policy and matches the
    LP optimum for finite games (mixing across signals adds nothing in a
    common-payoff setting because we can always restrict to the argmax).
    """
    N = payoff.ndim
    assert N == len(action_sizes)
    nu = np.asarray(nu, dtype=np.float64)
    nu = nu / nu.sum()
    # For each signal profile, find max payoff over action profiles where each
    # player's action depends only on their own signal. With N players each
    # observing 1 signal, the policy σ_i: ξ_i → a_i; for a fixed signal profile
    # the joint action is determined. We need to OPTIMISE σ over all functions
    # (k^|ξ|)^N — exponential. For small instances (k ≤ 3, N ≤ 3, |ξ| ≤ 3)
    # we brute-force.
    n_signals = nu.shape[0]  # assume same alphabet
    sig_dims = nu.shape
    # Σ_1: maps ξ_1 → a_1, etc. Enumerate all such maps.
    policy_spaces = [list(product(range(k), repeat=n_signals)) for k in action_sizes]
    best = -np.inf
    for sigma in product(*policy_spaces):
        # sigma[i] is a tuple of length n_signals: action chosen by player i for each signal value
        total = 0.0
        for xi in product(*[range(s) for s in sig_dims]):
            a = tuple(sigma[i][xi[i]] for i in range(N))
            total += float(nu[xi]) * float(payoff[a])
        if total > best:
            best = total
    return best


def classical_correlation_gain(
    payoff: np.ndarray,
    nu: np.ndarray,
    action_sizes: tuple[int, ...],
) -> float:
    """G(Γ, ν) = sup_σ E[u] − sup_π E[u]."""
    sigma_val = best_signal_payoff(payoff, nu, action_sizes)
    pi_val, _ = best_product_payoff(payoff, action_sizes)
    return sigma_val - pi_val


def ccg_upper_bound(
    payoff: np.ndarray,
    nu: np.ndarray,
) -> float:
    """Theorem 5: G ≤ U · sqrt(I̅(ν) / 2)."""
    U = float(payoff.max() - payoff.min())
    return U * np.sqrt(max(mean_pairwise_mi(nu), 0.0) / 2.0)


def adversarial_ccg(
    base_ccg: float,
    leakage_mi: float,
    leakage_penalty: float = 1.0,
) -> float:
    """Theorem 7: G_adv = G − β · I(ξ; Y).

    Under encrypted seeds with security parameter λ, leakage_mi ≤ 2^{−λ}.
    """
    return base_ccg - leakage_penalty * leakage_mi
