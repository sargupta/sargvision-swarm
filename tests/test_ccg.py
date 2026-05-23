"""CCG — verification of Theorems 5-7 from 35_THEOREMS_AND_PROOFS.md."""

from __future__ import annotations

import numpy as np

from sargvision_swarm.core.ccg import (
    adversarial_ccg,
    best_product_payoff,
    best_signal_payoff,
    ccg_upper_bound,
    classical_correlation_gain,
    mean_pairwise_mi,
    mutual_information,
    shannon_entropy,
)


# ── Definition 2 sanity ──────────────────────────────────────────────


def test_ccg_nonnegative_for_coordination_game():
    """CCG ≥ 0 by construction (product policy is a degenerate device)."""
    payoff = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Perfect-correlation device — players see same signal
    nu = np.array([[0.5, 0.0], [0.0, 0.5]])
    G = classical_correlation_gain(payoff, nu, action_sizes=(2, 2))
    assert G >= -1e-9


def test_ccg_is_zero_for_independent_signals():
    """Independent product device has G = 0."""
    payoff = np.array([[1.0, 0.0], [0.0, 1.0]])
    nu = np.array([[0.25, 0.25], [0.25, 0.25]])  # independent uniform
    G = classical_correlation_gain(payoff, nu, action_sizes=(2, 2))
    # Independent signals — players can already coordinate on the constant strategy
    # Coordination game has G = 0 because best product (a=b=0 always) already wins.
    assert abs(G) < 1e-9


# ── Theorem 5 (Upper bound G ≤ U sqrt(I̅/2)) ─────────────────────────


def test_ccg_upper_bound_holds_for_random_devices():
    """Theorem 5: empirical CCG ≤ U·sqrt(I̅(ν)/2) for every device."""
    rng = np.random.default_rng(0)
    payoff = np.array([[1.0, 0.0], [0.0, 1.0]])
    for _ in range(10):
        # Random 2×2 joint
        nu = rng.dirichlet([1, 1, 1, 1]).reshape(2, 2)
        G = classical_correlation_gain(payoff, nu, action_sizes=(2, 2))
        bound = ccg_upper_bound(payoff, nu)
        assert G <= bound + 1e-9, f"G={G:.4f} > bound={bound:.4f}"


def test_ccg_bound_upper_bound_satisfied():
    """Theorem 5 holds for any 2×2 device, even pure-coordination games where
    G is trivially 0 (best product policy already wins)."""
    payoff = np.array([[1.0, 0.0], [0.0, 1.0]])
    nu = np.array([[0.5, 0.0], [0.0, 0.5]])
    G = classical_correlation_gain(payoff, nu, action_sizes=(2, 2))
    bound = ccg_upper_bound(payoff, nu)
    assert G <= bound + 1e-9
    # The bound is loose here because best product is already optimal — this is
    # the expected behaviour for common-payoff games with a dominant pure profile.


# ── Theorem 6 (Tightness — deferred, needs state-dependent payoff) ───
# Definition 2 as currently stated covers common-payoff games where signals don't
# alter the payoff. In this regime G ≥ 0 but G = 0 whenever a dominant pure profile
# exists. The Theorem 6 construction in 35_THEOREMS_AND_PROOFS.md needs payoffs
# that depend on the hidden state revealed by the shared seed (a Bayesian-game
# extension). That extension is deferred to the second-pass mathematician review;
# see 35_THEOREMS_AND_PROOFS.md "Part III — Honest assessment" for the full caveat.


# ── Theorem 7 (Adversarial CCG positivity) ───────────────────────────


def test_adversarial_ccg_survives_encrypted_seed():
    """Theorem 7: G_adv ≥ G − β · 2^{−λ}.

    With λ = 128, the leakage penalty is astronomical; G_adv ≈ G.
    """
    G = 0.5
    leakage = 2 ** -128
    beta = 1.0
    G_adv = adversarial_ccg(G, leakage, beta)
    assert abs(G_adv - G) < 1e-30  # essentially no loss


def test_adversarial_ccg_degrades_with_high_leakage():
    G = 0.5
    leakage = 0.4
    beta = 1.0
    G_adv = adversarial_ccg(G, leakage, beta)
    assert G_adv == 0.5 - 0.4


# ── Information-measure sanity ───────────────────────────────────────


def test_mi_zero_for_independent_pairs():
    nu = np.array([[0.25, 0.25], [0.25, 0.25]])
    assert abs(mutual_information(nu)) < 1e-9


def test_mi_positive_for_correlated_pairs():
    nu = np.array([[0.5, 0.0], [0.0, 0.5]])
    assert mutual_information(nu) > 0.6


def test_mean_pairwise_mi_correct_for_3_var_block():
    # Three variables, all identical with uniform prior on {0,1} → all pairs I = log 2
    nu = np.zeros((2, 2, 2))
    nu[0, 0, 0] = 0.5
    nu[1, 1, 1] = 0.5
    mi_bar = mean_pairwise_mi(nu)
    expected = np.log(2)
    assert abs(mi_bar - expected) < 0.05


def test_mean_pairwise_mi_zero_for_independent_block():
    rng = np.random.default_rng(0)
    nu = rng.dirichlet([1] * 8).reshape(2, 2, 2)
    # Force independence by replacing with product of marginals
    p_x = nu.sum(axis=(1, 2))
    p_y = nu.sum(axis=(0, 2))
    p_z = nu.sum(axis=(0, 1))
    indep = p_x[:, None, None] * p_y[None, :, None] * p_z[None, None, :]
    indep = indep / indep.sum()
    mi_bar = mean_pairwise_mi(indep)
    assert mi_bar < 1e-6
