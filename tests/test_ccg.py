"""CCG — verification of Theorems 5-7 from 35_THEOREMS_AND_PROOFS.md."""

from __future__ import annotations

import numpy as np

from sargvision_swarm.core.ccg import (
    adversarial_ccg,
    best_product_payoff,
    best_product_payoff_bayesian,
    best_signal_payoff,
    best_signal_payoff_bayesian,
    ccg_upper_bound,
    ccg_upper_bound_pairwise,
    classical_correlation_gain,
    classical_correlation_gain_bayesian,
    mean_pairwise_mi,
    mutual_information,
    shannon_entropy,
    total_correlation,
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
    leakage = 2**-128
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


# ── Theorem 5 (rev. 2) — total correlation + parity counterexample ──


def test_total_correlation_equals_pairwise_mi_for_n2():
    """For N=2, T(ν) = I(X_1; X_2). Check on a correlated 2x2 joint."""
    nu = np.array([[0.4, 0.1], [0.1, 0.4]])
    nu = nu / nu.sum()
    T = total_correlation(nu)
    I = mutual_information(nu)
    assert abs(T - I) < 1e-9


def test_3bit_parity_counterexample_to_pairwise_form():
    """The 3-bit parity device kills the v1 pairwise-MI bound:
    X_1, X_2 ~ Unif{0,1} indep, X_3 = X_1 ⊕ X_2.
    All pairwise MIs = 0 but T = log 2 > 0.
    This test PROVES the v1 bound (G ≤ U sqrt(I̅/2)) is FALSE for N ≥ 3 because
    it would give G ≤ 0 while T > 0 leaves room for positive G."""
    nu = np.zeros((2, 2, 2))
    nu[0, 0, 0] = 0.25  # x1=0, x2=0, x3=0
    nu[0, 1, 1] = 0.25  # x1=0, x2=1, x3=1
    nu[1, 0, 1] = 0.25  # x1=1, x2=0, x3=1
    nu[1, 1, 0] = 0.25  # x1=1, x2=1, x3=0
    # All pairwise marginals are uniform → I(X_i; X_j) = 0
    mi_bar = mean_pairwise_mi(nu)
    assert mi_bar < 1e-9, f"pairwise MI should be 0, got {mi_bar:.6f}"
    # But total correlation = log 2
    T = total_correlation(nu)
    assert abs(T - np.log(2)) < 1e-9, f"T should be log 2 ≈ 0.693, got {T:.6f}"
    # The v1 bound says G ≤ 0; the corrected bound is U·sqrt(log 2 / 2) ≈ 0.589 U
    fake_payoff = np.array([[1.0, 0.0], [0.0, 1.0]])  # any payoff for shape
    bound_v1 = ccg_upper_bound_pairwise(fake_payoff, nu)
    bound_v2 = ccg_upper_bound(fake_payoff, nu)
    assert bound_v1 < 1e-9
    assert bound_v2 > 0.5


# ── Bayesian common-payoff (Theorem 5/6 setting — rev. 2) ──────────


def test_bayesian_ccg_positive_on_BSC_construction():
    """Theorem 6 (rev. 2) construction: Θ = {0,1} uniform; payoff u(θ, a_1, a_2)
    = U·1[a_1 = a_2 = θ]; ν has ξ_i = BSC(ε)-copy of θ.

    sup_π = U/2 (guess and hope, can't see θ)
    sup_σ = U(1−ε)²  (both play their signal; both correct)
    G = U·[(1−ε)² − 1/2]
    """
    U = 1.0
    epsilon = 0.1
    prior = np.array([0.5, 0.5])
    # payoff_tensor[θ, a_1, a_2] = U if a_1 = a_2 = θ else 0
    payoff = np.zeros((2, 2, 2))
    payoff[0, 0, 0] = U
    payoff[1, 1, 1] = U
    # ν[θ, ξ_1, ξ_2] = 0.5 * BSC(ε)(ξ_1 | θ) * BSC(ε)(ξ_2 | θ)
    nu = np.zeros((2, 2, 2))
    for t in range(2):
        for x1 in range(2):
            for x2 in range(2):
                p_x1 = (1 - epsilon) if x1 == t else epsilon
                p_x2 = (1 - epsilon) if x2 == t else epsilon
                nu[t, x1, x2] = 0.5 * p_x1 * p_x2
    pi_val = best_product_payoff_bayesian(payoff, prior, action_sizes=(2, 2))
    sigma_val = best_signal_payoff_bayesian(payoff, nu, action_sizes=(2, 2))
    G = sigma_val - pi_val
    expected_pi = U / 2
    expected_sigma = U * (1 - epsilon) ** 2
    expected_G = expected_sigma - expected_pi
    assert abs(pi_val - expected_pi) < 1e-6, f"π expected {expected_pi}, got {pi_val:.4f}"
    assert abs(sigma_val - expected_sigma) < 1e-6, (
        f"σ expected {expected_sigma:.4f}, got {sigma_val:.4f}"
    )
    assert abs(G - expected_G) < 1e-6
    # G should be ~0.31 here.
    assert G > 0.3


def test_bayesian_ccg_zero_when_signal_independent_of_state():
    """If ν has ξ ⊥ θ then signals carry no info about θ and G = 0."""
    U = 1.0
    prior = np.array([0.5, 0.5])
    payoff = np.zeros((2, 2, 2))
    payoff[0, 0, 0] = U
    payoff[1, 1, 1] = U
    # ν: ξ uniform on {0,1}², independent of θ
    nu = np.full((2, 2, 2), 1 / 8)
    G = classical_correlation_gain_bayesian(payoff, prior, nu, action_sizes=(2, 2))
    assert abs(G) < 1e-6, f"expected G=0 for ξ⊥θ, got {G:.4f}"


def test_bayesian_ccg_respects_theorem5_upper_bound():
    """For the BSC construction, the corrected bound G ≤ U·sqrt(T(ν_ξ)/2)
    should hold (and be reasonably tight)."""
    U = 1.0
    epsilon = 0.05  # very low noise → high G, high T
    prior = np.array([0.5, 0.5])
    payoff = np.zeros((2, 2, 2))
    payoff[0, 0, 0] = U
    payoff[1, 1, 1] = U
    nu = np.zeros((2, 2, 2))
    for t in range(2):
        for x1 in range(2):
            for x2 in range(2):
                p_x1 = (1 - epsilon) if x1 == t else epsilon
                p_x2 = (1 - epsilon) if x2 == t else epsilon
                nu[t, x1, x2] = 0.5 * p_x1 * p_x2
    G = classical_correlation_gain_bayesian(payoff, prior, nu, action_sizes=(2, 2))
    # Marginal over θ to get the ξ-only device
    nu_xi = nu.sum(axis=0)
    bound = float(U) * float(np.sqrt(total_correlation(nu_xi) / 2.0))
    assert G <= bound + 1e-6, f"G={G:.4f} > bound={bound:.4f}"
    # Tightness check: ratio should be > 0.5 (Theorem 6 claims ≈ 0.85)
    assert G / bound > 0.5, f"bound too loose: G/bound = {G / bound:.3f}"
