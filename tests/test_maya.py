"""MAYA — replicator Nash + Wasserstein-DRO + Bayesian persuasion."""

from __future__ import annotations

import numpy as np

from sargvision_swarm.orchestrator.maya import (
    DEFAULT_PAYOFF,
    HOSTILE_CLASSES,
    POSTURE_ACTIONS,
    MayaState,
    bayesian_persuasion,
    estimate_hostile_mix,
    maya_tick,
    nash_replicator,
    posture_dict,
    shannon_entropy,
    solve_maya,
    wasserstein_dro_inner,
)

# ── 1. Nash replicator ────────────────────────────────────────────────


def test_replicator_converges_to_matching_pennies_nash():
    # Matching pennies: row maximises, col minimises; unique Nash = (0.5, 0.5).
    payoff = np.array(
        [
            [+1.0, -1.0],
            [-1.0, +1.0],
        ]
    )
    mu_f, mu_h = nash_replicator(payoff, iters=2000, lr=0.05)
    assert abs(mu_f[0] - 0.5) < 0.05, f"row Nash: {mu_f}"
    assert abs(mu_h[0] - 0.5) < 0.05, f"col Nash: {mu_h}"


def test_replicator_pure_strategy_when_dominant():
    # Row 0 dominates row 1 in every column → friendly should put all mass on row 0.
    payoff = np.array(
        [
            [+2.0, +3.0],
            [-1.0, +0.5],
        ]
    )
    mu_f, _ = nash_replicator(payoff, iters=500, lr=0.1)
    assert mu_f[0] > 0.95, f"dominant strategy not selected: {mu_f}"


# ── 2. Wasserstein-DRO ────────────────────────────────────────────────


def test_dro_shifts_hostile_to_worst_case_within_ball():
    mu_F = np.array([0.0, 1.0, 0.0, 0.0, 0.0])  # all-intercept
    mu_H_hat = np.array([0.4, 0.5, 0.1])  # mostly kinetic
    mu_H_worst, val_dro = wasserstein_dro_inner(
        mu_F,
        DEFAULT_PAYOFF,
        mu_H_hat,
        epsilon=0.20,
    )
    # Adversary should shift mass AWAY from kinetic (row 'intercept' beats kinetic)
    # and TOWARD decoy (row 'intercept' loses to decoy).
    assert mu_H_worst[1] < mu_H_hat[1], "worst-case did not reduce kinetic share"
    assert mu_H_worst[0] > mu_H_hat[0], "worst-case did not increase decoy share"
    # Worst-case value should be ≤ optimistic value at the empirical estimate.
    val_nominal = float(mu_F @ DEFAULT_PAYOFF @ mu_H_hat)
    assert val_dro <= val_nominal + 1e-6


def test_dro_collapses_to_empirical_when_epsilon_zero():
    mu_F = np.array([0.2, 0.5, 0.1, 0.1, 0.1])
    mu_H_hat = np.array([0.3, 0.5, 0.2])
    mu_H_worst, _ = wasserstein_dro_inner(mu_F, DEFAULT_PAYOFF, mu_H_hat, epsilon=0.0)
    assert np.linalg.norm(mu_H_worst - mu_H_hat) < 1e-3


# ── 3. Bayesian persuasion ────────────────────────────────────────────


def test_persuasion_picks_high_entropy_signal():
    # Three signals: first is informative (delta), third is uniform-likelihood (uninformative).
    prior = np.array([0.5, 0.5])
    likelihoods = np.array(
        [
            [1.0, 0.0],  # signal 0 → fully reveals intent 0
            [0.0, 1.0],  # signal 1 → fully reveals intent 1
            [0.5, 0.5],  # signal 2 → uninformative (preserves prior entropy)
        ]
    )
    sig, H, post = bayesian_persuasion(likelihoods, prior)
    assert sig == 2, f"persuasion should pick uninformative signal, got {sig}"
    assert abs(H - shannon_entropy(prior)) < 1e-6


# ── 4. solve_maya end-to-end ──────────────────────────────────────────


def test_solve_maya_produces_simplex_posture():
    sol = solve_maya(hostile_posterior_est=np.array([0.4, 0.5, 0.1]))
    assert sol.posture.shape == (len(POSTURE_ACTIONS),)
    assert abs(sol.posture.sum() - 1.0) < 1e-6
    assert (sol.posture >= 0).all()


def test_solve_maya_intercept_favoured_when_kinetic_dominates():
    sol = solve_maya(hostile_posterior_est=np.array([0.05, 0.90, 0.05]))
    pd = posture_dict(sol.posture)
    assert pd["intercept"] > 0.3, f"intercept share too low: {pd}"


def test_solve_maya_recon_or_decoy_favoured_when_decoys_dominate():
    sol = solve_maya(hostile_posterior_est=np.array([0.90, 0.05, 0.05]))
    pd = posture_dict(sol.posture)
    # When decoys dominate, intercept is a trap; defensive postures + decoy emitters win.
    assert pd["intercept"] < 0.4
    assert (pd["decoy_emitter"] + pd["recon"] + pd["defend"]) > 0.4


# ── 5. estimate_hostile_mix ───────────────────────────────────────────


def test_estimate_hostile_mix_aggregates_posteriors():
    posts = [
        np.array([0.1, 0.8, 0.1]),
        np.array([0.2, 0.7, 0.1]),
        np.array([0.3, 0.6, 0.1]),
    ]
    mix = estimate_hostile_mix(posts)
    assert abs(mix.sum() - 1.0) < 1e-9
    assert mix[1] > 0.5
    assert mix[2] < 0.2


# ── 6. maya_tick cadence ──────────────────────────────────────────────


def test_maya_tick_respects_refresh_cadence():
    state = MayaState()
    posts = [np.array([0.3, 0.5, 0.2])]
    # First call solves.
    did, _ = maya_tick(sim_time=0.0, state=state, hostile_posteriors=posts)
    assert did
    # Call again 10s later — refresh window is 30s → no re-solve.
    did2, _ = maya_tick(sim_time=10.0, state=state, hostile_posteriors=posts)
    assert not did2
    # 35s later — refresh fires.
    did3, _ = maya_tick(sim_time=35.0, state=state, hostile_posteriors=posts)
    assert did3
    assert state.n_solves == 2
