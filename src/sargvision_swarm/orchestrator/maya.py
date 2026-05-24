"""MAYA — Mean-field Adversarial Yielding Algorithm.

Strategic posture solver. Recomputes the friendly posture mix every ~30s
from the current hostile-class estimate, hedged against intel uncertainty,
and chooses an emission signal that maximises adversary classifier entropy.

The math from §4 of the formulations doc, compressed into a discrete game
that's cheap enough to solve in a single tick on CPU:

  posture vector  μ_F ∈ Δ^|A|   over A = {defend, intercept, recon, decoy, retreat}
  adversary      μ_H ∈ Δ^|B|   over B = {decoy, kinetic, nuisance}
  payoff matrix  ℓ ∈ R^{|A|×|B|}  (negative loss — we MAXIMISE)
  composite      max_{μ_F}  min_{μ_H ∈ U(μ̂_H, ε)}  μ_F^T ℓ μ_H  +  β H(θ|σ)

Solvers (no PyTorch dep — pure NumPy):
  * `nash_replicator(payoff)` — replicator dynamics → Nash on the 2-pop matrix game
  * `wasserstein_dro_inner(mu_F, payoff, mu_H_hat, eps)` — projected gradient
    finds the worst-case μ_H within TV-ball ε of the empirical estimate
    (TV ≤ W_1 on a finite simplex with cost = 1 between distinct classes)
  * `bayesian_persuasion(signals, prior, classifier)` — selects the signal
    that maximises adversary posterior entropy
  * `solve_maya(...)` — composes all of the above into one call returning
    the posture vector + chosen signal + telemetry

Updates LiveSession's per-tick params (SHIELD trust threshold, VAJRA
voronoi bonus, jamming-aware concurrency cap) from the posture mix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

POSTURE_ACTIONS = ("defend", "intercept", "recon", "decoy_emitter", "retreat")
HOSTILE_CLASSES = ("decoy", "kinetic", "nuisance")

# Per-row: friendly action; per-col: hostile class.  Entries = net friendly
# payoff (kills + intel — wasted shots − damage taken). Hand-tuned from the
# CNAS Sep 2025 + RUSI Sindoor analysis cited in §4 of the formulations doc.
DEFAULT_PAYOFF = np.array(
    [
        # decoy   kinetic  nuisance
        [-0.20, -0.60, 0.10],  # defend   — wastes shots on decoys but tanks kinetics
        [-0.40, 1.00, 0.20],  # intercept— shines vs kinetics, exposed to decoys
        [0.30, -0.10, 0.40],  # recon    — gathers intel cheaply
        [0.60, -0.30, 0.10],  # decoy_emitter — confuses real strikers
        [-0.05, -0.80, -0.05],  # retreat  — only good when overwhelmed
    ],
    dtype=np.float64,
)


# ── 1. Nash via replicator dynamics ──────────────────────────────────


def nash_replicator(
    payoff: np.ndarray,
    iters: int = 300,
    lr: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-population replicator dynamics → mean-field Nash.

    Returns (μ_F, μ_H) — friendly + hostile mixed strategies on the
    discrete posture × hostile-class game. Friendly maximises, hostile
    minimises (zero-sum reduction of the bilinear payoff).
    """
    n_f, n_h = payoff.shape
    mu_f = np.ones(n_f) / n_f
    mu_h = np.ones(n_h) / n_h
    for _ in range(iters):
        # Expected payoff per action vs current opponent mix
        f_fitness = payoff @ mu_h  # friendly value per action
        h_fitness = -payoff.T @ mu_f  # hostile value per action (opposite sign)
        # Replicator step (multiplicative weights)
        mu_f = mu_f * np.exp(lr * (f_fitness - f_fitness.dot(mu_f)))
        mu_h = mu_h * np.exp(lr * (h_fitness - h_fitness.dot(mu_h)))
        mu_f = mu_f / mu_f.sum()
        mu_h = mu_h / mu_h.sum()
    return mu_f, mu_h


# ── 2. Wasserstein-DRO inner loop ────────────────────────────────────


def _project_to_tv_ball(mu: np.ndarray, mu_hat: np.ndarray, eps: float) -> np.ndarray:
    """Project mu onto {ν : TV(ν, μ̂) ≤ ε, ν ∈ Δ} via bisection on a scaling λ.

    TV(ν, μ̂) = ½ Σ |ν - μ̂|. We blend toward μ̂ until the TV constraint holds.
    """
    mu = np.clip(mu, 1e-9, None)
    mu = mu / mu.sum()
    tv = 0.5 * np.abs(mu - mu_hat).sum()
    if tv <= eps:
        return mu
    # Convex interpolation toward mu_hat: ν = (1-λ) μ + λ μ̂.
    # TV is linear in λ on this segment → solve directly.
    lam = max(0.0, 1.0 - eps / max(tv, 1e-12))
    return (1.0 - lam) * mu + lam * mu_hat


def wasserstein_dro_inner(
    mu_F: np.ndarray,
    payoff: np.ndarray,
    mu_H_hat: np.ndarray,
    epsilon: float,
    iters: int = 200,
    lr: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Find the worst-case μ_H ∈ U(μ̂_H, ε) given the friendly mix μ_F.

    Returns (μ_H_worst, value). On a finite simplex with unit cost between
    distinct classes, the Wasserstein-1 ball equals the TV ball; the
    projection above is exact.
    """
    mu_H = mu_H_hat.copy()
    # Hostile gradient = derivative of μ_F^T ℓ μ_H wrt μ_H = ℓ^T μ_F.
    # Hostile MINIMISES the friendly objective.
    grad = payoff.T @ mu_F
    for _ in range(iters):
        mu_H = mu_H * np.exp(-lr * grad)
        mu_H = mu_H / mu_H.sum()
        mu_H = _project_to_tv_ball(mu_H, mu_H_hat, epsilon)
    value = float(mu_F @ payoff @ mu_H)
    return mu_H, value


# ── 3. Bayesian persuasion / formless info-max ───────────────────────


def shannon_entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def bayesian_persuasion(
    signal_likelihoods: np.ndarray,
    prior: np.ndarray,
) -> tuple[int, float, np.ndarray]:
    """Choose the signal σ that maximises adversary posterior entropy H(θ|σ).

    Args
    ----
    signal_likelihoods : (n_signals, n_intents). Row s gives the adversary
                         classifier's likelihood P(σ=s | θ) for each true intent θ.
    prior              : (n_intents,) adversary prior over our intent.

    Returns (best_signal_idx, entropy, posterior).
    """
    n_signals, n_intents = signal_likelihoods.shape
    best_idx = 0
    best_H = -np.inf
    best_post = prior.copy()
    for s in range(n_signals):
        likelihood = signal_likelihoods[s]
        unnorm = likelihood * prior
        z = unnorm.sum()
        if z < 1e-12:
            continue
        post = unnorm / z
        H = shannon_entropy(post)
        if H > best_H:
            best_H = H
            best_idx = s
            best_post = post
    return best_idx, float(best_H), best_post


# ── 4. End-to-end solver ─────────────────────────────────────────────


@dataclass
class MayaParams:
    payoff: np.ndarray = field(default_factory=lambda: DEFAULT_PAYOFF.copy())
    wasserstein_epsilon: float = 0.15
    entropy_weight: float = 0.20
    refresh_seconds: float = 30.0


@dataclass
class MayaState:
    posture: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.4, 0.1, 0.1, 0.1]))
    hostile_posterior: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.5, 0.1]))
    worst_case_hostile: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.5, 0.1]))
    signal_choice: int = 0
    classifier_entropy: float = 0.0
    last_value: float = 0.0
    last_solved_t: float = -1e9
    n_solves: int = 0


@dataclass
class MayaSolution:
    posture: np.ndarray
    hostile_worst: np.ndarray
    signal: int
    entropy: float
    value: float


def estimate_hostile_mix(posteriors_by_hostile: list[np.ndarray]) -> np.ndarray:
    """Aggregate SHIELD's per-hostile threat posteriors into μ̂_H (the empirical mix)."""
    if not posteriors_by_hostile:
        return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    stacked = np.stack(posteriors_by_hostile, axis=0)
    mean = stacked.mean(axis=0)
    return mean / mean.sum()


def solve_maya(
    hostile_posterior_est: np.ndarray,
    signal_likelihoods: np.ndarray | None = None,
    adversary_prior: np.ndarray | None = None,
    params: MayaParams | None = None,
) -> MayaSolution:
    """One-shot MAYA solve.  Cheap enough for live use.

    Steps:
      1. Replicator-dynamics Nash → μ_F^Nash
      2. Wasserstein-DRO inner loop → μ_H_worst, value
      3. Replicator AGAIN on payoff vs μ_H_worst → μ_F^DRO (best response to worst case)
      4. Bayesian persuasion → choose signal maximising H(θ|σ)
      5. Composite value = μ_F · ℓ · μ_H_worst + β H
    """
    p = params or MayaParams()
    # 1. Nash starting point
    mu_F_nash, _mu_H_nash = nash_replicator(p.payoff)
    # 2. Worst-case adversary in W-ball around the empirical estimate
    mu_H_worst, _ = wasserstein_dro_inner(
        mu_F_nash,
        p.payoff,
        hostile_posterior_est,
        epsilon=p.wasserstein_epsilon,
    )
    # 3. Best-response posture vs the worst case
    # Solve a single-population fitness: μ_F maximises payoff @ mu_H_worst
    fitness = p.payoff @ mu_H_worst
    mu_F_dro = np.exp(8.0 * (fitness - fitness.max()))
    mu_F_dro = mu_F_dro / mu_F_dro.sum()
    value = float(mu_F_dro @ p.payoff @ mu_H_worst)
    # 4. Bayesian persuasion signal
    if signal_likelihoods is not None and adversary_prior is not None:
        sig, H, _post = bayesian_persuasion(signal_likelihoods, adversary_prior)
        composite = value + p.entropy_weight * H
    else:
        sig, H, composite = 0, 0.0, value
    return MayaSolution(
        posture=mu_F_dro,
        hostile_worst=mu_H_worst,
        signal=sig,
        entropy=H,
        value=composite,
    )


def maya_tick(
    sim_time: float,
    state: MayaState,
    hostile_posteriors: list[np.ndarray],
    signal_likelihoods: np.ndarray | None = None,
    adversary_prior: np.ndarray | None = None,
    params: MayaParams | None = None,
    force: bool = False,
) -> tuple[bool, MayaSolution | None]:
    """Cadence-gated MAYA tick. Resolves every params.refresh_seconds.

    Returns (did_solve, solution_or_None). Updates `state` in place.
    """
    p = params or MayaParams()
    due = (sim_time - state.last_solved_t) >= p.refresh_seconds
    if not (due or force):
        return False, None
    mu_H_hat = estimate_hostile_mix(hostile_posteriors)
    state.hostile_posterior = mu_H_hat
    sol = solve_maya(
        hostile_posterior_est=mu_H_hat,
        signal_likelihoods=signal_likelihoods,
        adversary_prior=adversary_prior,
        params=p,
    )
    state.posture = sol.posture
    state.worst_case_hostile = sol.hostile_worst
    state.signal_choice = sol.signal
    state.classifier_entropy = sol.entropy
    state.last_value = sol.value
    state.last_solved_t = sim_time
    state.n_solves += 1
    return True, sol


def posture_dict(posture: np.ndarray) -> dict[str, float]:
    return {a: float(posture[i]) for i, a in enumerate(POSTURE_ACTIONS)}
