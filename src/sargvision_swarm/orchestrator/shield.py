"""SHIELD — Sheaf-Harmonic Identity & Engagement-Loyalty Defender.

Composes four primitives:
  1. Sheaf-Laplacian sensor cross-check → per-drone loyalty
  2. Damped PageRank trust propagation
  3. Bayesian posterior over hostile threat class
  4. Trust-weighted engagement priority for auction allocation

Hostile threat classes: decoy / kinetic / nuisance. Real kinetic hostiles
look like ballistic-trajectory + heat signature + small RCS. Decoys are
foam + Luneburg lens with overpowered emitters. We discriminate by
trajectory smoothness + emitted-RF/observed-RCS ratio.

The bid an interceptor i emits for hostile j is:

    E_ij = trust_i * E_theta[damage(theta_j)] / dist(i,j)^alpha

Decoys carry near-zero expected damage so interceptors don't waste shots.
Hijacked teammates carry low trust so their (likely adversarial) bids
are ignored.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class ShieldParams:
    sigma_n: float = 1.5
    smoothing: float = 0.6
    pagerank_damping: float = 0.85
    pagerank_iters: int = 12
    distance_falloff: float = 1.5
    decoy_damage: float = 0.05
    kinetic_damage: float = 1.0
    nuisance_damage: float = 0.15
    trust_kill_threshold: float = 0.25  # below this, drone is "kill-switched"


@dataclass
class ShieldState:
    loyalty: np.ndarray = field(default_factory=lambda: np.zeros(0))
    trust: np.ndarray = field(default_factory=lambda: np.zeros(0))
    posteriors: dict[int, np.ndarray] = field(default_factory=dict)

    def init(self, n: int) -> None:
        self.loyalty = np.ones(n, dtype=np.float64)
        self.trust = np.ones(n, dtype=np.float64)
        self.posteriors = {}


THREAT_CLASSES = ("decoy", "kinetic", "nuisance")
PRIOR = np.array([0.33, 0.50, 0.17])  # mild Bayesian prior — most contacts are kinetic


def update_threat_posterior(
    state: ShieldState,
    hostile_id: int,
    observation: dict,
    smoothing: float = 0.85,
) -> np.ndarray:
    """Bayesian update on hostile threat class.

    observation features (heuristics tuned to sim):
      - rcs:  observed radar cross section, normalised to [0, 1]
      - rf_emit: observed RF emission, [0, 1] (decoys overemit)
      - traj_jerk: trajectory smoothness inverse, [0, 1] (kinetics smoother)
      - terminal: 1 if drone is in TERMINAL mode (nuisance unlikely)
    """
    prev = state.posteriors.get(hostile_id, PRIOR.copy())
    rcs = float(observation.get("rcs", 0.5))
    rf = float(observation.get("rf_emit", 0.5))
    jerk = float(observation.get("traj_jerk", 0.3))
    terminal = float(observation.get("terminal", 0.0))

    # Likelihoods per class (decoy / kinetic / nuisance)
    # Decoy: high rf, high rcs, high jerk
    L_decoy = 0.1 + 0.4 * rf + 0.3 * rcs + 0.2 * jerk
    # Kinetic: low jerk, terminal, mid rf
    L_kinetic = 0.1 + 0.6 * (1.0 - jerk) + 0.3 * terminal
    # Nuisance: low rcs, low rf, low terminal
    L_nuisance = 0.05 + 0.4 * (1.0 - rf) + 0.4 * (1.0 - terminal)

    L = np.array([L_decoy, L_kinetic, L_nuisance])
    bayes = L * prev
    bayes = bayes / max(bayes.sum(), 1e-9)
    # smooth so a single noisy obs doesn't flip the class
    posterior = smoothing * prev + (1.0 - smoothing) * bayes
    posterior = posterior / max(posterior.sum(), 1e-9)
    state.posteriors[hostile_id] = posterior
    return posterior


def expected_damage(posterior: np.ndarray, params: ShieldParams) -> float:
    return float(
        posterior[0] * params.decoy_damage
        + posterior[1] * params.kinetic_damage
        + posterior[2] * params.nuisance_damage
    )


def threat_class(posterior: np.ndarray) -> str:
    return THREAT_CLASSES[int(np.argmax(posterior))]


def shield_priorities(
    friendly_positions: np.ndarray,
    friendly_roles: list,
    hostiles: list,
    adjacency: np.ndarray,
    state: ShieldState,
    params: ShieldParams | None = None,
    spoofed_ids: set[int] | None = None,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Run a SHIELD tick and return the per-pair priority matrix.

    Returns (E, friendly_indices, hostile_ids) where
      E[i, j] = trust[i] * E[damage(θ_j)] / dist(i, j)^α
    is zero for non-strikers, kill-switched bidders, and non-TERMINAL
    hostiles. SHIELD state (loyalty / trust / posteriors) is updated in
    place.

    This is the building block VAJRA composes on top of — VAJRA takes
    E and adds Voronoi-hysteresis ownership + tropical-attention
    resolution under jamming-aware concurrency.
    """
    from sargvision_swarm.core.sheaf import (
        loyalty_from_positions, SheafState, SheafParams,
    )
    from sargvision_swarm.comms.trust import pagerank_trust

    p = params or ShieldParams()
    n = friendly_positions.shape[0]
    if state.loyalty.shape[0] != n:
        state.init(n)

    # ── 1. Loyalty via sheaf-Laplacian residual
    sheaf_state = SheafState(loyalty=state.loyalty)
    loyalty = loyalty_from_positions(
        friendly_positions, adjacency, sheaf_state,
        SheafParams(sigma_n=p.sigma_n, smoothing=p.smoothing),
        spoofed_ids=spoofed_ids,
    )
    state.loyalty = loyalty

    # ── 2. Trust via damped PageRank
    state.trust = pagerank_trust(
        adjacency, loyalty,
        damping=p.pagerank_damping, iters=p.pagerank_iters,
    )

    # ── 3. Update posteriors for every alive hostile
    eligible_hostiles = [h for h in hostiles if h.alive]
    for h in eligible_hostiles:
        obs = {
            "rcs": getattr(h, "rcs", 0.5),
            "rf_emit": getattr(h, "rf_emit", 0.5),
            "traj_jerk": getattr(h, "traj_jerk", 0.3),
            "terminal": 1.0 if h.intent_label == "TERMINAL" else 0.0,
        }
        update_threat_posterior(state, h.id, obs)

    # ── 4. Build priority matrix over (strikers × TERMINAL hostiles).
    terminal_hostiles = [h for h in eligible_hostiles if h.intent_label == "TERMINAL"]
    m = len(terminal_hostiles)
    E = np.zeros((n, m), dtype=np.float64)
    hostile_ids = [h.id for h in terminal_hostiles]
    for j, h in enumerate(terminal_hostiles):
        post = state.posteriors.get(h.id, PRIOR.copy())
        e_dmg = expected_damage(post, p)
        for i, role in enumerate(friendly_roles):
            if role != "worker":
                continue
            if state.trust[i] < p.trust_kill_threshold:
                continue
            d = float(np.linalg.norm(friendly_positions[i] - h.pos))
            E[i, j] = state.trust[i] * e_dmg / (d ** p.distance_falloff + 1e-6)
    return E, list(range(n)), hostile_ids


def shield_assign(
    friendly_positions: np.ndarray,
    friendly_roles: list,
    hostiles: list,
    adjacency: np.ndarray,
    state: ShieldState,
    params: ShieldParams | None = None,
    spoofed_ids: set[int] | None = None,
    already_assigned: dict[int, int] | None = None,
) -> dict[int, int]:
    """Run SHIELD tick and return assignment {friendly_id: hostile_id}.

    Greedy resolution of the priority matrix — kept for callers that don't
    want VAJRA's Voronoi + tropical attention layer.
    """
    E, friendly_ids, hostile_ids = shield_priorities(
        friendly_positions, friendly_roles, hostiles, adjacency,
        state, params, spoofed_ids,
    )
    already_assigned = already_assigned or {}

    # Mask out already-assigned friendlies + hostiles.
    if E.size == 0:
        return {}
    for i in already_assigned.keys():
        if 0 <= i < E.shape[0]:
            E[i, :] = 0.0
    for hid in already_assigned.values():
        if hid in hostile_ids:
            j = hostile_ids.index(hid)
            E[:, j] = 0.0

    # Greedy resolve: highest priority first, one drone per hostile.
    bids: list[tuple[float, int, int]] = []
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if E[i, j] > 0:
                bids.append((float(E[i, j]), i, hostile_ids[j]))
    bids.sort(reverse=True)
    used_friend: set[int] = set()
    used_host: set[int] = set()
    assignment: dict[int, int] = {}
    for _prio, fid, hid in bids:
        if fid in used_friend or hid in used_host:
            continue
        assignment[fid] = hid
        used_friend.add(fid)
        used_host.add(hid)
    return assignment
