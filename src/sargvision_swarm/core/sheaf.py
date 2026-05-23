"""Sheaf-Laplacian sensor cross-checking — SHIELD loyalty layer.

Each drone is a stalk over a vertex of a cellular sheaf on the comm graph.
Loyal drones produce sensor reports whose pairwise disagreements stay
bounded by sensor noise; spoofed or hijacked drones blow up the Dirichlet
residual on the sheaf Laplacian.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SheafParams:
    sigma_n: float = 1.5
    smoothing: float = 0.65
    spoof_bias_m: float = 6.0


@dataclass
class SheafState:
    loyalty: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def init(self, n: int) -> None:
        self.loyalty = np.ones(n, dtype=np.float64)


def loyalty_from_positions(
    positions: np.ndarray,
    adjacency: np.ndarray,
    state: SheafState,
    params: SheafParams | None = None,
    spoofed_ids: set[int] | None = None,
) -> np.ndarray:
    """Per-drone loyalty via Dirichlet residual on the comm graph.

    Each drone reports its observed centroid relative to itself.
    Spoofed drones inject a bias vector that creates inconsistency with
    neighbours. The per-vertex sheaf Laplacian residual then deviates from
    the noise floor, dropping loyalty toward 0.
    """
    n = positions.shape[0]
    if state.loyalty.shape[0] != n:
        state.init(n)
    p = params or SheafParams()

    # ── Each drone's report in a COMMON frame (the swarm-centroid estimate).
    # Loyal drones report `centroid + noise`; spoofed drones add a large bias.
    # Reports must be in a common frame for sheaf disagreement to be meaningful —
    # restriction maps in a real cellular sheaf transform local frames into
    # the global frame; here we model that translation as already applied.
    rng = np.random.default_rng()
    truth = np.tile(positions.mean(axis=0), (n, 1))    # (N, 3) — same centroid
    noise = rng.normal(scale=p.sigma_n * 0.3, size=truth.shape)
    reports = truth + noise
    if spoofed_ids:
        for sid in spoofed_ids:
            if 0 <= sid < n:
                bias = rng.normal(scale=1.0, size=3)
                bias = bias / (np.linalg.norm(bias) + 1e-9) * p.spoof_bias_m
                reports[sid] = reports[sid] + bias

    # ── Dirichlet residual: per drone, MEDIAN edge disagreement.
    # Median (vs L2 of sum) is robust to a minority of spoofed neighbours —
    # a loyal vertex with k bad neighbours out of N still reads ~noise floor
    # so long as k < N/2. The bad vertex itself disagrees with EVERY loyal
    # neighbour and its median stays high.
    residual = np.zeros(n)
    for i in range(n):
        nbrs = np.where(adjacency[i])[0]
        if len(nbrs) == 0:
            residual[i] = 0.0
            continue
        diffs = reports[i] - reports[nbrs]            # (k, 3)
        edge_mag = np.linalg.norm(diffs, axis=1)       # (k,)
        residual[i] = float(np.median(edge_mag))

    # ── Loyalty = Gaussian on residual, EMA-smoothed across ticks
    raw_loyalty = np.exp(-(residual ** 2) / (2.0 * p.sigma_n ** 2))
    state.loyalty = p.smoothing * state.loyalty + (1 - p.smoothing) * raw_loyalty
    return state.loyalty.copy()
