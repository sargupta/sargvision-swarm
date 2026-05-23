"""VAJRA — Voronoi-Auction Jam-Resilient Allocator.

Composes four primitives that sit ON TOP of SHIELD's per-pair priority E_ij:

  1. HSL-CC Lanchester with concurrency cap — break-even mass theorem
     gives a closed-form interceptor-fleet sizing recommendation per AFB.
  2. Dynamic Voronoi cell ownership with hysteresis — provisional assignment
     by airspace, no ping-pong handovers.
  3. Tropical-attention assignment — sub-10ms allocation over the priority
     matrix, replaces SHIELD's greedy auction.
  4. Algebraic-connectivity fragmentation alarm — λ₂(L_G) crash triggers
     local-cluster fallback rather than a failing global assignment.

VAJRA decomposes a single counter-swarm problem into per-component sub-
problems when the comm graph fragments — graceful degradation, not collapse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np

from sargvision_swarm.core.tropical import tropical_attention_assignment


# ── 1. Lanchester break-even ─────────────────────────────────────────


def break_even_interceptors(
    hostiles: int,
    friendly_kill_rate: float,
    hostile_kill_rate: float,
    jamming_factor: float = 0.0,
    safety_margin: float = 1.10,
) -> int:
    """Closed-form solution to N² a > M² b (1 − κ_jam).

    Inputs
    ------
    hostiles            : M, incoming wave size.
    friendly_kill_rate  : a, friendly per-unit kill rate (kills / unit time / unit).
    hostile_kill_rate   : b, hostile per-unit damage rate.
    jamming_factor      : κ_jam ∈ [0, 1] — concurrency penalty under EW.
    safety_margin       : multiplicative buffer over the break-even mass.

    Returns the minimum integer N such that the friendly survives.
    """
    if friendly_kill_rate <= 0:
        return 10**9
    kappa = float(np.clip(jamming_factor, 0.0, 0.999))
    ratio = (hostile_kill_rate / friendly_kill_rate) * (1.0 - kappa)
    n_star = hostiles * sqrt(max(ratio, 0.0))
    # Round before ceil to avoid float-imprecision over-counting (110.0000001 → 111).
    return int(np.ceil(round(n_star * safety_margin, 6)))


# ── 2. Dynamic Voronoi cell ownership with hysteresis ────────────────


@dataclass
class VoronoiHysteresisState:
    """Tracks which friendly currently owns each hostile, for hysteretic handover."""

    owner: dict[int, int] = field(default_factory=dict)  # hostile_id → friendly_idx

    def cell_owner(
        self,
        friendly_positions: np.ndarray,
        hostile_id: int,
        hostile_pos: np.ndarray,
        margin: float = 1.5,
    ) -> int:
        """Return the friendly_idx owning `hostile_pos` under hysteretic Voronoi.

        Standard Voronoi: argmin_i ||x − p_i||. Hysteresis: incumbent retains
        ownership unless a challenger is closer by more than `margin`.
        """
        dists = np.linalg.norm(friendly_positions - hostile_pos, axis=1)
        nearest = int(np.argmin(dists))
        incumbent = self.owner.get(hostile_id)
        if incumbent is None or incumbent >= friendly_positions.shape[0]:
            self.owner[hostile_id] = nearest
            return nearest
        if nearest == incumbent:
            return incumbent
        if dists[incumbent] - dists[nearest] > margin:
            # Challenger is meaningfully closer — hand over.
            self.owner[hostile_id] = nearest
            return nearest
        return incumbent

    def forget(self, hostile_id: int) -> None:
        self.owner.pop(hostile_id, None)


# ── 3. Fragmentation detection ───────────────────────────────────────


def algebraic_connectivity(adjacency: np.ndarray) -> float:
    """λ₂ of the (symmetric, normalised) graph Laplacian.

    λ₂ = 0 iff the graph has more than one connected component. Larger λ₂
    = more robust to single-link failures.
    """
    n = adjacency.shape[0]
    if n < 2:
        return 0.0
    A = (adjacency + adjacency.T) / 2.0
    A = (A > 0).astype(np.float64)
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    eigvals = np.linalg.eigvalsh(L)
    # Smallest is ~0 (constant eigenvector); λ₂ is the second-smallest.
    return float(eigvals[1])


def connected_components(adjacency: np.ndarray) -> list[set[int]]:
    """Partition the comm graph into connected components via BFS."""
    n = adjacency.shape[0]
    seen: set[int] = set()
    components: list[set[int]] = []
    for start in range(n):
        if start in seen:
            continue
        stack = [start]
        comp: set[int] = set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            nbrs = np.where(adjacency[u] > 0)[0]
            for v in nbrs:
                if v not in comp:
                    stack.append(int(v))
        seen |= comp
        components.append(comp)
    return components


# ── 4. End-to-end VAJRA assignment ───────────────────────────────────


@dataclass
class VajraParams:
    voronoi_margin: float = 1.5
    voronoi_bonus: float = 1.4    # multiplicative bonus for the cell owner
    tropical_beta: float = 8.0
    tropical_iters: int = 6
    jamming_factor: float = 0.0   # 0 = clear, 1 = fully jammed
    fragmentation_threshold: float = 1e-3  # λ₂ below this → fragmented mode


@dataclass
class VajraState:
    voronoi: VoronoiHysteresisState = field(default_factory=VoronoiHysteresisState)
    lambda2: float = 0.0
    n_components: int = 1
    handover_events: list[dict] = field(default_factory=list)


def vajra_assign(
    priority_matrix: np.ndarray,
    friendly_positions: np.ndarray,
    friendly_ids: list[int],
    hostile_positions: np.ndarray,
    hostile_ids: list[int],
    adjacency: np.ndarray,
    state: VajraState,
    params: VajraParams | None = None,
    already_assigned: dict[int, int] | None = None,
) -> dict[int, int]:
    """Resolve an assignment using Voronoi + tropical attention.

    Inputs
    ------
    priority_matrix : (N, M) — SHIELD's per-pair E_ij. Zero rows = bidder
                      kill-switched; zero columns = no eligible hostile.
    friendly_positions, friendly_ids : (N, 3), len-N
    hostile_positions,  hostile_ids  : (M, 3), len-M
    adjacency       : (N, N) comm graph for fragmentation handling.
    already_assigned : friendly_idx → hostile_id, pre-existing engagements.

    Returns {friendly_idx → hostile_id}. Composes with already_assigned —
    pre-existing engagements are respected; only newly-assignable
    friendlies and unclaimed hostiles compete.
    """
    p = params or VajraParams()
    n, m = priority_matrix.shape
    already_assigned = already_assigned or {}

    # Always refresh λ₂ + components — telemetry consumers (Gradio top-bar)
    # show this even when no TERMINAL hostiles are pending assignment.
    if adjacency.size > 0:
        state.lambda2 = algebraic_connectivity(adjacency)
        state.n_components = len(connected_components(adjacency))

    if n == 0 or m == 0:
        return {}

    # ── 4.1 Apply Voronoi-hysteresis ownership bonus ──
    C = priority_matrix.copy()
    incumbent_assignments: dict[int, int] = {}
    for j, h_id in enumerate(hostile_ids):
        owner_idx = state.voronoi.cell_owner(
            friendly_positions, h_id, hostile_positions[j], margin=p.voronoi_margin,
        )
        if 0 <= owner_idx < n:
            C[owner_idx, j] *= p.voronoi_bonus
        # Record handover for telemetry if it changed.
        prev_assigned = next(
            (fid for fid, hid in already_assigned.items() if hid == h_id), None,
        )
        if (
            prev_assigned is not None
            and owner_idx != prev_assigned
            and owner_idx >= 0
            and owner_idx < n
        ):
            state.handover_events.append({"hostile_id": h_id, "from": prev_assigned, "to": owner_idx})
        if prev_assigned is not None:
            incumbent_assignments[prev_assigned] = h_id

    # Don't re-bid on already-assigned hostiles or friendlies.
    used_friendlies = set(already_assigned.keys())
    used_hostiles = {hostile_ids.index(h) for h in already_assigned.values() if h in hostile_ids}
    for i in used_friendlies:
        if i < n:
            C[i, :] = 0.0
    for j in used_hostiles:
        C[:, j] = 0.0

    # ── 4.2 Per-component assignment ──
    components = connected_components(adjacency) if state.lambda2 < p.fragmentation_threshold else [set(range(n))]

    # ── 4.3 Tropical attention per component ──
    assignment: dict[int, int] = {}
    for comp in components:
        comp_idx = sorted(comp)
        if not comp_idx:
            continue
        sub_C = C[np.ix_(comp_idx, range(m))]
        # HSL-CC concurrency cap under jamming — drop a fraction of bids
        # uniformly to model the loss of simultaneous engagement bandwidth.
        if p.jamming_factor > 0:
            cap_scale = max(0.0, 1.0 - p.jamming_factor)
            sub_C = sub_C * cap_scale
        sub_assign = tropical_attention_assignment(
            sub_C, beta=p.tropical_beta, iters=p.tropical_iters,
        )
        for local_i, j in sub_assign.items():
            global_i = comp_idx[local_i]
            assignment[global_i] = hostile_ids[j]

    return assignment
