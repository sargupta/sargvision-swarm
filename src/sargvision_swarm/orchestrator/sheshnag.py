"""SHESHNAG — Swarm Hysteresis-Engineered Switching for Hypernormal-Network
Asymmetric Gain.

Offensive psyops + defensive tacit coordination layer. Four primitives composed:

  1. Couzin-Krause multistable swarm dynamics with engineered hysteresis —
     spoofed-beacon EW pulses nudge enemy swarm across the polarized →
     milling-vortex phase boundary and lock it there.
  2. Coupled fear-contagion SIR — panic state propagates over enemy comm
     graph; broadcast injection drives R₀ above the epidemic threshold.
  3. Pseudo-Quantum Correlated Coordination Equilibrium (PQ-CCE) — pre-shared
     correlated seeds let the friendly swarm coordinate WITHOUT comms
     (e.g., split attack arcs at deterministic phases under jamming).
  4. Distributed OODA Tempo Asymmetry (D-OTA) — Kuramoto-style phase
     oscillators; the friendly swarm operates at a tempo set so that the
     tempo gap Δτ = (τ_H − τ_F)/τ_H is positive even under jamming.

Composite objective: drive P(enemy → milling) up + Δτ up, minus EW emission
cost, plus tacit-coordination gain.

Engagement gate: SHESHNAG is a STRATEGIC weapon — sim integration must
require a BFT vote authorisation (K=7 SwarmRaft) before broadcasts fire.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ── 1. Couzin-Krause swarm-phase detector ────────────────────────────


def swarm_phase_metrics(velocities: np.ndarray, positions: np.ndarray) -> dict:
    """Return polarisation P and rotation R of a swarm.

    - P = |<v_i / |v_i|>|  ∈ [0, 1]  — 1 = perfectly polarised
    - R = |<(r_i × v_i) · ẑ>|  ∈ [0, 1]  — 1 = perfectly milling around centroid

    Classifies into POLARIZED (P > 0.6), MILLING (R > 0.55 and P < 0.5),
    or SWARM (neither). Phase labels follow Couzin et al.
    """
    if velocities.shape[0] == 0:
        return {"P": 0.0, "R": 0.0, "phase": "SWARM"}
    speeds = np.linalg.norm(velocities, axis=1)
    speeds_safe = np.where(speeds > 1e-6, speeds, 1.0)
    unit_v = velocities / speeds_safe[:, None]
    P = float(np.linalg.norm(unit_v.mean(axis=0)))

    centroid = positions.mean(axis=0)
    rel = positions - centroid
    rel_norm = np.linalg.norm(rel, axis=1, keepdims=True)
    rel_norm = np.where(rel_norm > 1e-6, rel_norm, 1.0)
    rel_hat = rel / rel_norm
    cross_z = rel_hat[:, 0] * unit_v[:, 1] - rel_hat[:, 1] * unit_v[:, 0]
    R = float(abs(cross_z.mean()))

    if P > 0.6 and R < 0.4:
        phase = "POLARIZED"
    elif R > 0.55 and P < 0.5:
        phase = "MILLING"
    else:
        phase = "SWARM"
    return {"P": P, "R": R, "phase": phase}


# ── 2. SIR fear-contagion ────────────────────────────────────────────


@dataclass
class SIRParams:
    beta: float = 0.55     # contact rate (per neighbour per second)
    gamma: float = 0.05    # recovery rate (per second)
    beacon_kick: float = 0.40  # per-broadcast injection ceiling
    neighbour_radius: float = 12.0


def sir_step(
    panic: np.ndarray,
    hostile_positions: np.ndarray,
    beacon_targets: np.ndarray | None,
    dt: float,
    params: SIRParams | None = None,
) -> np.ndarray:
    """Advance per-hostile panic state one tick.

        dI_j / dt = β Σ_{k ∈ N_j} I_k (1 − I_j) − γ I_j + η_j

    η_j is the spoofed-beacon injection — a Gaussian kick around any
    broadcast target. Returns the new panic vector.
    """
    p = params or SIRParams()
    n = hostile_positions.shape[0]
    if n == 0:
        return panic
    # Pairwise distances → adversary-side comm graph proxy.
    diffs = hostile_positions[:, None, :] - hostile_positions[None, :, :]
    dist = np.linalg.norm(diffs, axis=2)
    A = (dist < p.neighbour_radius).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    contact = A @ panic
    # SIR update
    eta = np.zeros(n)
    if beacon_targets is not None and beacon_targets.size > 0:
        for tgt in beacon_targets:
            d = np.linalg.norm(hostile_positions - tgt, axis=1)
            eta = eta + p.beacon_kick * np.exp(-(d ** 2) / (2 * 8.0 ** 2))
    dI = (p.beta * contact * (1.0 - panic) - p.gamma * panic + eta) * dt
    panic_next = np.clip(panic + dI, 0.0, 1.0)
    return panic_next


def basic_reproduction_number(adjacency: np.ndarray, params: SIRParams) -> float:
    """R₀ = β · <k> / γ — epidemic threshold; values > 1 imply spread."""
    if adjacency.size == 0:
        return 0.0
    mean_k = float(adjacency.sum(axis=1).mean())
    if params.gamma < 1e-9:
        return float("inf")
    return params.beta * mean_k / params.gamma


# ── 3. Pseudo-Quantum Correlated Coordination Equilibrium ────────────


@dataclass
class CorrelatedSeeds:
    """Pre-shared correlated random seeds for tacit, jam-resistant coordination.

    Each drone gets a deterministic share of a master seed. At decision time,
    drone i samples action a_i = π_i(state_i, seed_i, t). Because every drone
    derives its draw from the same master, the joint distribution attains
    correlations no independent policy can — a separable QCCE in the sense
    of classical correlated equilibrium without CHSH-violating entanglement.
    """

    master_seed: int = 0xDEAD_BEEF
    n_drones: int = 0
    _shares: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint64))

    def __post_init__(self) -> None:
        if self.n_drones > 0:
            self._shares = self._derive_shares(self.master_seed, self.n_drones)

    @staticmethod
    def _derive_shares(master: int, n: int) -> np.ndarray:
        rng = np.random.default_rng(master)
        return rng.integers(0, 2**32, size=n, dtype=np.uint64)

    def draw_action(self, drone_idx: int, tick: int, n_actions: int) -> int:
        """Return action ∈ [0, n_actions) — deterministic given (seed, tick)."""
        if drone_idx < 0 or drone_idx >= self._shares.size:
            return 0
        local_seed = int(self._shares[drone_idx]) ^ (tick * 0x9E37_79B9)
        rng = np.random.default_rng(local_seed)
        return int(rng.integers(0, n_actions))

    def correlated_angles(self, n_drones: int, tick: int) -> np.ndarray:
        """Coordinated attack-arc angles for tacit fan-out under jamming.

        Every drone independently derives the SAME tick-keyed rotation, then
        adds its share-fixed offset → angles tile the circle deterministically.
        """
        # Tick-keyed common rotation (every drone agrees on this offset)
        common = np.random.default_rng(int(tick * 0x9E37_79B9)).uniform(0, 2 * np.pi)
        base = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
        return (base + common) % (2 * np.pi)


def coordination_gain(actions: np.ndarray, payoff_independent: float) -> float:
    """Quick test fixture: PQ-CCE coordination should EXCEED indep baseline.

    Toy payoff = 1 if all actions equal, else 0.  An independent random
    policy gives 1/k. A correlated policy fixed at the master draw gives 1.
    Returns the empirical correlated payoff minus the independent baseline.
    """
    coord = float((actions == actions[0]).all())
    return coord - payoff_independent


# ── 4. D-OTA Kuramoto tempo asymmetry ────────────────────────────────


def kuramoto_step(
    phases: np.ndarray,
    natural_freq: np.ndarray,
    adjacency: np.ndarray,
    coupling: float,
    dt: float,
) -> np.ndarray:
    """One Euler step of Kuramoto oscillator dynamics."""
    n = phases.shape[0]
    if n == 0:
        return phases
    diff = phases[None, :] - phases[:, None]   # (n, n) of φ_j - φ_i
    interaction = (adjacency * np.sin(diff)).sum(axis=1)
    return (phases + dt * (natural_freq + coupling * interaction)) % (2 * np.pi)


def tempo_gap(
    friendly_periods: np.ndarray,
    hostile_periods: np.ndarray,
) -> float:
    """Δτ = (τ_H − τ_F) / τ_H using SWARM tempo = max τ_i (slowest link binds)."""
    if friendly_periods.size == 0 or hostile_periods.size == 0:
        return 0.0
    tau_F = float(friendly_periods.max())
    tau_H = float(hostile_periods.max())
    if tau_H < 1e-9:
        return 0.0
    return (tau_H - tau_F) / tau_H


# ── 5. End-to-end SHESHNAG state + tick ──────────────────────────────


@dataclass
class SheshnagParams:
    sir: SIRParams = field(default_factory=SIRParams)
    panic_milling_threshold: float = 0.45
    broadcasts_per_tick: int = 3
    ew_emission_cost: float = 0.02   # per-broadcast cost in composite objective
    coupling: float = 0.6
    friendly_period_target: float = 0.40   # τ_F, seconds
    hostile_period_assumed: float = 0.95   # τ_H, conservative estimate


@dataclass
class SheshnagState:
    authorized: bool = False           # BFT-gated psyops emission permission
    broadcast_targets: list[np.ndarray] = field(default_factory=list)
    broadcasts_emitted: int = 0
    last_phase: dict = field(default_factory=lambda: {"P": 0.0, "R": 0.0, "phase": "SWARM"})
    mean_panic: float = 0.0
    fraction_panicked: float = 0.0
    composite_value: float = 0.0


def select_broadcast_targets(
    hostile_positions: np.ndarray,
    panic: np.ndarray,
    n_targets: int,
) -> list[np.ndarray]:
    """Pick the n_targets hostiles whose panic injection gives best leverage.

    Heuristic: cluster centroids of the LEAST-panicked dense neighbourhoods —
    those are the SIR seeds that haven't fired yet.
    """
    n = hostile_positions.shape[0]
    if n == 0 or n_targets == 0:
        return []
    # Score = (1 − panic) · neighbour density.
    diffs = hostile_positions[:, None, :] - hostile_positions[None, :, :]
    dist = np.linalg.norm(diffs, axis=2)
    density = (dist < 15.0).sum(axis=1)
    score = (1.0 - panic) * density
    top = np.argsort(score)[::-1][:n_targets]
    return [hostile_positions[int(i)].copy() for i in top]


def sheshnag_tick(
    hostile_positions: np.ndarray,
    hostile_velocities: np.ndarray,
    panic: np.ndarray,
    dt: float,
    state: SheshnagState,
    params: SheshnagParams | None = None,
) -> np.ndarray:
    """One SHESHNAG tick. Updates state in place; returns new panic vector.

    Gated on `state.authorized` — without BFT authorisation the broadcasts
    don't fire and only the SIR decay + phase-classification logic runs.
    """
    p = params or SheshnagParams()
    if hostile_positions.shape[0] == 0:
        state.last_phase = {"P": 0.0, "R": 0.0, "phase": "SWARM"}
        state.mean_panic = 0.0
        state.fraction_panicked = 0.0
        return panic
    if state.authorized:
        state.broadcast_targets = select_broadcast_targets(
            hostile_positions, panic, p.broadcasts_per_tick,
        )
        state.broadcasts_emitted += len(state.broadcast_targets)
        beacons = np.array(state.broadcast_targets) if state.broadcast_targets else None
    else:
        state.broadcast_targets = []
        beacons = None
    panic_next = sir_step(panic, hostile_positions, beacons, dt, p.sir)
    state.mean_panic = float(panic_next.mean())
    state.fraction_panicked = float((panic_next > p.panic_milling_threshold).mean())
    state.last_phase = swarm_phase_metrics(hostile_velocities, hostile_positions)
    # Composite scalar — useful for telemetry, not directly used as gradient.
    delta_tau = tempo_gap(
        np.array([p.friendly_period_target]),
        np.array([p.hostile_period_assumed]),
    )
    p_milling = 1.0 if state.last_phase["phase"] == "MILLING" else state.fraction_panicked
    state.composite_value = (
        p_milling
        + 0.3 * delta_tau
        - p.ew_emission_cost * len(state.broadcast_targets)
    )
    return panic_next
