"""SHESHNAG — Couzin-Krause phase, SIR contagion, correlated seeds, Kuramoto."""

from __future__ import annotations

import numpy as np

from sargvision_swarm.orchestrator.sheshnag import (
    CorrelatedSeeds,
    SheshnagParams,
    SheshnagState,
    SIRParams,
    basic_reproduction_number,
    kuramoto_step,
    select_broadcast_targets,
    sheshnag_tick,
    sir_step,
    swarm_phase_metrics,
    tempo_gap,
)

# ── 1. Phase metrics ──────────────────────────────────────────────────


def test_phase_polarized_when_all_aligned():
    n = 12
    pos = np.random.default_rng(0).uniform(-5, 5, size=(n, 3))
    vel = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    m = swarm_phase_metrics(vel, pos)
    assert m["phase"] == "POLARIZED"
    assert m["P"] > 0.9


def test_phase_milling_when_tangential():
    n = 16
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.stack([5 * np.cos(angles), 5 * np.sin(angles), np.zeros(n)], axis=1)
    # velocity perpendicular to radius (counter-clockwise circulation)
    vel = np.stack([-np.sin(angles), np.cos(angles), np.zeros(n)], axis=1)
    m = swarm_phase_metrics(vel, pos)
    assert m["phase"] == "MILLING"
    assert m["R"] > 0.9


# ── 2. SIR contagion ──────────────────────────────────────────────────


def test_sir_spreads_from_seed_under_high_beta():
    n = 12
    pos = np.array([[i * 3.0, 0.0, 0.0] for i in range(n)])  # tight line, dense graph
    panic = np.zeros(n)
    panic[0] = 0.9  # seed
    params = SIRParams(beta=0.8, gamma=0.02, neighbour_radius=8.0)
    for _ in range(200):
        panic = sir_step(panic, pos, beacon_targets=None, dt=0.1, params=params)
    assert panic.mean() > 0.5, f"contagion failed to spread: mean={panic.mean():.2f}"


def test_sir_decays_under_low_r0():
    n = 6
    pos = np.array([[i * 20.0, 0.0, 0.0] for i in range(n)])  # sparse, low contact
    panic = np.full(n, 0.5)
    params = SIRParams(beta=0.01, gamma=0.5, neighbour_radius=5.0)
    for _ in range(200):
        panic = sir_step(panic, pos, beacon_targets=None, dt=0.1, params=params)
    assert panic.mean() < 0.1


def test_beacon_kick_raises_panic():
    """Beacon is area-effect EW — its kick should drop monotonically with distance,
    not be a delta function."""
    n = 5
    pos = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0], [40.0, 0.0, 0.0]]
    )
    panic = np.zeros(n)
    params = SIRParams(beta=0.0, gamma=0.0, beacon_kick=0.6, neighbour_radius=4.0)
    beacons = np.array([[20.0, 0.0, 0.0]])  # broadcast right at drone 2
    panic_next = sir_step(panic, pos, beacons, dt=1.0, params=params)
    # Hostile at the broadcast position should be panicked.
    assert panic_next[2] > 0.5
    # Far hostile should be a fraction of that.
    assert panic_next[0] < 0.1
    # Monotonic falloff with distance.
    assert panic_next[2] > panic_next[1] > panic_next[0]
    assert panic_next[2] > panic_next[3] > panic_next[4]


def test_r0_high_when_dense_low_gamma():
    n = 10
    pos = np.array([[i * 2.0, 0.0, 0.0] for i in range(n)])
    diff = pos[:, None, :] - pos[None, :, :]
    A = (np.linalg.norm(diff, axis=2) < 5.0).astype(float)
    np.fill_diagonal(A, 0)
    r0 = basic_reproduction_number(A, SIRParams(beta=0.5, gamma=0.05))
    assert r0 > 1.0


# ── 3. Correlated seeds (PQ-CCE) ──────────────────────────────────────


def test_correlated_seeds_are_deterministic_per_drone_tick():
    seeds = CorrelatedSeeds(master_seed=42, n_drones=8)
    a1 = seeds.draw_action(drone_idx=3, tick=10, n_actions=4)
    a2 = seeds.draw_action(drone_idx=3, tick=10, n_actions=4)
    assert a1 == a2


def test_correlated_angles_tile_circle():
    seeds = CorrelatedSeeds(master_seed=42, n_drones=8)
    angles = seeds.correlated_angles(n_drones=8, tick=5)
    # Adjacent gaps after sorting should all be close to 2π/8.
    sorted_a = np.sort(angles)
    gaps = np.diff(sorted_a)
    expected = 2 * np.pi / 8
    assert (np.abs(gaps - expected) < 1e-6).all()


def test_correlated_seeds_different_drones_get_different_actions():
    seeds = CorrelatedSeeds(master_seed=42, n_drones=16)
    actions = [seeds.draw_action(i, tick=1, n_actions=8) for i in range(16)]
    # At least 4 distinct actions across 16 drones — extreme collision is suspicious.
    assert len(set(actions)) >= 4


# ── 4. Kuramoto tempo gap ─────────────────────────────────────────────


def test_kuramoto_synchronises_under_high_coupling():
    n = 8
    rng = np.random.default_rng(0)
    phases = rng.uniform(0, 2 * np.pi, size=n)
    omega = np.full(n, 1.0)
    A = np.ones((n, n)) - np.eye(n)
    for _ in range(800):
        phases = kuramoto_step(phases, omega, A, coupling=2.0, dt=0.02)
    # Order parameter close to 1 = synchronised.
    R = np.abs(np.exp(1j * phases).mean())
    assert R > 0.95


def test_tempo_gap_positive_when_friendly_faster():
    fp = np.array([0.40, 0.42, 0.41])
    hp = np.array([0.95, 1.0, 0.90])
    dt = tempo_gap(fp, hp)
    assert 0.5 < dt < 0.7


# ── 5. End-to-end sheshnag_tick ───────────────────────────────────────


def test_sheshnag_tick_only_broadcasts_when_authorized():
    n = 6
    pos = np.array([[i * 2.0, 0.0, 0.0] for i in range(n)])
    vel = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    panic = np.zeros(n)
    state = SheshnagState(authorized=False)
    p1 = sheshnag_tick(pos, vel, panic, dt=0.1, state=state)
    assert state.broadcasts_emitted == 0
    assert p1.max() < 0.01

    state.authorized = True
    for _ in range(15):
        p1 = sheshnag_tick(pos, vel, p1, dt=0.1, state=state)
    assert state.broadcasts_emitted > 0
    assert p1.max() > 0.1


def test_select_broadcast_targets_returns_unpanicked_dense_seeds():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],  # dense cluster
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [50.0, 0.0, 0.0],  # isolated
        ]
    )
    panic = np.array([0.0, 0.0, 0.0, 0.0])
    targets = select_broadcast_targets(pos, panic, n_targets=1)
    # Best target is in the dense cluster, not the isolated drone.
    chosen = targets[0]
    assert np.linalg.norm(chosen - np.array([50.0, 0.0, 0.0])) > 10.0
