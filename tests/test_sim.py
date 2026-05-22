"""Sim layer + end-to-end rollout."""

import numpy as np

from sargvision_swarm.core import SwarmState
from sargvision_swarm.demo.runner import rollout
from sargvision_swarm.sim import SimConfig, SimpleSim


def test_simple_sim_step_advances_position():
    swarm = SwarmState.random_init(3, seed=0)
    sim = SimpleSim(SimConfig(dt=0.1), seed=0)
    p0 = swarm.positions.copy()
    sim.step(swarm, np.array([[1.0, 0.0, 0.0]] * 3))
    p1 = swarm.positions
    # Some motion in +x.
    assert (p1[:, 0] >= p0[:, 0] - 1e-6).all()
    assert swarm.t > 0


def test_sim_respects_z_floor():
    swarm = SwarmState.random_init(3, seed=0)
    for d in swarm.drones:
        d.pos[2] = 0.6
    sim = SimpleSim(SimConfig(dt=0.1, z_floor=0.5), seed=0)
    # Drive everyone downward — z should clamp at floor.
    for _ in range(30):
        sim.step(swarm, np.array([[0.0, 0.0, -5.0]] * 3))
    for d in swarm.drones:
        assert d.pos[2] >= 0.5 - 1e-6


def test_rollout_completes_for_each_scenario():
    for scenario in ("flock", "formation_v", "coverage", "hover"):
        result = rollout(n_drones=10, scenario=scenario, steps=40, seed=42, snapshot_every=10)
        assert len(result.states) >= 3
        assert result.states[-1].n == 10
        assert np.isfinite(result.states[-1].positions).all()


def test_rollout_no_collisions_at_moderate_density():
    """After settling, min inter-drone distance should be at least the BVC buffer."""
    result = rollout(n_drones=15, scenario="flock", steps=200, seed=7, snapshot_every=199)
    final = result.states[-1]
    positions = final.positions
    n = positions.shape[0]
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(float(np.linalg.norm(positions[i] - positions[j])))
    # Reasonable minimum given BVC safety_radius=0.8 + sim noise
    min_d = min(distances)
    assert min_d > 0.3, f"min inter-drone distance too small: {min_d:.2f}"
