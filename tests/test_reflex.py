"""Reflex layer behavior tests."""

import numpy as np

from sargvision_swarm.core.boids import BoidsParams, boids_velocity
from sargvision_swarm.core.bvc import bvc_safe_velocity
from sargvision_swarm.core.olfati_saber import OlfatiSaberParams, olfati_saber_velocity
from sargvision_swarm.core.reflex import ReflexParams, compose_reflex


def test_boids_speed_clipped_to_max():
    positions = np.array([[0.0, 0.0, 5.0], [3.0, 0.0, 5.0], [6.0, 0.0, 5.0]])
    velocities = np.array([[10.0, 0.0, 0.0]] * 3)
    params = BoidsParams(max_speed=4.0)
    out = boids_velocity(positions, velocities, params)
    speeds = np.linalg.norm(out, axis=1)
    assert (speeds <= params.max_speed + 1e-6).all()


def test_boids_separation_pushes_close_drones_apart():
    # Two drones nearly co-located along x — should be pushed apart in x.
    positions = np.array([[0.0, 0.0, 5.0], [0.5, 0.0, 5.0]])
    velocities = np.zeros((2, 3))
    params = BoidsParams(separation_radius=2.0, perception_radius=4.0)
    out = boids_velocity(positions, velocities, params)
    # Drone 0 should be pushed in -x, drone 1 in +x.
    assert out[0, 0] < 0
    assert out[1, 0] > 0


def test_bvc_does_not_modify_safe_velocity():
    positions = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0]])  # far apart
    desired = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])  # moving toward each other slowly
    safe = bvc_safe_velocity(positions, desired, safety_radius=0.5, dt=0.1)
    # 10 m apart, moving at 1 m/s for 0.1 s → 0.1 m closer. Safe.
    np.testing.assert_allclose(safe, desired, atol=1e-6)


def test_bvc_constrains_colliding_motion():
    # Drones 2 m apart heading straight at each other.
    positions = np.array([[0.0, 0.0, 5.0], [2.0, 0.0, 5.0]])
    desired = np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    safe = bvc_safe_velocity(positions, desired, safety_radius=0.8, dt=0.2)
    # Drone 0 should slow down (less +x), drone 1 less -x.
    assert safe[0, 0] < desired[0, 0]
    assert safe[1, 0] > desired[1, 0]


def test_olfati_saber_pulls_toward_goal():
    positions = np.array([[0.0, 0.0, 5.0]])
    velocities = np.zeros((1, 3))
    goal = np.array([10.0, 0.0, 5.0])
    out = olfati_saber_velocity(positions, velocities, goal, params=OlfatiSaberParams())
    # Should have +x component (toward goal).
    assert out[0, 0] > 0


def test_compose_reflex_runs_all_scenarios():
    positions = np.array([[0.0, 0.0, 5.0], [2.0, 0.0, 5.0], [4.0, 0.0, 5.0]])
    velocities = np.zeros((3, 3))
    for algo in ("boids", "olfati_saber"):
        out = compose_reflex(
            positions, velocities, algorithm=algo, goal_pos=np.array([10.0, 0.0, 5.0])
        )
        assert out.shape == (3, 3)
        assert np.isfinite(out).all()


def test_compose_reflex_scales_to_50_drones():
    """Smoke test for reasonable swarm size."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(-10, 10, size=(50, 3))
    positions[:, 2] = np.clip(positions[:, 2], 1, 10)
    velocities = rng.normal(scale=0.1, size=(50, 3))
    out = compose_reflex(positions, velocities, algorithm="boids")
    assert out.shape == (50, 3)
    assert np.isfinite(out).all()
