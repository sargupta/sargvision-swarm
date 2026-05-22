"""SwarmState basic invariants."""

import numpy as np
import pytest

from sargvision_swarm.core.state import DroneState, Role, SwarmState


def test_drone_state_validates_shape():
    with pytest.raises(ValueError):
        DroneState(id=0, pos=np.array([0.0, 0.0]), vel=np.zeros(3))


def test_swarm_random_init_has_correct_shape():
    s = SwarmState.random_init(20, seed=0)
    assert s.n == 20
    assert s.positions.shape == (20, 3)
    assert s.velocities.shape == (20, 3)
    assert s.healthy_mask.all()


def test_swarm_apply_velocities_advances_time():
    s = SwarmState.random_init(5, seed=0)
    t0 = s.t
    s.apply_velocities(np.ones((5, 3)), dt=0.1)
    assert s.t == pytest.approx(t0 + 0.1)
    assert s.drones[0].vel[0] == pytest.approx(1.0)


def test_swarm_apply_velocities_rejects_wrong_shape():
    s = SwarmState.random_init(5, seed=0)
    with pytest.raises(ValueError):
        s.apply_velocities(np.ones((3, 3)), dt=0.1)


def test_role_enum():
    s = SwarmState.random_init(3, seed=0)
    assert all(d.role == Role.WORKER for d in s.drones)
