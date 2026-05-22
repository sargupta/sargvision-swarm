"""Shared data types — drone + swarm state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Role(str, Enum):
    WORKER = "worker"
    SCOUT = "scout"
    RELAY = "relay"
    LEADER = "leader"


@dataclass
class DroneState:
    """Single drone runtime state."""

    id: int
    pos: np.ndarray  # (3,) — x, y, z in meters
    vel: np.ndarray  # (3,) — m/s
    role: Role = Role.WORKER
    battery: float = 1.0  # 0..1
    healthy: bool = True

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.vel = np.asarray(self.vel, dtype=np.float64)
        if self.pos.shape != (3,):
            raise ValueError(f"pos must be shape (3,), got {self.pos.shape}")
        if self.vel.shape != (3,):
            raise ValueError(f"vel must be shape (3,), got {self.vel.shape}")


@dataclass
class SwarmState:
    """Whole-swarm runtime state."""

    drones: list[DroneState] = field(default_factory=list)
    t: float = 0.0

    @property
    def n(self) -> int:
        return len(self.drones)

    @property
    def positions(self) -> np.ndarray:
        """(N, 3) position matrix."""
        if not self.drones:
            return np.zeros((0, 3))
        return np.vstack([d.pos for d in self.drones])

    @property
    def velocities(self) -> np.ndarray:
        """(N, 3) velocity matrix."""
        if not self.drones:
            return np.zeros((0, 3))
        return np.vstack([d.vel for d in self.drones])

    @property
    def healthy_mask(self) -> np.ndarray:
        """(N,) bool — True if drone is healthy."""
        return np.array([d.healthy for d in self.drones], dtype=bool)

    def apply_velocities(self, velocities: np.ndarray, dt: float) -> None:
        """Integrate positions forward using given velocities, then store them."""
        if velocities.shape != (self.n, 3):
            raise ValueError(
                f"velocities shape {velocities.shape} != ({self.n}, 3)"
            )
        for i, drone in enumerate(self.drones):
            if drone.healthy:
                drone.vel = velocities[i].copy()
                drone.pos = drone.pos + drone.vel * dt
        self.t += dt

    @classmethod
    def random_init(
        cls,
        n: int,
        spawn_box: tuple[float, float, float] = (20.0, 20.0, 10.0),
        z_offset: float = 5.0,
        seed: int | None = None,
    ) -> SwarmState:
        """Create swarm with N drones spawned in a box, near-zero velocity."""
        rng = np.random.default_rng(seed)
        bx, by, bz = spawn_box
        positions = rng.uniform(
            low=[-bx / 2, -by / 2, z_offset - bz / 2],
            high=[bx / 2, by / 2, z_offset + bz / 2],
            size=(n, 3),
        )
        velocities = rng.normal(scale=0.1, size=(n, 3))
        drones = [
            DroneState(id=i, pos=positions[i], vel=velocities[i])
            for i in range(n)
        ]
        return cls(drones=drones)
