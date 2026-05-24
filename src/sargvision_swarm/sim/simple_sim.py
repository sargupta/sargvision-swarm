"""Light double-integrator drone sim. Pure NumPy. macOS-friendly.

For higher-fidelity quadrotor dynamics + collisions, use the optional
``gym-pybullet-drones`` wrapper (`pip install '.[sim]'`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sargvision_swarm.core.state import SwarmState


@dataclass
class SimConfig:
    dt: float = 0.05  # 20 Hz step
    max_accel: float = 6.0  # m/s² — quadrotor-ish
    velocity_tau: float = 0.25  # first-order vel tracking time constant
    wind_std: float = 0.0  # m/s noise on velocity each step
    world_bounds: tuple[float, float, float] = (60.0, 60.0, 30.0)
    z_floor: float = 0.5  # don't dip below this


class SimpleSim:
    """First-order velocity tracker + cubic-bounds world.

    State held externally in `SwarmState`. Each step takes a (N, 3) velocity
    command, integrates first-order toward it, clips accel + speed, and
    advances positions.
    """

    def __init__(self, config: SimConfig | None = None, seed: int | None = None) -> None:
        self.cfg = config or SimConfig()
        self._rng = np.random.default_rng(seed)

    def step(self, swarm: SwarmState, velocity_cmd: np.ndarray) -> None:
        """Advance the swarm one timestep."""
        if velocity_cmd.shape != (swarm.n, 3):
            raise ValueError(f"velocity_cmd shape {velocity_cmd.shape} != ({swarm.n}, 3)")

        dt = self.cfg.dt
        current_vel = swarm.velocities

        # First-order tracking: a = (v_cmd - v) / tau
        accel = (velocity_cmd - current_vel) / self.cfg.velocity_tau
        accel_norm = np.linalg.norm(accel, axis=1, keepdims=True)
        too_high = accel_norm > self.cfg.max_accel
        accel = np.where(
            too_high,
            accel / (accel_norm + 1e-6) * self.cfg.max_accel,
            accel,
        )

        new_vel = current_vel + accel * dt
        if self.cfg.wind_std > 0:
            new_vel += self._rng.normal(scale=self.cfg.wind_std, size=new_vel.shape)

        # Apply
        swarm.apply_velocities(new_vel, dt)

        # Bounds clamp (soft floor + hard walls)
        bx, by, bz = self.cfg.world_bounds
        for drone in swarm.drones:
            drone.pos[0] = float(np.clip(drone.pos[0], -bx / 2, bx / 2))
            drone.pos[1] = float(np.clip(drone.pos[1], -by / 2, by / 2))
            drone.pos[2] = float(np.clip(drone.pos[2], self.cfg.z_floor, bz))
