"""Reflex composition — pick algorithm by scenario, wrap in BVC safety filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sargvision_swarm.core.boids import BoidsParams, boids_velocity
from sargvision_swarm.core.bvc import bvc_safe_velocity
from sargvision_swarm.core.olfati_saber import (
    OlfatiSaberParams,
    olfati_saber_velocity,
)


@dataclass
class ReflexParams:
    """Composite parameters for the full reflex stack."""

    boids: BoidsParams | None = None
    olfati_saber: OlfatiSaberParams | None = None
    bvc_safety_radius: float = 0.8
    bvc_dt: float = 0.1
    bvc_influence: float = 6.0

    def __post_init__(self) -> None:
        if self.boids is None:
            self.boids = BoidsParams()
        if self.olfati_saber is None:
            self.olfati_saber = OlfatiSaberParams()


def compose_reflex(
    positions: np.ndarray,
    velocities: np.ndarray,
    algorithm: str = "boids",
    goal_pos: np.ndarray | None = None,
    goal_vel: np.ndarray | None = None,
    params: ReflexParams | None = None,
) -> np.ndarray:
    """Run the chosen flocking algorithm, then wrap with BVC collision filter.

    algorithm: 'boids' | 'olfati_saber'
    """
    if params is None:
        params = ReflexParams()

    if algorithm == "boids":
        desired = boids_velocity(positions, velocities, params.boids)
    elif algorithm == "olfati_saber":
        if goal_pos is None:
            goal_pos = positions.mean(axis=0)
        desired = olfati_saber_velocity(
            positions, velocities, goal_pos, goal_vel, params.olfati_saber
        )
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")

    safe = bvc_safe_velocity(
        positions,
        desired,
        safety_radius=params.bvc_safety_radius,
        dt=params.bvc_dt,
        influence_radius=params.bvc_influence,
    )
    return safe
