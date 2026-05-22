"""Reynolds Boids (1987) — separation, alignment, cohesion. Vectorized.

Reference: see ~/Documents/AI_Workspace/drone_swarm_research/07_classical_swarm.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BoidsParams:
    """Boids tuning parameters."""

    perception_radius: float = 8.0      # meters
    separation_radius: float = 2.0      # too-close threshold
    weight_separation: float = 1.6
    weight_alignment: float = 1.0
    weight_cohesion: float = 0.8
    max_speed: float = 4.0              # m/s (Vásárhelyi 2018 used 8 m/s)


def _pairwise_offsets(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (offsets, distances). offsets[i, j] = pos[j] - pos[i]."""
    offsets = positions[None, :, :] - positions[:, None, :]  # (N, N, 3)
    distances = np.linalg.norm(offsets, axis=-1)             # (N, N)
    return offsets, distances


def boids_velocity(
    positions: np.ndarray,
    velocities: np.ndarray,
    params: BoidsParams | None = None,
) -> np.ndarray:
    """Compute Boids steering vector for each drone.

    Returns (N, 3) desired velocities clipped to max_speed.
    """
    if params is None:
        params = BoidsParams()
    n = positions.shape[0]
    if n == 0:
        return np.zeros((0, 3))
    if n == 1:
        return velocities.copy()

    offsets, distances = _pairwise_offsets(positions)
    # Mask self
    self_mask = np.eye(n, dtype=bool)
    distances_masked = np.where(self_mask, np.inf, distances)

    neighbors = distances_masked < params.perception_radius        # (N, N) bool
    too_close = distances_masked < params.separation_radius

    # --- Separation: push away from too-close neighbors, weighted by inverse distance.
    eps = 1e-6
    push = -offsets / (distances_masked[..., None] + eps)          # unit vec away
    push_weight = too_close.astype(np.float64)
    sep_count = push_weight.sum(axis=1, keepdims=True)
    sep_count = np.where(sep_count == 0, 1.0, sep_count)
    separation = (push * push_weight[..., None]).sum(axis=1) / sep_count

    # --- Alignment: match velocity of perceived neighbors.
    nbr_count = neighbors.sum(axis=1, keepdims=True).astype(np.float64)
    nbr_count = np.where(nbr_count == 0, 1.0, nbr_count)
    avg_vel = (velocities[None, :, :] * neighbors[..., None]).sum(axis=1) / nbr_count
    alignment = avg_vel - velocities

    # --- Cohesion: steer toward centroid of perceived neighbors.
    avg_pos = (positions[None, :, :] * neighbors[..., None]).sum(axis=1) / nbr_count
    cohesion = avg_pos - positions

    steering = (
        params.weight_separation * separation
        + params.weight_alignment * alignment
        + params.weight_cohesion * cohesion
    )

    desired = velocities + steering
    speeds = np.linalg.norm(desired, axis=1, keepdims=True)
    too_fast = speeds > params.max_speed
    desired = np.where(
        too_fast,
        desired / (speeds + eps) * params.max_speed,
        desired,
    )
    return desired
