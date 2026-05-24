"""Buffered Voronoi Cells (BVC) — collision-avoidance velocity projection.

Each drone broadcasts position. For drone i with desired velocity v_d, find the
nearest constraint plane from any neighbor and project the predicted next-step
position back inside drone i's buffered Voronoi cell.

Reference: Zhou et al., "Fast, On-line Collision Avoidance for Dynamic Vehicles
Using Buffered Voronoi Cells", IEEE RAL 2017.
"""

from __future__ import annotations

import numpy as np


def bvc_safe_velocity(
    positions: np.ndarray,
    desired_velocities: np.ndarray,
    safety_radius: float = 0.8,
    dt: float = 0.1,
    influence_radius: float = 6.0,
) -> np.ndarray:
    """Project desired velocity into each drone's buffered Voronoi cell.

    positions: (N, 3)
    desired_velocities: (N, 3)
    safety_radius: shift each half-plane toward self by this much (meters).
    dt: prediction horizon (seconds).
    influence_radius: only consider neighbors within this range.

    Returns (N, 3) collision-safe velocity.
    """
    n = positions.shape[0]
    if n <= 1:
        return desired_velocities.copy()

    predicted = positions + desired_velocities * dt  # (N, 3)
    safe_positions = predicted.copy()

    # Pairwise offsets to current positions
    offsets = positions[None, :, :] - positions[:, None, :]  # (N, N, 3)
    distances = np.linalg.norm(offsets, axis=-1)  # (N, N)
    self_mask = np.eye(n, dtype=bool)
    near_mask = (distances < influence_radius) & ~self_mask  # (N, N)

    # For each drone i, iterate over near neighbors j and check the half-plane:
    # plane normal n_ij = (pos_j - pos_i) / ||...||
    # midpoint with buffer m_ij = (pos_i + pos_j)/2 - n_ij * safety_radius
    # Constraint: dot(n_ij, predicted_i - m_ij) <= 0   (i must stay on its side)
    eps = 1e-6
    for i in range(n):
        for j in np.where(near_mask[i])[0]:
            n_ij = offsets[i, j] / (distances[i, j] + eps)
            midpoint = (positions[i] + positions[j]) / 2.0 - n_ij * safety_radius
            v = safe_positions[i] - midpoint
            overshoot = float(n_ij @ v)
            if overshoot > 0.0:
                # project back to plane
                safe_positions[i] = safe_positions[i] - overshoot * n_ij

    safe_vel = (safe_positions - positions) / dt
    return safe_vel
