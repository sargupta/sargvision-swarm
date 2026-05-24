"""Olfati-Saber Algorithm 3 — α-agents flock toward γ-agent (goal), avoid β-obstacles.

Reference: Olfati-Saber 2006, "Flocking for Multi-Agent Dynamic Systems".
Simplified implementation suitable for drone swarm reflex layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OlfatiSaberParams:
    interaction_range: float = 6.0
    desired_spacing: float = 2.5
    epsilon: float = 0.1  # σ-norm parameter
    c1_alpha: float = 1.0  # cohesion / spacing gain
    c2_alpha: float = 0.8  # velocity-matching gain
    c1_gamma: float = 0.6  # goal-seeking gain (position)
    c2_gamma: float = 0.4  # goal-tracking gain (velocity)
    h_bump: float = 0.2  # bump function transition
    max_speed: float = 5.0


def _sigma_norm(z: np.ndarray, eps: float) -> np.ndarray:
    """σ-norm: smooth differentiable proxy for Euclidean norm.

    Accepts either a vector field (..., 3) giving σ-norm of each vector, OR
    a 1-D array of pre-computed scalars (interpreted as Euclidean magnitudes).
    """
    z = np.asarray(z)
    if z.shape[-1] == 3 and z.ndim >= 2:
        sq = np.sum(z * z, axis=-1)
    else:
        sq = z * z
    return (np.sqrt(1.0 + eps * sq) - 1.0) / eps


def _sigma_norm_scalar(r: float, eps: float) -> float:
    """σ-norm of a scalar magnitude."""
    return (float(np.sqrt(1.0 + eps * r * r)) - 1.0) / eps


def _sigma_gradient(z: np.ndarray, eps: float) -> np.ndarray:
    """∇σ-norm. z shape (..., 3). Returns (..., 3)."""
    denom = np.sqrt(1.0 + eps * np.sum(z * z, axis=-1, keepdims=True))
    return z / denom


def _bump(z: np.ndarray, h: float) -> np.ndarray:
    """Bump function — smooth indicator over [0, 1]."""
    out = np.zeros_like(z)
    mask1 = (z >= 0.0) & (z < h)
    mask2 = (z >= h) & (z <= 1.0)
    out[mask1] = 1.0
    cos_arg = np.pi * (z[mask2] - h) / (1.0 - h)
    out[mask2] = 0.5 * (1.0 + np.cos(cos_arg))
    return out


def _phi_alpha(z: np.ndarray, r_alpha: float, d_alpha: float, h: float) -> np.ndarray:
    """Action function for α-α interaction. Pulls when far, pushes when close."""
    normalised = z / r_alpha
    rho = _bump(normalised, h)
    # Smooth attractive-repulsive (paper's φ_α). We use a saturating φ for stability.
    phi = 0.5 * ((z - d_alpha) / np.sqrt(1.0 + (z - d_alpha) ** 2))
    return rho * phi


def olfati_saber_velocity(
    positions: np.ndarray,
    velocities: np.ndarray,
    goal_pos: np.ndarray,
    goal_vel: np.ndarray | None = None,
    params: OlfatiSaberParams | None = None,
) -> np.ndarray:
    """Compute Algorithm 3 steering for the swarm.

    positions: (N, 3)
    velocities: (N, 3)
    goal_pos: (3,) — γ-agent position
    goal_vel: (3,) or None — γ-agent velocity (default zeros)

    Returns (N, 3) desired velocity.
    """
    if params is None:
        params = OlfatiSaberParams()
    n = positions.shape[0]
    if n == 0:
        return np.zeros((0, 3))
    if goal_vel is None:
        goal_vel = np.zeros(3)

    eps = params.epsilon
    r = params.interaction_range
    d = params.desired_spacing
    r_alpha = _sigma_norm_scalar(r, eps)
    d_alpha = _sigma_norm_scalar(d, eps)

    # Pairwise offsets
    offsets = positions[None, :, :] - positions[:, None, :]  # (N, N, 3)
    norms_sigma = _sigma_norm(offsets, eps)  # (N, N)
    grads = _sigma_gradient(offsets, eps)  # (N, N, 3)

    # α-α gradient term
    phi = _phi_alpha(norms_sigma, r_alpha, d_alpha, params.h_bump)  # (N, N)
    np.fill_diagonal(phi, 0.0)
    u_alpha_pos = (phi[..., None] * grads).sum(axis=1)  # (N, 3)

    # α-α velocity consensus (within neighborhood)
    in_range = (norms_sigma < r_alpha) & ~np.eye(n, dtype=bool)
    a_ij = _bump(norms_sigma / r_alpha, params.h_bump) * in_range.astype(np.float64)
    vel_diff = velocities[None, :, :] - velocities[:, None, :]
    u_alpha_vel = (a_ij[..., None] * vel_diff).sum(axis=1)

    u_alpha = params.c1_alpha * u_alpha_pos + params.c2_alpha * u_alpha_vel

    # γ-agent (goal) navigation
    pos_err = goal_pos[None, :] - positions  # (N, 3)
    vel_err = goal_vel[None, :] - velocities  # (N, 3)
    u_gamma = params.c1_gamma * pos_err + params.c2_gamma * vel_err

    accel = u_alpha + u_gamma
    desired_vel = velocities + accel

    # clip speed
    speeds = np.linalg.norm(desired_vel, axis=1, keepdims=True)
    too_fast = speeds > params.max_speed
    desired_vel = np.where(
        too_fast,
        desired_vel / (speeds + 1e-6) * params.max_speed,
        desired_vel,
    )
    return desired_vel
