"""Threat field — Φ(x, t) kernel-density sum over hostile defense assets.

Each defense asset z_k with engagement radius ρ_k contributes a Gaussian
centred at z_k with σ = ρ_k. Only active assets count. The field is
called every planner step + every geodesic integration substep, so the
implementation is fully vectorised in NumPy.

A defender's perspective: where on the map is it most dangerous to fly?
A planner's perspective: this is the conformal factor for the Riemannian
engagement metric g_ij = δ_ij (1 + β Φ)^γ — high-threat regions become
"longer to traverse" so geodesics curve around them naturally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DefenseAsset:
    """A hostile radar / SAM / EW battery."""

    pos: np.ndarray  # (3,) world position
    engagement_radius: float  # ρ_k — effective threat half-width
    active: bool = True
    name: str = ""  # callsign e.g., "S-400-N1"


def threat_field(
    query_points: np.ndarray,
    assets: list[DefenseAsset],
) -> np.ndarray:
    """Φ(x) at each row of `query_points`. Shape (M,) for (M, 3) input.

    Vectorised — O(M * K) for M queries and K assets. M=10⁴ × K=10 runs in
    well under 10 ms on Apple silicon.
    """
    if query_points.ndim == 1:
        query_points = query_points.reshape(1, -1)
    n_query = query_points.shape[0]
    if not assets:
        return np.zeros(n_query)
    field = np.zeros(n_query)
    for a in assets:
        if not a.active:
            continue
        rho = max(float(a.engagement_radius), 1e-6)
        # Gaussian normalised by sqrt(2π) ρ so peak value = 1 / (sqrt(2π) ρ).
        diffs = query_points - a.pos[None, :]
        d2 = np.einsum("mi,mi->m", diffs, diffs)
        field = field + np.exp(-d2 / (2.0 * rho * rho)) / (np.sqrt(2.0 * np.pi) * rho)
    return field


def threat_field_gradient(
    query_points: np.ndarray,
    assets: list[DefenseAsset],
) -> np.ndarray:
    """∇Φ(x) at each row of `query_points`. Shape (M, 3).

    Useful for soft repulsion / reflex steering even when the full geodesic
    planner isn't running (e.g., reactive avoidance during ingress when the
    grid plan is replanning).
    """
    if query_points.ndim == 1:
        query_points = query_points.reshape(1, -1)
    n_query, dim = query_points.shape
    grad = np.zeros((n_query, dim))
    if not assets:
        return grad
    for a in assets:
        if not a.active:
            continue
        rho = max(float(a.engagement_radius), 1e-6)
        diffs = query_points - a.pos[None, :]
        d2 = np.einsum("mi,mi->m", diffs, diffs)
        weight = np.exp(-d2 / (2.0 * rho * rho)) / (np.sqrt(2.0 * np.pi) * rho)
        # ∇ exp(-||x-z||²/2ρ²) = -(x-z)/ρ² * gauss
        grad = grad - (diffs / (rho * rho)) * weight[:, None]
    return grad
