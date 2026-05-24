"""CHANAKYA — Curvature-Hamilton Action-minimizing Network for Kinematic
Yield Advantage.

GPS-denied SEAD ingress planner. Each friendly drone plans an individual
geodesic from current position to its target on the Riemannian threat
manifold built from the live hostile defense field.

Pieces (§3 of the formulations doc):
  1. Threat field Φ(x) — `core.threat_field`
  2. Riemannian metric g_ij = δ(1+βΦ)^γ — `core.riemannian.MetricParams`
  3. Grid-Dijkstra geodesic — `core.riemannian.geodesic_path`
  4. Action functional ∫ sqrt(g) ds — `core.riemannian.straight_line_action`
     (and the geodesic Dijkstra cost) for the AMST sanity check.
  5. Reactive replan when the defense field changes (radar activates / dies).

`chanakya_plan_swarm` plans paths for every drone in the friendly swarm,
returning a per-drone waypoint queue. The LiveSession reflex layer
consumes the queue, popping waypoints as drones pass through them.

For the live console we run the planner once on scenario start and replan
incrementally only when `defense_field_dirty` is set (asset added /
removed / toggled). This keeps the per-tick cost near zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sargvision_swarm.core.riemannian import (
    Grid2D,
    MetricParams,
    geodesic_path,
    straight_line_action,
)
from sargvision_swarm.core.threat_field import DefenseAsset


@dataclass
class ChanakyaParams:
    grid: Grid2D = field(
        default_factory=lambda: Grid2D(
            x_min=-40.0,
            x_max=40.0,
            y_min=-40.0,
            y_max=40.0,
            nx=41,
            ny=41,
            z_fixed=6.0,
        )
    )
    metric: MetricParams = field(default_factory=MetricParams)
    waypoint_reach_radius: float = 1.8  # drone "reaches" a waypoint within this
    replan_on_field_change: bool = True


@dataclass
class ChanakyaPlan:
    """A planned geodesic for one drone."""

    waypoints: np.ndarray  # (K, 3)
    action_cost: float
    straight_cost: float  # baseline for cost-saving telemetry

    @property
    def savings_ratio(self) -> float:
        """Fraction of action saved vs straight-line baseline."""
        if self.straight_cost < 1e-9:
            return 0.0
        return max(0.0, 1.0 - self.action_cost / self.straight_cost)


@dataclass
class ChanakyaState:
    plans: dict[int, ChanakyaPlan] = field(default_factory=dict)  # drone_idx → plan
    next_waypoint_idx: dict[int, int] = field(default_factory=dict)
    last_field_hash: int = 0
    n_plans_total: int = 0


def _hash_defense_field(assets: list[DefenseAsset]) -> int:
    """Cheap hash so we replan only when the threat picture actually changes."""
    return hash(
        tuple(
            (
                round(float(a.pos[0]), 1),
                round(float(a.pos[1]), 1),
                round(float(a.engagement_radius), 1),
                bool(a.active),
            )
            for a in assets
        )
    )


def chanakya_plan_swarm(
    swarm_positions: np.ndarray,
    targets: np.ndarray,
    defense_field: list[DefenseAsset],
    state: ChanakyaState,
    params: ChanakyaParams | None = None,
    force: bool = False,
) -> dict[int, ChanakyaPlan]:
    """Plan a geodesic per drone. Updates `state` in place.

    Returns the per-drone plan dict. If the defense field hasn't changed
    since the last planning round (and `force` is False), returns the
    cached plans without re-running Dijkstra.
    """
    p = params or ChanakyaParams()
    field_hash = _hash_defense_field(defense_field)
    if not force and field_hash == state.last_field_hash and state.plans:
        return state.plans
    state.last_field_hash = field_hash
    new_plans: dict[int, ChanakyaPlan] = {}
    for i in range(swarm_positions.shape[0]):
        start = swarm_positions[i].copy()
        tgt = targets[i].copy()
        # Lift z to grid's fixed altitude — planner is 2D on cruise plane.
        start[2] = p.grid.z_fixed
        tgt[2] = p.grid.z_fixed
        waypoints, cost = geodesic_path(start, tgt, p.grid, defense_field, p.metric)
        straight = straight_line_action(start, tgt, defense_field, p.metric)
        new_plans[i] = ChanakyaPlan(
            waypoints=waypoints,
            action_cost=cost,
            straight_cost=straight,
        )
        state.next_waypoint_idx[i] = 1 if waypoints.shape[0] > 1 else 0
    state.plans = new_plans
    state.n_plans_total += 1
    return new_plans


def desired_velocity(
    drone_idx: int,
    drone_pos: np.ndarray,
    state: ChanakyaState,
    params: ChanakyaParams | None = None,
    cruise_speed: float = 1.0,
) -> np.ndarray:
    """Compute the velocity command that pulls a drone toward its next waypoint.

    Pops waypoints as the drone reaches them. If all waypoints consumed,
    returns a zero vector (drone arrived).
    """
    p = params or ChanakyaParams()
    plan = state.plans.get(drone_idx)
    if plan is None or plan.waypoints.shape[0] == 0:
        return np.zeros(3)
    k = state.next_waypoint_idx.get(drone_idx, 0)
    # Advance until we find an unreached waypoint.
    while k < plan.waypoints.shape[0]:
        wp = plan.waypoints[k]
        if np.linalg.norm(drone_pos - wp) > p.waypoint_reach_radius:
            break
        k += 1
    state.next_waypoint_idx[drone_idx] = k
    if k >= plan.waypoints.shape[0]:
        return np.zeros(3)
    wp = plan.waypoints[k]
    direction = wp - drone_pos
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        return np.zeros(3)
    return direction / dist * cruise_speed


def plan_summary(state: ChanakyaState) -> dict:
    """Aggregate telemetry for the LiveSession render_stats."""
    if not state.plans:
        return {
            "n_drones_planned": 0,
            "total_action_cost": 0.0,
            "total_straight_cost": 0.0,
            "mean_savings_ratio": 0.0,
            "n_replans": int(state.n_plans_total),
        }
    plans = list(state.plans.values())
    action_sum = float(sum(p.action_cost for p in plans))
    straight_sum = float(sum(p.straight_cost for p in plans))
    mean_savings = float(np.mean([p.savings_ratio for p in plans]))
    return {
        "n_drones_planned": len(plans),
        "total_action_cost": action_sum,
        "total_straight_cost": straight_sum,
        "mean_savings_ratio": mean_savings,
        "n_replans": int(state.n_plans_total),
    }
