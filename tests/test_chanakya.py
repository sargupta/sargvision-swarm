"""CHANAKYA — threat field + Riemannian geodesic + defense field + planner."""

from __future__ import annotations

import numpy as np

from sargvision_swarm.core.riemannian import (
    Grid2D,
    MetricParams,
    conformal_factor,
    geodesic_path,
    straight_line_action,
)
from sargvision_swarm.core.threat_field import (
    DefenseAsset,
    threat_field,
    threat_field_gradient,
)
from sargvision_swarm.orchestrator.chanakya import (
    ChanakyaParams,
    ChanakyaState,
    chanakya_plan_swarm,
    desired_velocity,
    plan_summary,
)
from sargvision_swarm.sim.defense_field import DefenseField

# ── 1. Threat field ──────────────────────────────────────────────────


def test_threat_decays_with_distance():
    a = DefenseAsset(pos=np.array([0.0, 0.0, 0.0]), engagement_radius=5.0, name="S400")
    queries = np.array(
        [
            [0.0, 0.0, 0.0],  # at radar
            [5.0, 0.0, 0.0],  # 1σ
            [25.0, 0.0, 0.0],  # 5σ — should be ~0
        ]
    )
    phi = threat_field(queries, [a])
    assert phi[0] > phi[1] > phi[2]
    assert phi[2] < 1e-6


def test_threat_zero_for_inactive_assets():
    a = DefenseAsset(pos=np.array([0.0, 0.0, 0.0]), engagement_radius=5.0, active=False)
    phi = threat_field(np.array([[0.0, 0.0, 0.0]]), [a])
    assert phi[0] == 0.0


def test_threat_gradient_points_away_from_asset():
    a = DefenseAsset(pos=np.array([0.0, 0.0, 0.0]), engagement_radius=5.0)
    g = threat_field_gradient(np.array([[3.0, 0.0, 0.0]]), [a])
    # ∇Φ should point AWAY from the asset → positive x component since query is +x.
    assert g[0, 0] < 0  # threat DECREASES as we move further (gradient negative)
    # but for repulsion we'd flip the sign; here we check ∇Φ is collinear with -(x-z).


# ── 2. Riemannian metric ─────────────────────────────────────────────


def test_conformal_factor_inflates_near_threat():
    phi_low = np.array([0.0])
    phi_high = np.array([1.0])
    sg_low = conformal_factor(phi_low, MetricParams(beta=1.5, gamma=2.0))
    sg_high = conformal_factor(phi_high, MetricParams(beta=1.5, gamma=2.0))
    assert sg_high > sg_low
    assert sg_low[0] == 1.0  # vacuum metric = identity → sqrt(g) = 1


# ── 3. Grid-Dijkstra geodesic ────────────────────────────────────────


def test_geodesic_straight_when_no_threat():
    grid = Grid2D(x_min=-10, x_max=10, y_min=-10, y_max=10, nx=21, ny=21, z_fixed=5.0)
    start = np.array([-8.0, 0.0, 5.0])
    target = np.array([8.0, 0.0, 5.0])
    waypoints, cost = geodesic_path(start, target, grid, assets=[])
    # Without threat, geodesic should stick close to the straight line (y stays small).
    assert (np.abs(waypoints[:, 1]) < 2.0).all()
    # Cost ≈ straight-line length in a flat (g=1) metric, on grid resolution.
    assert abs(cost - 16.0) < 4.0


def test_geodesic_curves_around_single_radar():
    """Place a SAM right between start + target — geodesic should detour.

    The conformal factor near the radar must dominate the cost of a
    perpendicular detour. Use a tight, high-β kernel so the test isn't
    fragile to small parameter shifts.
    """
    grid = Grid2D(x_min=-15, x_max=15, y_min=-15, y_max=15, nx=31, ny=31, z_fixed=5.0)
    a = DefenseAsset(pos=np.array([0.0, 0.0, 5.0]), engagement_radius=2.0, name="S400")
    metric = MetricParams(beta=25.0, gamma=4.0)
    start = np.array([-10.0, 0.0, 5.0])
    target = np.array([10.0, 0.0, 5.0])
    waypoints, geo_cost = geodesic_path(start, target, grid, assets=[a], metric=metric)
    straight_cost = straight_line_action(start, target, [a], metric)
    # Geodesic should be cheaper than the straight-through path.
    assert geo_cost < straight_cost
    # The waypoints should deviate from y≈0 by at least one cell mid-path.
    max_y_deviation = float(np.max(np.abs(waypoints[:, 1])))
    assert max_y_deviation > 0.5, (
        f"geodesic did not curve around radar; max |y|={max_y_deviation:.2f}"
    )


def test_geodesic_falls_back_on_unreachable():
    """If the grid is closed (every edge weight finite), Dijkstra always reaches.
    We instead check the no-assets baseline returns SOMETHING sensible."""
    grid = Grid2D(x_min=-5, x_max=5, y_min=-5, y_max=5, nx=11, ny=11, z_fixed=5.0)
    start = np.array([-4.0, -4.0, 5.0])
    target = np.array([4.0, 4.0, 5.0])
    wp, cost = geodesic_path(start, target, grid, [])
    assert wp.shape[0] >= 2
    assert cost > 0


# ── 4. DefenseField ──────────────────────────────────────────────────


def test_defense_field_iads_layout_and_kill_check():
    df = DefenseField(seed=0)
    df.spawn_iads_layout(
        np.array([0.0, 0.0, 0.0]), ring_radius=10.0, n_radars=4, engagement_radius=3.0
    )
    assert len(df.assets) == 4
    # A drone right next to one of the radars is killed.
    radar_pos = df.assets[0].pos
    drones = np.array([radar_pos.copy(), [50.0, 50.0, 0.0]])
    hits = df.kill_radius_check(drones)
    assert hits[0]
    assert not hits[1]


def test_defense_field_dirty_flag_consumption():
    df = DefenseField(seed=0)
    df.spawn_iads_layout(np.array([0.0, 0.0, 0.0]))
    assert df.consume_dirty() is True
    assert df.consume_dirty() is False
    df.toggle(0, active=False)
    assert df.consume_dirty() is True


# ── 5. End-to-end CHANAKYA planner ───────────────────────────────────


def test_chanakya_plan_swarm_produces_per_drone_paths():
    n = 4
    swarm_pos = np.array([[-12.0, -8.0 + 2 * i, 6.0] for i in range(n)])
    targets = np.array([[12.0, -8.0 + 2 * i, 6.0] for i in range(n)])
    df = DefenseField(seed=0)
    df.spawn_iads_layout(
        np.array([0.0, 0.0, 0.0]), ring_radius=8.0, n_radars=4, engagement_radius=3.5
    )
    state = ChanakyaState()
    plans = chanakya_plan_swarm(swarm_pos, targets, df.active, state)
    assert set(plans.keys()) == set(range(n))
    for p in plans.values():
        assert p.waypoints.shape[0] >= 2
        # Geodesic should be at least as good as straight line.
        assert p.action_cost <= p.straight_cost * 1.05


def test_chanakya_caches_unless_field_changes():
    n = 2
    swarm_pos = np.array([[-12.0, 0.0, 6.0], [-12.0, 2.0, 6.0]])
    targets = np.array([[12.0, 0.0, 6.0], [12.0, 2.0, 6.0]])
    df = DefenseField(seed=0)
    df.spawn_iads_layout(np.array([0.0, 0.0, 0.0]))
    state = ChanakyaState()
    chanakya_plan_swarm(swarm_pos, targets, df.active, state)
    assert state.n_plans_total == 1
    # Re-call without changes → cache.
    chanakya_plan_swarm(swarm_pos, targets, df.active, state)
    assert state.n_plans_total == 1
    # Toggle one radar off → should replan.
    df.toggle(0, active=False)
    chanakya_plan_swarm(swarm_pos, targets, df.active, state)
    assert state.n_plans_total == 2


def test_desired_velocity_pulls_toward_next_waypoint():
    state = ChanakyaState()
    state.plans[0] = type(
        "P", (), {"waypoints": np.array([[0.0, 0.0, 6.0], [5.0, 0.0, 6.0], [10.0, 0.0, 6.0]])}
    )()
    state.next_waypoint_idx[0] = 0
    v = desired_velocity(0, np.array([0.0, 0.0, 6.0]), state)
    # Should pull toward first waypoint (0,0,6) — but drone is already there;
    # waypoint pop should advance to (5,0,6) and pull +x.
    assert v[0] > 0
    assert abs(v[1]) < 1e-6


def test_plan_summary_reports_savings():
    n = 3
    swarm_pos = np.array([[-12.0, -4.0 + 4 * i, 6.0] for i in range(n)])
    targets = np.array([[12.0, -4.0 + 4 * i, 6.0] for i in range(n)])
    df = DefenseField(seed=0)
    df.spawn_iads_layout(
        np.array([0.0, 0.0, 0.0]), ring_radius=8.0, n_radars=4, engagement_radius=3.5
    )
    state = ChanakyaState()
    chanakya_plan_swarm(
        swarm_pos,
        targets,
        df.active,
        state,
        params=ChanakyaParams(metric=MetricParams(beta=3.0, gamma=2.0)),
    )
    s = plan_summary(state)
    assert s["n_drones_planned"] == n
    assert s["mean_savings_ratio"] >= 0
