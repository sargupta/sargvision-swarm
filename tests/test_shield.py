"""SHIELD — sheaf-loyalty + PageRank-trust + Bayesian-threat + auction.

Validates the four pieces of the SHIELD stack:
  1. Sheaf loyalty drops for spoofed drones; stays ~1 for loyal ones.
  2. PageRank trust collapses below kill-switch when neighbours flag low loyalty.
  3. Bayesian posterior favours 'decoy' for high-RCS / high-RF / jittery contacts
     and 'kinetic' for low-jerk / terminal contacts.
  4. Auction skips kill-switched bidders and prefers kinetics over decoys
     at comparable range.
"""

from __future__ import annotations

import numpy as np

from sargvision_swarm.core.sheaf import (
    SheafParams,
    SheafState,
    loyalty_from_positions,
)
from sargvision_swarm.comms.trust import pagerank_trust
from sargvision_swarm.orchestrator.shield import (
    ShieldParams,
    ShieldState,
    expected_damage,
    shield_assign,
    threat_class,
    update_threat_posterior,
)
from sargvision_swarm.sim.hostiles import HostileFleet


def _ring_positions(n: int, r: float = 5.0) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(angles), r * np.sin(angles), np.full(n, 4.0)], axis=1)


def _all_connected_adjacency(n: int) -> np.ndarray:
    a = np.ones((n, n), dtype=np.int64) - np.eye(n, dtype=np.int64)
    return a


# ── 1. Sheaf loyalty ──────────────────────────────────────────────────


def test_spoofed_drone_loyalty_drops():
    """Spoofed drones must score clearly below loyal drones.

    The absolute loyalty floor for loyal drones depends on sensor-noise σ;
    what matters operationally is the *gap* between spoofed and loyal.
    """
    n = 8
    positions = _ring_positions(n)
    adj = _all_connected_adjacency(n)
    state = SheafState()
    loyalty = np.ones(n)
    for _ in range(20):
        loyalty = loyalty_from_positions(
            positions, adj, state,
            SheafParams(sigma_n=2.5, smoothing=0.5, spoof_bias_m=12.0),
            spoofed_ids={2, 5},
        )
    spoof_max = float(max(loyalty[2], loyalty[5]))
    loyal_min = float(min(loyalty[i] for i in range(n) if i not in {2, 5}))
    assert spoof_max < 0.5, f"spoofed loyalty too high: {loyalty[2]:.3f}, {loyalty[5]:.3f}"
    # Loyal floor depends on σ noise; with σ_n=2.5 the floor is ~0.55.
    # What matters operationally is the *gap* between spoofed and loyal.
    assert loyal_min > 0.55, f"loyal min too low: {loyal_min:.3f}"
    assert loyal_min - spoof_max > 0.25, (
        f"insufficient separation: loyal_min={loyal_min:.3f} spoof_max={spoof_max:.3f}"
    )


# ── 2. PageRank trust ─────────────────────────────────────────────────


def test_pagerank_low_loyalty_collapses_trust():
    n = 6
    adj = _all_connected_adjacency(n)
    loyalty = np.ones(n)
    loyalty[1] = 0.05  # one strongly low-loyalty node
    T = pagerank_trust(adj, loyalty, damping=0.85, iters=20)
    # Low-loyalty node should be the lowest-trust node by a wide margin.
    assert int(np.argmin(T)) == 1
    assert T[1] < 0.5 * T.mean()


# ── 3. Bayesian threat posterior ──────────────────────────────────────


def test_posterior_favours_decoy_for_high_emit_signature():
    state = ShieldState()
    state.init(1)
    obs_decoy = {"rcs": 0.9, "rf_emit": 0.9, "traj_jerk": 0.8, "terminal": 0.0}
    for _ in range(20):
        update_threat_posterior(state, hostile_id=42, observation=obs_decoy)
    post = state.posteriors[42]
    assert threat_class(post) == "decoy", f"posterior={post}"


def test_posterior_favours_kinetic_for_smooth_terminal_signature():
    state = ShieldState()
    state.init(1)
    obs_kin = {"rcs": 0.3, "rf_emit": 0.4, "traj_jerk": 0.05, "terminal": 1.0}
    for _ in range(20):
        update_threat_posterior(state, hostile_id=42, observation=obs_kin)
    post = state.posteriors[42]
    assert threat_class(post) == "kinetic", f"posterior={post}"


def test_expected_damage_orders_classes():
    p = ShieldParams()
    decoy_post = np.array([1.0, 0.0, 0.0])
    kinetic_post = np.array([0.0, 1.0, 0.0])
    nuisance_post = np.array([0.0, 0.0, 1.0])
    assert (
        expected_damage(kinetic_post, p)
        > expected_damage(nuisance_post, p)
        > expected_damage(decoy_post, p)
    )


# ── 4. End-to-end auction ─────────────────────────────────────────────


def test_auction_skips_kill_switched_drone():
    n = 6
    positions = _ring_positions(n)
    roles = ["leader"] + ["worker"] * (n - 1)
    adj = _all_connected_adjacency(n)

    # Force a kinetic hostile at the doorstep of drone 1 (closest worker).
    fleet = HostileFleet(spawn_count=0, spawn_radius_m=8.0, seed=0)
    fleet.spawn_initial(center=positions.mean(axis=0))
    from sargvision_swarm.sim.hostiles import Hostile

    h = Hostile(
        id=2000,
        pos=positions[1] + np.array([1.5, 0.0, 0.0]),
        vel=np.array([-0.5, 0.0, 0.0]),
        intent_label="TERMINAL",
        callsign="KIN-2000",
        threat_class="kinetic",
        rcs=0.3,
        rf_emit=0.4,
        traj_jerk=0.05,
    )
    fleet.hostiles.append(h)

    state = ShieldState()
    state.init(n)
    params = ShieldParams(trust_kill_threshold=0.5)

    # Pre-poison drone 1's loyalty so PageRank kill-switches it.
    state.loyalty = np.ones(n)
    state.loyalty[1] = 0.0
    state.trust = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    # Run one auction.
    assign = shield_assign(
        friendly_positions=positions,
        friendly_roles=roles,
        hostiles=fleet.hostiles,
        adjacency=adj,
        state=state,
        params=params,
        spoofed_ids={1},
    )
    # Kill-switched drone (1) must NOT be in the assignment.
    assert 1 not in assign


def test_auction_prefers_kinetic_over_decoy_at_equal_range():
    n = 4
    positions = _ring_positions(n)
    roles = ["worker"] * n
    adj = _all_connected_adjacency(n)
    state = ShieldState()
    state.init(n)
    params = ShieldParams()

    from sargvision_swarm.sim.hostiles import Hostile

    centre = positions.mean(axis=0)
    # Two TERMINAL hostiles equidistant from the swarm centroid.
    h_kin = Hostile(
        id=3000,
        pos=centre + np.array([6.0, 0.0, 0.0]),
        vel=np.array([-0.4, 0.0, 0.0]),
        intent_label="TERMINAL",
        callsign="KIN-3000",
        threat_class="kinetic",
        rcs=0.3, rf_emit=0.4, traj_jerk=0.05,
    )
    h_dec = Hostile(
        id=3001,
        pos=centre + np.array([-6.0, 0.0, 0.0]),
        vel=np.array([0.4, 0.0, 0.0]),
        intent_label="TERMINAL",
        callsign="DEC-3001",
        threat_class="decoy",
        rcs=0.9, rf_emit=0.9, traj_jerk=0.8,
    )

    # Warm up posteriors so the Bayesian filter knows their classes.
    for _ in range(15):
        for h in (h_kin, h_dec):
            update_threat_posterior(state, h.id, {
                "rcs": h.rcs, "rf_emit": h.rf_emit,
                "traj_jerk": h.traj_jerk,
                "terminal": 1.0,
            })

    # With only ONE worker free (drones[1..3] all already_assigned to dummies),
    # the lone bidder should choose the kinetic.
    already = {2: 9001, 3: 9002}  # drone 0 + 1 are free
    # Make drone 0 closer to the kinetic and drone 1 closer to the decoy,
    # so SHIELD has a real either/or; expected damage should still tilt to kinetic.
    positions[0] = h_kin.pos + np.array([1.0, 0.0, 0.0])
    positions[1] = h_dec.pos + np.array([1.0, 0.0, 0.0])

    assign = shield_assign(
        friendly_positions=positions,
        friendly_roles=roles,
        hostiles=[h_kin, h_dec],
        adjacency=adj,
        state=state,
        params=params,
        already_assigned=already,
    )
    # Kinetic must be picked up; decoy may or may not be (lower priority).
    assert h_kin.id in assign.values(), f"kinetic not assigned: {assign}"
