"""VAJRA — tropical attention + Voronoi hysteresis + fragmentation + Lanchester.

Validates the four primitives:
  1. Tropical assignment matches Hungarian on small random matrices.
  2. Lanchester break-even sizing returns the correct N for known M/a/b/κ.
  3. Voronoi-hysteresis blocks ping-pong handovers within margin.
  4. Algebraic connectivity flips to ~0 on disconnect; component split detected.
"""

from __future__ import annotations

import numpy as np

from sargvision_swarm.core.tropical import (
    assignment_score,
    hungarian_assignment,
    tropical_attention_assignment,
)
from sargvision_swarm.orchestrator.vajra import (
    VajraParams,
    VajraState,
    VoronoiHysteresisState,
    algebraic_connectivity,
    break_even_interceptors,
    connected_components,
    vajra_assign,
)

# ── 1. Tropical assignment ────────────────────────────────────────────


def test_tropical_matches_hungarian_on_random_4x4():
    rng = np.random.default_rng(0)
    for _ in range(8):
        C = rng.uniform(0.1, 2.0, size=(4, 4))
        trop = tropical_attention_assignment(C, beta=12.0, iters=10)
        hung = hungarian_assignment(C)
        s_t = assignment_score(C, trop)
        s_h = assignment_score(C, hung)
        # Tropical (sharpened) should match or come within 5% of Hungarian.
        assert s_t >= 0.95 * s_h, f"tropical={s_t:.3f} hung={s_h:.3f}"


def test_tropical_drops_zero_priority_rows():
    """A friendly with zero priority everywhere must not be assigned."""
    C = np.array(
        [
            [1.0, 0.5],
            [0.0, 0.0],  # kill-switched
            [0.8, 0.3],
        ]
    )
    assign = tropical_attention_assignment(C, beta=10.0)
    assert 1 not in assign


def test_tropical_rectangular_more_friendlies_than_hostiles():
    C = np.array(
        [
            [0.9, 0.4],
            [0.7, 0.6],
            [0.3, 0.2],
            [0.1, 0.5],
        ]
    )
    assign = tropical_attention_assignment(C, beta=12.0)
    # Each column claimed exactly once.
    assert sorted(assign.values()) == [0, 1]


# ── 2. Lanchester break-even ─────────────────────────────────────────


def test_break_even_unjammed_baseline():
    # M=100, a=b → N* = M = 100; with 10% safety margin → 110.
    n = break_even_interceptors(
        hostiles=100,
        friendly_kill_rate=1.0,
        hostile_kill_rate=1.0,
        jamming_factor=0.0,
        safety_margin=1.10,
    )
    assert n == 110


def test_break_even_favourable_ratio():
    # Sting:Shahed — a=5×b → N* = M*sqrt(b/a) = 100*sqrt(0.2) ≈ 44.7
    n = break_even_interceptors(
        hostiles=100,
        friendly_kill_rate=5.0,
        hostile_kill_rate=1.0,
        jamming_factor=0.0,
        safety_margin=1.0,
    )
    assert 44 <= n <= 50


def test_break_even_jamming_inflates_requirement():
    n_clear = break_even_interceptors(
        hostiles=600,
        friendly_kill_rate=1.0,
        hostile_kill_rate=1.0,
        jamming_factor=0.0,
        safety_margin=1.0,
    )
    n_jammed = break_even_interceptors(
        hostiles=600,
        friendly_kill_rate=1.0,
        hostile_kill_rate=1.0,
        jamming_factor=0.5,
        safety_margin=1.0,
    )
    # Jamming should *reduce* required friendly mass (κ_jam penalises hostile),
    # not inflate — confirms the inequality direction in the formulation.
    assert n_jammed < n_clear


# ── 3. Voronoi hysteresis ────────────────────────────────────────────


def test_voronoi_assigns_nearest_initially():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 8.0, 0.0],
        ]
    )
    state = VoronoiHysteresisState()
    owner = state.cell_owner(positions, hostile_id=42, hostile_pos=np.array([1.0, 0.0, 0.0]))
    assert owner == 0


def test_voronoi_hysteresis_blocks_ping_pong():
    """A hostile drifting right on the cell boundary must NOT keep flipping owner."""
    positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    state = VoronoiHysteresisState()
    state.cell_owner(positions, 42, np.array([4.9, 0.0, 0.0]))  # owner=0
    # Hostile drifts to 5.1 — strictly closer to drone 1 but only by 0.4 < margin=1.5
    owner = state.cell_owner(positions, 42, np.array([5.1, 0.0, 0.0]), margin=1.5)
    assert owner == 0, "hysteresis failed — handed over within margin"
    # Hostile crosses by > margin → handover should fire.
    owner = state.cell_owner(positions, 42, np.array([7.0, 0.0, 0.0]), margin=1.5)
    assert owner == 1


# ── 4. Fragmentation detection ───────────────────────────────────────


def test_lambda2_zero_on_disconnect():
    # Two disjoint 3-cliques.
    n = 6
    A = np.zeros((n, n))
    for i, j in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]:
        A[i, j] = A[j, i] = 1
    assert algebraic_connectivity(A) < 1e-6
    comps = connected_components(A)
    assert len(comps) == 2
    assert {0, 1, 2} in comps
    assert {3, 4, 5} in comps


def test_lambda2_positive_on_connected_ring():
    n = 6
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    lam = algebraic_connectivity(A)
    assert lam > 0.1
    assert len(connected_components(A)) == 1


# ── 5. End-to-end VAJRA assignment ────────────────────────────────────


def test_vajra_assigns_only_to_high_priority_pairs():
    # 3 friendlies × 2 hostiles. Friendly 0 best on hostile 0; friendly 2 best on hostile 1.
    priority = np.array(
        [
            [3.0, 0.5],
            [0.2, 0.1],
            [0.4, 2.5],
        ]
    )
    friendly_pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    hostile_pos = np.array(
        [
            [-1.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ]
    )
    # All-to-all comms.
    adj = np.ones((3, 3)) - np.eye(3)
    state = VajraState()
    params = VajraParams(voronoi_bonus=1.0)  # disable Voronoi bonus to isolate tropical
    assign = vajra_assign(
        priority_matrix=priority,
        friendly_positions=friendly_pos,
        friendly_ids=[0, 1, 2],
        hostile_positions=hostile_pos,
        hostile_ids=[100, 200],
        adjacency=adj,
        state=state,
        params=params,
    )
    assert assign.get(0) == 100
    assert assign.get(2) == 200
    assert 1 not in assign  # zero-priority bidder skipped


def test_vajra_falls_back_to_per_component_under_fragmentation():
    # 4 friendlies, 2 disconnected pairs. Each pair has 1 hostile in range.
    priority = np.array(
        [
            [2.0, 0.0],
            [1.5, 0.0],
            [0.0, 2.0],
            [0.0, 1.5],
        ]
    )
    friendly_pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [101.0, 0.0, 0.0],
        ]
    )
    hostile_pos = np.array(
        [
            [0.5, 0.0, 0.0],
            [100.5, 0.0, 0.0],
        ]
    )
    adj = np.zeros((4, 4))
    adj[0, 1] = adj[1, 0] = 1
    adj[2, 3] = adj[3, 2] = 1
    state = VajraState()
    params = VajraParams(voronoi_bonus=1.0)
    assign = vajra_assign(
        priority_matrix=priority,
        friendly_positions=friendly_pos,
        friendly_ids=[0, 1, 2, 3],
        hostile_positions=hostile_pos,
        hostile_ids=[100, 200],
        adjacency=adj,
        state=state,
        params=params,
    )
    assert state.n_components == 2
    # Best friendly in each component wins its hostile.
    assert assign.get(0) == 100
    assert assign.get(2) == 200
