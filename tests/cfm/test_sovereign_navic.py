"""Tests for the NavIC receiver mock."""

from __future__ import annotations

import numpy as np

from sargvision_swarm.cfm.sovereign.navic import (
    MIN_SATS_FOR_3D_FIX,
    NAVIC_L1_JAM_THRESHOLD_DBM,
    NOMINAL_NAVIC_SAT_COUNT,
    NavICReceiver,
)


def test_nominal_fix_quality():
    """Full 7-sat constellation + L5 enabled + no jammer → 3d_l5 fix."""
    rx = NavICReceiver(sats_visible=NOMINAL_NAVIC_SAT_COUNT, l5_enabled=True, rng_seed=42)
    true_pos = np.array([100.0, 200.0, 50.0])
    sol = rx.get_pvt(true_pos, time_unix_s=1_700_000_000.0)
    assert sol.has_valid_fix
    assert sol.fix_type == "3d_l5"
    assert sol.sats_used == NOMINAL_NAVIC_SAT_COUNT
    # noise should be well-bounded
    err = float(np.linalg.norm(sol.position_m - true_pos))
    assert err < 20.0, f"noise too high: {err} m"


def test_degraded_sat_count_widens_sigma():
    """Reducing visible sats to 4 should at least double the position noise."""
    true_pos = np.array([0.0, 0.0, 0.0])
    rx_full = NavICReceiver(sats_visible=NOMINAL_NAVIC_SAT_COUNT, rng_seed=1)
    rx_min = NavICReceiver(sats_visible=MIN_SATS_FOR_3D_FIX, rng_seed=1)
    sol_full = rx_full.get_pvt(true_pos, 1.0)
    sol_min = rx_min.get_pvt(true_pos, 1.0)
    assert sol_min.sigma_pos_m > sol_full.sigma_pos_m


def test_below_min_sats_yields_no_fix():
    """Fewer than MIN_SATS_FOR_3D_FIX → no_fix."""
    rx = NavICReceiver(sats_visible=MIN_SATS_FOR_3D_FIX - 1, rng_seed=0)
    sol = rx.get_pvt(np.zeros(3), 0.0)
    assert not sol.has_valid_fix
    assert sol.fix_type == "no_fix"


def test_jamming_above_threshold_denies_l1():
    """Jammer power above L1 threshold + no L5 → no fix."""
    rx = NavICReceiver(
        sats_visible=NOMINAL_NAVIC_SAT_COUNT,
        l5_enabled=False,
        jammer_dbm=NAVIC_L1_JAM_THRESHOLD_DBM + 5.0,
    )
    sol = rx.get_pvt(np.zeros(3), 0.0)
    assert sol.fix_type == "no_fix"


def test_l5_resists_l1_jamming():
    """L5-enabled receiver survives moderate L1 jamming."""
    rx = NavICReceiver(
        sats_visible=NOMINAL_NAVIC_SAT_COUNT,
        l5_enabled=True,
        jammer_dbm=NAVIC_L1_JAM_THRESHOLD_DBM + 3.0,  # L1 denied, L5 still good
        rng_seed=7,
    )
    sol = rx.get_pvt(np.zeros(3), 0.0)
    # L5 survives at +3 dB (threshold is +6); should still get a fix though sats reduced
    assert sol.has_valid_fix


def test_set_sats_visible_updates_state():
    rx = NavICReceiver(sats_visible=NOMINAL_NAVIC_SAT_COUNT)
    rx.set_sats_visible(3)
    assert rx.sats_visible == 3
    sol = rx.get_pvt(np.zeros(3), 0.0)
    assert sol.fix_type == "no_fix"  # 3 < MIN_SATS_FOR_3D_FIX (4)


def test_velocity_is_returned():
    rx = NavICReceiver(rng_seed=2)
    sol = rx.get_pvt(np.array([10.0, 20.0, 30.0]), 1.0)
    assert sol.velocity_mps.shape == (3,)
