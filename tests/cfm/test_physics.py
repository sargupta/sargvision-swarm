"""Tests for the physics-realism layer — values cross-checked vs research files."""

from __future__ import annotations

import numpy as np
import pytest

from sargvision_swarm.cfm.physics.battery import BatteryModel, Chemistry
from sargvision_swarm.cfm.physics.environment import (
    altitude_derate_factor,
    hover_power_w,
    isa_density,
    wind_tolerated,
)
from sargvision_swarm.cfm.physics.pnt_fusion import PNTSource, fuse_pnt
from sargvision_swarm.cfm.physics.rf_link import (
    LinkBudget,
    friis_path_loss_db,
    link_capacity_mbps,
    link_snr_db,
)

# ── RF link (vs file 49) ────────────────────────────────────────────────────


def test_fspl_1km_5p8ghz():
    # ~107.7 dB free-space; allow the small absorption add-on
    loss = friis_path_loss_db(1000.0, 5.8e9, with_absorption=False)
    assert 107.0 <= loss <= 108.5


def test_fspl_doubling_distance_adds_6db():
    a = friis_path_loss_db(100.0, 2.4e9, with_absorption=False)
    b = friis_path_loss_db(200.0, 2.4e9, with_absorption=False)
    assert b == pytest.approx(a + 6.0, abs=0.05)


def test_60ghz_absorption_kills_range():
    # 60 GHz has ~15 dB/km O2 absorption → much shorter max range than 5.8 GHz
    lb_low = LinkBudget(frequency_hz=5.8e9)
    lb_mm = LinkBudget(frequency_hz=60e9)
    assert lb_mm.max_range_m() < lb_low.max_range_m()


def test_link_connected_near_denied_far():
    lb = LinkBudget(tx_power_dbm=30.0, frequency_hz=2.4e9)
    assert lb.is_connected(1000.0)  # 1 km: connected
    assert not lb.is_connected(500_000.0)  # 500 km: not


def test_jammer_degrades_snr():
    lb = LinkBudget(tx_power_dbm=30.0, frequency_hz=2.4e9)
    clean = lb.snr_db(2000.0, None)
    jammed = lb.snr_db(2000.0, jammer_power_dbm=-40.0)  # strong nearby jammer
    assert jammed < clean


def test_shannon_capacity_monotone_in_snr():
    assert link_capacity_mbps(20.0) > link_capacity_mbps(10.0) > link_capacity_mbps(0.0)


def test_link_snr_wrapper_runs():
    assert isinstance(link_snr_db(1000.0), float)


# ── Environment (vs file 46) ────────────────────────────────────────────────


def test_isa_density_sea_level():
    assert isa_density(0.0) == pytest.approx(1.225, abs=1e-3)


def test_isa_density_leh():
    # ~3500 m → ~0.86 kg/m^3
    rho = isa_density(3500.0)
    assert 0.82 <= rho <= 0.90


def test_altitude_derate_leh_and_siachen():
    leh = altitude_derate_factor(3500.0)
    siachen = altitude_derate_factor(5400.0)
    assert leh == pytest.approx(1.19, abs=0.04)
    assert siachen == pytest.approx(1.32, abs=0.05)
    assert siachen > leh > 1.0


def test_hover_power_increases_with_altitude():
    # same airframe costs more power to hover at Leh than sea level
    p_sea = hover_power_w(6.4, 0.5, altitude_m=0.0)
    p_leh = hover_power_w(6.4, 0.5, altitude_m=3500.0)
    assert p_leh > p_sea
    assert p_leh / p_sea == pytest.approx(altitude_derate_factor(3500.0), abs=0.02)


def test_wind_tolerance_rule():
    assert wind_tolerated(30.0, 9.0)  # 9 m/s < 1/3 of 30
    assert not wind_tolerated(30.0, 15.0)  # 15 m/s > 1/3 of 30


# ── Battery (vs files 47/90) ────────────────────────────────────────────────


def test_pack_energy_by_chemistry():
    b_lipo = BatteryModel(mass_kg=1.0, chemistry=Chemistry.LIPO_2024)
    b_limetal = BatteryModel(mass_kg=1.0, chemistry=Chemistry.LI_METAL_2026)
    assert b_lipo.nominal_wh == pytest.approx(150.0)
    assert b_limetal.nominal_wh == pytest.approx(240.0)
    assert b_limetal.nominal_wh > b_lipo.nominal_wh


def test_cold_derate_reduces_usable():
    b = BatteryModel(mass_kg=1.0, chemistry=Chemistry.LIPO_2024)
    warm = b.usable_wh(temp_c=25.0)
    cold = b.usable_wh(temp_c=-10.0)
    assert cold < warm
    assert cold == pytest.approx(warm * 0.5, rel=0.05)  # ~50% at -10C


def test_endurance_small_quad_realistic():
    # 0.25 kg LiPo pack at ~80 W hover ≈ ~24 min order-of-magnitude
    b = BatteryModel(mass_kg=0.25, chemistry=Chemistry.LIPO_2024)
    mins = b.endurance_min(80.0, temp_c=25.0)
    assert 15.0 <= mins <= 40.0


def test_ladakh_compound_derate_about_two_thirds():
    # cold (-10C) + higher altitude power ≈ ~60-70% of sea-level endurance
    b = BatteryModel(mass_kg=1.0, chemistry=Chemistry.LIPO_2024)
    p_sea = hover_power_w(6.4, 0.5, altitude_m=0.0)
    p_leh = hover_power_w(6.4, 0.5, altitude_m=3500.0)
    frac = b.ladakh_derate_vs_sealevel(p_sea, p_leh, temp_c=-10.0)
    assert 0.35 <= frac <= 0.75


# ── PNT fusion (Round-2 reframe) ────────────────────────────────────────────


def _src(name, pos, sigma, available=True):
    return PNTSource(
        name=name, position_enu_m=np.array(pos, dtype=float), sigma_m=sigma, available=available
    )


def test_fusion_inverse_variance_weights_precise_source():
    truth = np.array([100.0, 200.0, 50.0])
    sources = [
        _src("gps", truth + np.array([1.0, 0, 0]), sigma=2.0),
        _src("navic", truth + np.array([8.0, 0, 0]), sigma=12.0),
        _src("vio", truth + np.array([3.0, 0, 0]), sigma=5.0),
    ]
    res = fuse_pnt(sources)
    assert res.has_fix
    # fused should be close to truth, pulled toward the precise GPS
    assert float(np.linalg.norm(res.fused_position_enu_m - truth)) < 4.0
    assert res.fused_sigma_m < 2.0  # fusion beats the best single source


def test_fusion_rejects_spoofed_source():
    truth = np.array([0.0, 0.0, 0.0])
    sources = [
        _src("gps_spoofed", truth + np.array([800.0, 0, 0]), sigma=3.0),  # claims tight, lies big
        _src("navic", truth + np.array([6.0, 0, 0]), sigma=10.0),
        _src("vio", truth + np.array([2.0, 0, 0]), sigma=5.0),
        _src("magnav", truth + np.array([10.0, 0, 0]), sigma=15.0),
    ]
    res = fuse_pnt(sources)
    assert "gps_spoofed" in res.rejected_sources
    assert float(np.linalg.norm(res.fused_position_enu_m - truth)) < 15.0


def test_fusion_no_rejection_with_two_sources():
    # cannot outvote a spoofer with only 2 sources → no rejection attempted
    sources = [
        _src("gps", np.array([500.0, 0, 0]), sigma=3.0),
        _src("vio", np.array([0.0, 0, 0]), sigma=5.0),
    ]
    res = fuse_pnt(sources)
    assert res.rejected_sources == []
    assert set(res.used_sources) == {"gps", "vio"}


def test_fusion_all_unavailable_no_fix():
    sources = [_src("gps", np.zeros(3), 3.0, available=False)]
    res = fuse_pnt(sources)
    assert not res.has_fix
