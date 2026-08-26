"""Tests for the C-UAS Defence scenario + 4 posture strategies."""

from __future__ import annotations

import numpy as np

from sargvision_swarm.cfm.scenarios.c_uas_defence import (
    CriticalInfrastructureAsset,
    CUASDefenceScenario,
    HostileWaveParams,
    ThreatClass,
)
from sargvision_swarm.cfm.strategies.c_uas import (
    CUASStrategy,
    plan_goals_for_strategy,
    recommend_strategy,
)


def _scenario() -> CUASDefenceScenario:
    return CUASDefenceScenario(
        assets=[
            CriticalInfrastructureAsset(
                asset_id="CIA-01",
                position_enu_m=np.array([0.0, 0.0, 50.0]),
                value_usd=10_000_000.0,
                radius_m=500.0,
            )
        ],
        wave_params=HostileWaveParams(
            n_fpv=4, n_owa=3, n_cruise=1, start_range_m=10_000.0, rng_seed=42
        ),
    )


def test_scenario_spawns_correct_number_of_hostiles():
    scn = _scenario()
    wave = scn.spawn_wave()
    assert len(wave.hostiles) == 8
    assert wave.alive_count == 8


def test_threat_class_split_matches_params():
    scn = _scenario()
    wave = scn.spawn_wave()
    by_class = wave.by_class
    assert by_class[ThreatClass.FPV] == 4
    assert by_class[ThreatClass.OWA] == 3
    assert by_class[ThreatClass.CRUISE] == 1


def test_step_wave_advances_positions():
    scn = _scenario()
    wave = scn.spawn_wave()
    initial = np.stack([h.position_enu_m.copy() for h in wave.hostiles])
    scn.step_wave(wave, dt_s=1.0)
    after = np.stack([h.position_enu_m for h in wave.hostiles])
    # at least one drone should have moved
    assert not np.allclose(initial, after)


def test_step_wave_eventually_impacts_asset():
    scn = _scenario()
    wave = scn.spawn_wave()
    # run for ~1000 s — FPVs at 25 m/s travel 25 km, plenty to reach 10 km away
    final = None
    for _ in range(1000):
        final = scn.step_wave(wave, dt_s=1.0)
        if final["alive"] == 0:
            break
    assert final is not None
    assert final["impacts"]["CIA-01"] > 0


def test_layered_strategy_returns_n_goals():
    scn = _scenario()
    wave = scn.spawn_wave()
    goals = plan_goals_for_strategy(CUASStrategy.LAYERED, scn.assets, wave, n_defenders=12)
    assert goals.shape == (12, 3)


def test_point_defence_all_at_same_radius():
    scn = _scenario()
    wave = scn.spawn_wave()
    goals = plan_goals_for_strategy(
        CUASStrategy.POINT_DEFENCE, scn.assets, wave, n_defenders=8
    )
    asset_pos = scn.assets[0].position_enu_m
    # all goals should be within tight bubble (~800 m default)
    distances = np.linalg.norm(goals[:, :2] - asset_pos[:2], axis=1)
    assert distances.max() < 1000.0


def test_area_defence_spreads_widely():
    scn = _scenario()
    wave = scn.spawn_wave()
    goals = plan_goals_for_strategy(
        CUASStrategy.AREA_DEFENCE, scn.assets, wave, n_defenders=12
    )
    asset_pos = scn.assets[0].position_enu_m
    # area defence should reach intermediate range
    distances = np.linalg.norm(goals[:, :2] - asset_pos[:2], axis=1)
    assert distances.max() > 5_000.0


def test_mobile_cap_covers_full_azimuth():
    scn = _scenario()
    wave = scn.spawn_wave()
    n = 12
    goals = plan_goals_for_strategy(CUASStrategy.MOBILE_CAP, scn.assets, wave, n_defenders=n)
    asset_pos = scn.assets[0].position_enu_m
    rel = goals[:, :2] - asset_pos[:2]
    azimuths = np.arctan2(rel[:, 1], rel[:, 0])
    # azimuths should span a wide range (> 180°)
    span_rad = float(azimuths.max() - azimuths.min())
    assert span_rad > np.deg2rad(180.0)


def test_recommend_strategy_picks_point_defence_for_cruise_heavy():
    wave = CUASDefenceScenario(
        wave_params=HostileWaveParams(n_fpv=2, n_owa=2, n_cruise=4)
    ).spawn_wave()
    assert recommend_strategy(wave) == CUASStrategy.POINT_DEFENCE


def test_recommend_strategy_picks_area_for_fpv_heavy():
    wave = CUASDefenceScenario(
        wave_params=HostileWaveParams(n_fpv=20, n_owa=2, n_cruise=0)
    ).spawn_wave()
    assert recommend_strategy(wave) == CUASStrategy.AREA_DEFENCE


def test_recommend_strategy_picks_layered_for_owa_heavy():
    wave = CUASDefenceScenario(
        wave_params=HostileWaveParams(n_fpv=2, n_owa=10, n_cruise=0)
    ).spawn_wave()
    assert recommend_strategy(wave) == CUASStrategy.LAYERED


def test_recommend_strategy_picks_mobile_cap_for_mixed():
    wave = CUASDefenceScenario(
        wave_params=HostileWaveParams(n_fpv=4, n_owa=4, n_cruise=2)
    ).spawn_wave()
    assert recommend_strategy(wave) == CUASStrategy.MOBILE_CAP


def test_cost_exchange_zero_when_no_kills():
    scn = _scenario()
    wave = scn.spawn_wave()
    # no kills yet → ratio is 0
    assert scn.cost_exchange_ratio(wave, interceptor_cost_usd_per_kill=2100.0) == 0.0


def test_cost_exchange_positive_when_kills_recorded():
    scn = _scenario()
    wave = scn.spawn_wave()
    # manually mark one OWA killed (NOT impacted) — simulate defender intercept
    for h in wave.hostiles:
        if h.threat_class == ThreatClass.OWA:
            h.alive = False
            h.impacted_asset = None  # killed by defender, not impacted
            break
    ratio = scn.cost_exchange_ratio(wave, interceptor_cost_usd_per_kill=2_100.0)
    assert ratio > 0.0
    # $35K OWA killed by $2100 interceptor → ratio ~16.7
    assert ratio > 10.0
