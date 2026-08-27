"""Four C-UAS posture strategies for the defender swarm.

Each strategy is a pure function: (assets, wave, n_defenders) → goal positions
(N_defenders, 3). The downstream coordination layer (TWSL trust + auction
allocation) decides which defender takes which goal.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from ..scenarios.c_uas_defence import (
    CriticalInfrastructureAsset,
    HostileWave,
)


class CUASStrategy(Enum):
    """Named C-UAS posture strategy."""

    LAYERED = "layered"
    POINT_DEFENCE = "point_defence"
    AREA_DEFENCE = "area_defence"
    MOBILE_CAP = "mobile_cap"


def layered_defence_goals(
    assets: list[CriticalInfrastructureAsset],
    wave: HostileWave,
    n_defenders: int,
    *,
    inner_radius_m: float = 1_500.0,
    middle_radius_m: float = 5_000.0,
    outer_radius_m: float = 12_000.0,
) -> np.ndarray:
    """Layered concentric defence around primary asset.

    Defenders split 30% inner / 40% middle / 30% outer rings.
    """
    asset = assets[0]
    n_inner = max(1, int(0.3 * n_defenders))
    n_outer = max(1, int(0.3 * n_defenders))
    n_middle = n_defenders - n_inner - n_outer

    az_lo, az_hi = wave.params.azimuth_sector_deg
    # bias defender placement toward the hostile-wave azimuth sector,
    # but cover ±30° margin
    az_lo_def = az_lo - 30.0
    az_hi_def = az_hi + 30.0

    goals: list[np.ndarray] = []
    for n_ring, r in [
        (n_outer, outer_radius_m),
        (n_middle, middle_radius_m),
        (n_inner, inner_radius_m),
    ]:
        for i in range(n_ring):
            t = (i + 0.5) / max(n_ring, 1)
            az_deg = az_lo_def + t * (az_hi_def - az_lo_def)
            az_rad = np.deg2rad(az_deg)
            altitude = float(asset.position_enu_m[2]) + 120.0
            offset = np.array([np.cos(az_rad) * r, np.sin(az_rad) * r, 0.0])
            goal = asset.position_enu_m + offset
            goal[2] = altitude
            goals.append(goal)
    return np.stack(goals[:n_defenders])


def point_defence_goals(
    assets: list[CriticalInfrastructureAsset],
    wave: HostileWave,
    n_defenders: int,
    *,
    bubble_radius_m: float = 800.0,
) -> np.ndarray:
    """Tight bubble immediately around primary asset.

    All defenders orbit at bubble_radius_m. Accept that outer engagement is
    limited; rely on last-ring kill probability. Useful when interceptor speed
    is low or the threat class is too fast for outer engagement (cruise).
    """
    asset = assets[0]
    altitude = float(asset.position_enu_m[2]) + 100.0
    goals: list[np.ndarray] = []
    for i in range(n_defenders):
        az_rad = 2.0 * np.pi * (i + 0.5) / max(n_defenders, 1)
        offset = np.array([np.cos(az_rad) * bubble_radius_m, np.sin(az_rad) * bubble_radius_m, 0.0])
        goal = asset.position_enu_m + offset
        goal[2] = altitude
        goals.append(goal)
    return np.stack(goals)


def area_defence_goals(
    assets: list[CriticalInfrastructureAsset],
    wave: HostileWave,
    n_defenders: int,
    *,
    spread_radius_m: float = 8_000.0,
) -> np.ndarray:
    """Broad area defence across the hostile-wave azimuth sector.

    Defenders spread across an expanded azimuth window at intermediate range.
    Optimised vs FPV mass where engagement-on-first-detection is critical.
    """
    asset = assets[0]
    altitude = float(asset.position_enu_m[2]) + 150.0
    az_lo, az_hi = wave.params.azimuth_sector_deg
    # broaden sector by ±45° for area defence
    az_lo_def = az_lo - 45.0
    az_hi_def = az_hi + 45.0
    goals: list[np.ndarray] = []
    for i in range(n_defenders):
        t = (i + 0.5) / max(n_defenders, 1)
        az_deg = az_lo_def + t * (az_hi_def - az_lo_def)
        az_rad = np.deg2rad(az_deg)
        # alternate inner/outer for layered coverage even inside the sector
        r = spread_radius_m * (0.7 if i % 2 == 0 else 1.0)
        offset = np.array([np.cos(az_rad) * r, np.sin(az_rad) * r, 0.0])
        goal = asset.position_enu_m + offset
        goal[2] = altitude
        goals.append(goal)
    return np.stack(goals)


def mobile_cap_goals(
    assets: list[CriticalInfrastructureAsset],
    wave: HostileWave,
    n_defenders: int,
    *,
    orbit_radius_m: float = 4_000.0,
    orbit_phase_offset_rad: float = 0.0,
) -> np.ndarray:
    """Combat Air Patrol — orbit at intermediate radius, converge on cue.

    Defenders distribute around a full-360° orbit at orbit_radius_m. The
    coordination layer dynamically reassigns when a hostile is detected.

    Useful when threat azimuth is uncertain (unknown-axis wave) or when
    Akashteer cueing is expected to drive last-mile engagement.
    """
    asset = assets[0]
    altitude = float(asset.position_enu_m[2]) + 130.0
    goals: list[np.ndarray] = []
    for i in range(n_defenders):
        az_rad = 2.0 * np.pi * (i / max(n_defenders, 1)) + orbit_phase_offset_rad
        offset = np.array([np.cos(az_rad) * orbit_radius_m, np.sin(az_rad) * orbit_radius_m, 0.0])
        goal = asset.position_enu_m + offset
        goal[2] = altitude
        goals.append(goal)
    return np.stack(goals)


# Strategy → function dispatch table
STRATEGY_DISPATCH = {
    CUASStrategy.LAYERED: layered_defence_goals,
    CUASStrategy.POINT_DEFENCE: point_defence_goals,
    CUASStrategy.AREA_DEFENCE: area_defence_goals,
    CUASStrategy.MOBILE_CAP: mobile_cap_goals,
}


def plan_goals_for_strategy(
    strategy: CUASStrategy,
    assets: list[CriticalInfrastructureAsset],
    wave: HostileWave,
    n_defenders: int,
) -> np.ndarray:
    """Dispatch to the named strategy + return per-drone goal positions."""
    fn = STRATEGY_DISPATCH[strategy]
    return fn(assets, wave, n_defenders)


def recommend_strategy(wave: HostileWave) -> CUASStrategy:
    """Recommend a strategy based on the dominant threat class in the wave.

    Rules:
      - cruise > 25% of wave → POINT_DEFENCE (cruise is too fast for outer)
      - FPV > 50% of wave    → AREA_DEFENCE  (FPV mass needs dispersion)
      - OWA > 50% of wave    → LAYERED       (OWA cost-exchange wants tiered)
      - else                  → MOBILE_CAP   (mixed wave, react dynamically)
    """
    total = max(wave.params.total_hostiles, 1)
    fpv_pct = wave.params.n_fpv / total
    owa_pct = wave.params.n_owa / total
    cruise_pct = wave.params.n_cruise / total

    if cruise_pct > 0.25:
        return CUASStrategy.POINT_DEFENCE
    if fpv_pct > 0.5:
        return CUASStrategy.AREA_DEFENCE
    if owa_pct > 0.5:
        return CUASStrategy.LAYERED
    return CUASStrategy.MOBILE_CAP
