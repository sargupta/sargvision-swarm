"""C-UAS Defence scenario.

The defender swarm protects one or more Critical Infrastructure Assets (CIAs)
against a mixed-threat hostile drone wave.

Civilian framing: this scenario applies equally to military bases, oil
refineries, power plants, airports, dams, satellite ground stations, embassies,
and high-value civilian gatherings. The coordination middleware (CFM) does not
discriminate between these — the asset is just a fixed point + value to defend.

Threat model: each hostile drone is one of three threat classes:

  - FPV (first-person-view racer / kamikaze) — small RCS, short range,
    low-altitude, ~$300-800 unit cost
  - OWA (one-way attack) — Shahed-136-class long-endurance loiter,
    100-200 km/h cruise, ~$20-50K unit cost
  - CRUISE (subsonic cruise / glide) — fast, low-altitude, sub-50K cost

The defender swarm holds N interceptor drones and may engage hostiles via
direct kinetic interception. Cost-exchange is a first-class metric.

Scenario references the C-UAS coordination problem statement in
file 16 (`16_indian_service_requirements.md`) and file 11 (think-tank
consensus on C-UAS as #1 unsolved). No reference to specific historical
operations or geographic locations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ThreatClass(Enum):
    """Threat category for incoming hostile drone."""

    FPV = "fpv"
    OWA = "owa"
    CRUISE = "cruise"


@dataclass(frozen=True)
class HostileWaveParams:
    """Parameters governing a single hostile drone wave."""

    n_fpv: int = 12
    n_owa: int = 8
    n_cruise: int = 4

    #: Azimuth sector (degrees) around the asset from which the wave originates.
    azimuth_sector_deg: tuple[float, float] = (45.0, 135.0)

    #: Wave range at start (metres from asset).
    start_range_m: float = 25_000.0

    #: Per-class cruise speeds (m/s).
    fpv_speed_mps: float = 25.0
    owa_speed_mps: float = 50.0
    cruise_speed_mps: float = 250.0

    #: Per-class unit costs (USD, used for cost-exchange metrics).
    fpv_cost_usd: float = 500.0
    owa_cost_usd: float = 35_000.0
    cruise_cost_usd: float = 80_000.0

    rng_seed: int = 0

    @property
    def total_hostiles(self) -> int:
        return self.n_fpv + self.n_owa + self.n_cruise

    @property
    def total_cost_usd(self) -> float:
        return (
            self.n_fpv * self.fpv_cost_usd
            + self.n_owa * self.owa_cost_usd
            + self.n_cruise * self.cruise_cost_usd
        )


@dataclass
class CriticalInfrastructureAsset:
    """A protected asset.

    Attributes
    ----------
    asset_id : str
        Identifier (generic — "CIA-01", "CIA-02"; do not embed real names).
    position_enu_m : np.ndarray (3,)
        Local-tangent ENU position. Defaults to origin.
    value_usd : float
        Asset replacement value in USD — used for cost-exchange ROI.
    radius_m : float
        Protection radius — outside this, hostile is considered "leaked".
    """

    asset_id: str = "CIA-01"
    position_enu_m: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 50.0])
    )
    value_usd: float = 10_000_000.0  # default $10M asset
    radius_m: float = 500.0


@dataclass
class HostileEntity:
    """A single hostile drone in the wave."""

    hostile_id: str
    threat_class: ThreatClass
    position_enu_m: np.ndarray
    velocity_mps: np.ndarray
    cost_usd: float
    alive: bool = True
    impacted_asset: str | None = None  # asset_id if it reached the asset


@dataclass
class HostileWave:
    """A spawned hostile wave."""

    params: HostileWaveParams
    hostiles: list[HostileEntity]

    @property
    def alive_count(self) -> int:
        return sum(1 for h in self.hostiles if h.alive)

    @property
    def by_class(self) -> dict[ThreatClass, int]:
        counts: dict[ThreatClass, int] = {tc: 0 for tc in ThreatClass}
        for h in self.hostiles:
            if h.alive:
                counts[h.threat_class] += 1
        return counts


@dataclass
class CUASDefenceScenario:
    """A C-UAS Defence scenario instance.

    Holds the defended asset(s) + parameters for spawning hostile waves.
    """

    assets: list[CriticalInfrastructureAsset] = field(default_factory=list)
    wave_params: HostileWaveParams = field(default_factory=HostileWaveParams)

    def __post_init__(self) -> None:
        if not self.assets:
            self.assets = [CriticalInfrastructureAsset()]

    def spawn_wave(self, target_asset_id: str | None = None) -> HostileWave:
        """Spawn a hostile wave aimed at the named asset (or first asset)."""
        target = self.assets[0]
        if target_asset_id is not None:
            for a in self.assets:
                if a.asset_id == target_asset_id:
                    target = a
                    break
        rng = np.random.default_rng(self.wave_params.rng_seed)
        hostiles: list[HostileEntity] = []
        p = self.wave_params

        az_lo, az_hi = p.azimuth_sector_deg

        def _spawn_one(idx: int, tc: ThreatClass, speed: float, cost: float) -> HostileEntity:
            az_deg = rng.uniform(az_lo, az_hi)
            az_rad = np.deg2rad(az_deg)
            # spawn at start_range_m from the target asset
            offset = p.start_range_m * np.array(
                [np.cos(az_rad), np.sin(az_rad), 0.0]
            )
            pos = target.position_enu_m + offset
            # vary altitude by class
            if tc is ThreatClass.FPV:
                pos[2] = float(target.position_enu_m[2] + rng.uniform(20, 150))
            elif tc is ThreatClass.OWA:
                pos[2] = float(target.position_enu_m[2] + rng.uniform(80, 500))
            else:
                pos[2] = float(target.position_enu_m[2] + rng.uniform(40, 200))
            # velocity points from spawn back toward target asset
            to_target = target.position_enu_m - pos
            dist = float(np.linalg.norm(to_target))
            if dist > 0:
                vel = (to_target / dist) * speed
            else:
                vel = np.zeros(3)
            return HostileEntity(
                hostile_id=f"H-{tc.value.upper()}-{idx:03d}",
                threat_class=tc,
                position_enu_m=pos,
                velocity_mps=vel,
                cost_usd=cost,
            )

        idx = 0
        for _ in range(p.n_fpv):
            hostiles.append(_spawn_one(idx, ThreatClass.FPV, p.fpv_speed_mps, p.fpv_cost_usd))
            idx += 1
        for _ in range(p.n_owa):
            hostiles.append(_spawn_one(idx, ThreatClass.OWA, p.owa_speed_mps, p.owa_cost_usd))
            idx += 1
        for _ in range(p.n_cruise):
            hostiles.append(
                _spawn_one(idx, ThreatClass.CRUISE, p.cruise_speed_mps, p.cruise_cost_usd)
            )
            idx += 1
        return HostileWave(params=p, hostiles=hostiles)

    def step_wave(self, wave: HostileWave, dt_s: float) -> dict:
        """Advance the wave one timestep. Returns a small stats dict.

        Hostiles that reach an asset's radius are marked impacted (alive=False,
        impacted_asset set).
        """
        impacts: dict[str, int] = {a.asset_id: 0 for a in self.assets}
        for h in wave.hostiles:
            if not h.alive:
                continue
            h.position_enu_m = h.position_enu_m + h.velocity_mps * dt_s
            # check impact against any asset
            for a in self.assets:
                d = float(np.linalg.norm(h.position_enu_m - a.position_enu_m))
                if d <= a.radius_m:
                    h.alive = False
                    h.impacted_asset = a.asset_id
                    impacts[a.asset_id] += 1
                    break
        return {
            "alive": wave.alive_count,
            "impacts": impacts,
            "by_class": {tc.value: n for tc, n in wave.by_class.items()},
        }

    def cost_exchange_ratio(
        self, wave: HostileWave, interceptor_cost_usd_per_kill: float
    ) -> float:
        """Compute defender-favourable cost-exchange ratio.

        Returns hostile-cost-killed / interceptor-cost-expended. Higher is
        better. Below 1.0 means defender is losing the economic battle.
        """
        killed_cost = sum(
            h.cost_usd for h in wave.hostiles if not h.alive and h.impacted_asset is None
        )
        kills = sum(
            1 for h in wave.hostiles if not h.alive and h.impacted_asset is None
        )
        if kills == 0:
            return 0.0
        expended = kills * interceptor_cost_usd_per_kill
        return killed_cost / max(expended, 1e-9)
