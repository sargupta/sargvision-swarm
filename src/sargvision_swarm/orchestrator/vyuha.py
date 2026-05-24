"""VYUHA — multi-strategy defensive formation library for Operation Trishul.

Real cross-border drone defence is not a single algorithm — it's a family of
strategies (CENTRAL / DISTRIBUTED / LAYERED / CAP) whose choice depends on
intel confidence, adversary doctrine, magazine economics, and time horizon.
See `AI_Workspace/drone_swarm_research/41_TRISHUL_REAL_WORLD_DOCTRINE.md` for
the operational analysis.

This module owns:
  1. Spawn-time placement of friendly drones per strategy
  2. Per-drone *sector / element* assignment (which HVT a drone belongs to,
     or which ring in the layered defence it occupies)
  3. Per-HVT metrics (time-to-first-impact, intercept lag, survival %)

The downstream SHIELD / VAJRA / MAYA stacks are unchanged — they consume the
priority matrix; only the placement + sector tasking changes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

VyuhaStrategy = Literal[
    # Trishul (border_strike — multi-HVT defence)
    "central",
    "distributed",
    "layered",
    "cap",
    # Coverage (counter-swarm — single protected zone)
    "ring_uniform",
    "azimuth_weighted",
    "layered_intercept",
    "flying_cap",
    # SEAD ingress (offensive penetration of hostile IADS)
    "geodesic_direct",
    "decoy_mass",
    "wild_weasel",
    "low_observable",
    # Migration (multi-corridor traversal)
    "load_balanced",
    "fastest_corridor",
    "safest_corridor",
    "adaptive_reroute",
]


# Per-scenario strategy menu — drives the TopBar selector
SCENARIO_STRATEGIES: dict[str, list[VyuhaStrategy]] = {
    "border_strike": ["central", "distributed", "layered", "cap"],
    "coverage": ["ring_uniform", "azimuth_weighted", "layered_intercept", "flying_cap"],
    "sead_ingress": ["geodesic_direct", "decoy_mass", "wild_weasel", "low_observable"],
    "migration": ["load_balanced", "fastest_corridor", "safest_corridor", "adaptive_reroute"],
}

# ── Per-HVT outcome metrics ──────────────────────────────────────────


@dataclass
class HVTMetrics:
    """Per-HVT operational outcomes tracked across the engagement window."""

    hvt_id: str
    first_detect_t: float | None = None  # when first hostile entered its threat ring
    first_hit_t: float | None = None  # when the HVT first took damage
    n_intercepts_in_sector: int = 0  # interceptors vectored at hostiles bound here
    n_kills_in_sector: int = 0  # kills credited to drones assigned this HVT
    n_drones_allocated: int = 0  # how many drones the strategy gave this HVT
    intercept_lag_samples: list[float] = field(default_factory=list)

    @property
    def mean_intercept_lag_s(self) -> float:
        if not self.intercept_lag_samples:
            return 0.0
        return float(sum(self.intercept_lag_samples) / len(self.intercept_lag_samples))


# ── Strategy placement specifications ────────────────────────────────


@dataclass
class DroneSpawnSpec:
    """One drone's strategy-assigned spawn point + sector membership."""

    pos: np.ndarray  # (3,) sim-local meters
    sector_hvt: str | None  # HVT id this drone is bound to defend, or None for free-roaming
    ring: Literal["outer", "middle", "inner", "central"] = "central"


@dataclass
class SeadRoleSpec:
    """Per-drone SEAD role + target override for ingress strategies."""

    role: Literal["strike", "decoy", "weasel"] = "strike"
    # Override target position (e.g. WILD-WEASEL points drones at SAMs first)
    override_target: np.ndarray | None = None
    # Wave start time — DECOY-MASS uses this to delay the strike package
    release_delay_s: float = 0.0


@dataclass
class VyuhaPlan:
    """Output of a strategy's placement function."""

    strategy: VyuhaStrategy
    spawns: list[DroneSpawnSpec]
    sector_drone_ids: dict[str, list[int]] = field(default_factory=dict)
    metrics: dict[str, HVTMetrics] = field(default_factory=dict)
    # SEAD scenario only — per-drone role + override target
    sead_roles: dict[int, SeadRoleSpec] = field(default_factory=dict)
    # Per-strategy override of the Riemannian metric (β, γ).
    # `None` means: use the scenario default.
    metric_override: tuple[float, float] | None = None

    def init_metrics(self, hvt_ids: list[str]) -> None:
        for hid in hvt_ids:
            allocated = len(self.sector_drone_ids.get(hid, []))
            self.metrics[hid] = HVTMetrics(hvt_id=hid, n_drones_allocated=allocated)


# ── Strategy 1: CENTRAL (current naive baseline) ─────────────────────


def plan_central(n_drones: int, hvt_positions: dict[str, np.ndarray]) -> VyuhaPlan:
    """All drones at one rally point between LoC and HVTs.
    Failure mode: outer HVTs lose because flight time > attack time."""
    centre = np.array([0.0, -2.0, 6.0])
    spawns: list[DroneSpawnSpec] = []
    # Tight ring around the rally point
    ring_radius = max(3.0, 0.3 * n_drones**0.5)
    for i in range(n_drones):
        theta = 2 * np.pi * i / n_drones
        pos = centre + np.array([ring_radius * np.cos(theta), ring_radius * np.sin(theta), 0.0])
        spawns.append(DroneSpawnSpec(pos=pos, sector_hvt=None, ring="central"))
    plan = VyuhaPlan(strategy="central", spawns=spawns)
    plan.init_metrics(list(hvt_positions.keys()))
    return plan


# ── Strategy 2: DISTRIBUTED (pre-positioned per HVT) ─────────────────


def plan_distributed(
    n_drones: int,
    hvt_positions: dict[str, np.ndarray],
    hvt_weights: dict[str, float] | None = None,
) -> VyuhaPlan:
    """Pre-position dedicated elements at each HVT proportional to value × P(attack).

    Default weights (per the Trishul scenario):
      LEH_AB   — military, high-value, central → 0.50
      KARU_PS  — energy, medium-high, eastern axis → 0.33
      DBO_FWD  — command, medium, western axis → 0.17
    """
    weights = hvt_weights or {"LEH_AB": 0.50, "KARU_PS": 0.33, "DBO_FWD": 0.17}
    # Normalise + clip to known HVTs
    valid = {k: w for k, w in weights.items() if k in hvt_positions}
    total = sum(valid.values()) or 1.0
    valid = {k: w / total for k, w in valid.items()}

    # Allocate per-HVT counts (integer rounding with remainder distribution)
    raw = {k: w * n_drones for k, w in valid.items()}
    counts = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = n_drones - sum(counts.values())
    # Hand out the remainder to HVTs with the highest fractional parts
    by_frac = sorted(((raw[k] - counts[k], k) for k in counts), reverse=True)
    for _, k in by_frac[:remainder]:
        counts[k] += 1

    spawns: list[DroneSpawnSpec] = []
    sector_ids: dict[str, list[int]] = {k: [] for k in valid}
    next_id = 0
    for hvt_id, n in counts.items():
        if n == 0:
            continue
        centre = hvt_positions[hvt_id].astype(float).copy()
        centre[2] = 6.0
        # Tight orbit around the HVT
        r = max(1.5, 0.4 * n**0.5)
        for i in range(n):
            theta = 2 * np.pi * i / max(n, 1)
            pos = centre + np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
            spawns.append(DroneSpawnSpec(pos=pos, sector_hvt=hvt_id, ring="inner"))
            sector_ids[hvt_id].append(next_id)
            next_id += 1

    plan = VyuhaPlan(strategy="distributed", spawns=spawns, sector_drone_ids=sector_ids)
    plan.init_metrics(list(hvt_positions.keys()))
    return plan


# ── Strategy 3: LAYERED (3-ring outer/middle/inner) ──────────────────


def plan_layered(
    n_drones: int,
    hvt_positions: dict[str, np.ndarray],
    loc_y: float = 18.0,
) -> VyuhaPlan:
    """Three-ring layered defence:
      OUTER (LoC-adjacent, recon + decoy filter): 25% of N
      MIDDLE (between LoC and HVTs, primary intercept): 50% of N
      INNER (point defence at each HVT, last-ditch): 25% of N split per HVT

    Decoy filtering happens at OUTER via SHIELD posterior; only kinetics get
    forwarded to MIDDLE for engagement. INNER engages only if a hostile
    penetrates within 3× impact_radius of its HVT.
    """
    n_outer = max(4, int(0.25 * n_drones))
    n_inner = max(len(hvt_positions) * 2, int(0.25 * n_drones))
    n_middle = max(1, n_drones - n_outer - n_inner)

    spawns: list[DroneSpawnSpec] = []
    sector_ids: dict[str, list[int]] = {k: [] for k in hvt_positions}
    next_id = 0

    # OUTER ring — evenly spaced across the LoC line at y ≈ loc_y - 6 (sensor standoff)
    outer_y = loc_y - 6.0
    xs_outer = np.linspace(-22.0, 22.0, n_outer)
    for x in xs_outer:
        spawns.append(
            DroneSpawnSpec(
                pos=np.array([x, outer_y, 6.0]),
                sector_hvt=None,
                ring="outer",
            )
        )
        next_id += 1

    # MIDDLE ring — primary intercept mass between LoC and HVTs at y ≈ -2
    middle_y = -2.0
    xs_middle = np.linspace(-15.0, 15.0, n_middle)
    for x in xs_middle:
        spawns.append(
            DroneSpawnSpec(
                pos=np.array([x, middle_y, 6.0]),
                sector_hvt=None,
                ring="middle",
            )
        )
        next_id += 1

    # INNER ring — point defence at each HVT, n_inner split evenly
    inner_per_hvt = max(2, n_inner // max(len(hvt_positions), 1))
    for hvt_id, hvt_pos in hvt_positions.items():
        centre = hvt_pos.astype(float).copy()
        centre[2] = 6.0
        r = 2.0
        for i in range(inner_per_hvt):
            theta = 2 * np.pi * i / inner_per_hvt
            pos = centre + np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
            spawns.append(DroneSpawnSpec(pos=pos, sector_hvt=hvt_id, ring="inner"))
            sector_ids[hvt_id].append(next_id)
            next_id += 1

    plan = VyuhaPlan(strategy="layered", spawns=spawns, sector_drone_ids=sector_ids)
    plan.init_metrics(list(hvt_positions.keys()))
    return plan


# ── Strategy 4: CAP (racetrack patrol) ───────────────────────────────


def plan_cap(
    n_drones: int,
    hvt_positions: dict[str, np.ndarray],
) -> VyuhaPlan:
    """Combat-air-patrol racetrack orbits between HVT pairs.

    For each pair of HVTs, allocate ~N/(n_pairs) drones cycling on a racetrack
    between them. Spawn positions are along the racetrack midline; the
    downstream reflex layer drives them along the orbit.
    """
    hvt_ids = list(hvt_positions.keys())
    pairs: list[tuple[str, str]] = []
    for i in range(len(hvt_ids)):
        for j in range(i + 1, len(hvt_ids)):
            pairs.append((hvt_ids[i], hvt_ids[j]))
    if not pairs:
        return plan_central(n_drones, hvt_positions)

    spawns: list[DroneSpawnSpec] = []
    sector_ids: dict[str, list[int]] = {k: [] for k in hvt_positions}
    per_pair = max(2, n_drones // len(pairs))
    next_id = 0
    remaining = n_drones
    for a, b in pairs:
        n_this = min(per_pair, remaining)
        if n_this == 0:
            break
        remaining -= n_this
        pa = hvt_positions[a].astype(float).copy()
        pb = hvt_positions[b].astype(float).copy()
        pa[2] = pb[2] = 6.0
        # Spawn drones evenly spaced along the line between A and B
        for i in range(n_this):
            t = (i + 0.5) / n_this
            pos = pa + (pb - pa) * t
            # Drones nearer A are tasked to A; nearer B to B
            sector = a if t < 0.5 else b
            spawns.append(DroneSpawnSpec(pos=pos, sector_hvt=sector, ring="middle"))
            sector_ids[sector].append(next_id)
            next_id += 1

    # Distribute any remainder across the first pair's drones at midpoint
    while remaining > 0:
        a, b = pairs[0]
        pa = hvt_positions[a].astype(float).copy()
        pb = hvt_positions[b].astype(float).copy()
        pa[2] = pb[2] = 6.0
        pos = (pa + pb) / 2.0
        spawns.append(DroneSpawnSpec(pos=pos, sector_hvt=a, ring="middle"))
        sector_ids[a].append(next_id)
        next_id += 1
        remaining -= 1

    plan = VyuhaPlan(strategy="cap", spawns=spawns, sector_drone_ids=sector_ids)
    plan.init_metrics(list(hvt_positions.keys()))
    return plan


# ── Coverage strategies — single protected zone instead of multi-HVT ─


COVERAGE_ZONE_ID = "ZONE"  # synthetic single-HVT id for coverage scenario


def _coverage_zone_dict(centre: np.ndarray) -> dict[str, np.ndarray]:
    """Wrap the protected zone as a single-HVT dict so the existing
    multi-HVT planners can be reused for coverage."""
    return {COVERAGE_ZONE_ID: centre}


def plan_ring_uniform(n_drones: int, centre: np.ndarray, radius: float = 7.0) -> VyuhaPlan:
    """Even ring around the protected zone — the current default."""
    spawns: list[DroneSpawnSpec] = []
    for i in range(n_drones):
        theta = 2 * np.pi * i / max(n_drones, 1)
        pos = centre + np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0])
        spawns.append(DroneSpawnSpec(pos=pos, sector_hvt=COVERAGE_ZONE_ID, ring="middle"))
    plan = VyuhaPlan(strategy="ring_uniform", spawns=spawns)
    plan.sector_drone_ids[COVERAGE_ZONE_ID] = list(range(n_drones))
    plan.init_metrics([COVERAGE_ZONE_ID])
    return plan


def plan_azimuth_weighted(
    n_drones: int,
    centre: np.ndarray,
    radius: float = 7.0,
    threat_bearing_rad: float = 0.0,
    concentration: float = 1.6,  # von-Mises κ; higher = tighter cluster on the bearing
) -> VyuhaPlan:
    """Heavier defence on the detected threat bearing — von-Mises spaced.

    Without intel, threat_bearing_rad defaults to 0 (east) — caller should
    override based on current detections.
    """
    spawns: list[DroneSpawnSpec] = []
    # von-Mises-spaced angles around the threat bearing
    # Equivalent of choosing N angles with density proportional to exp(κ cos(θ-μ))
    # Use inverse-CDF trick on a coarse grid for determinism
    grid = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    density = np.exp(concentration * np.cos(grid - threat_bearing_rad))
    cumdens = np.cumsum(density)
    cumdens /= cumdens[-1]
    quantiles = (np.arange(n_drones) + 0.5) / n_drones
    angles = np.array([grid[np.searchsorted(cumdens, q)] for q in quantiles])
    for i, theta in enumerate(angles):
        pos = centre + np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0])
        spawns.append(DroneSpawnSpec(pos=pos, sector_hvt=COVERAGE_ZONE_ID, ring="middle"))
    plan = VyuhaPlan(strategy="azimuth_weighted", spawns=spawns)
    plan.sector_drone_ids[COVERAGE_ZONE_ID] = list(range(n_drones))
    plan.init_metrics([COVERAGE_ZONE_ID])
    return plan


def plan_layered_intercept(n_drones: int, centre: np.ndarray) -> VyuhaPlan:
    """3-ring layered defence around the protected zone."""
    n_outer = max(4, int(0.25 * n_drones))
    n_inner = max(2, int(0.25 * n_drones))
    n_middle = max(1, n_drones - n_outer - n_inner)

    spawns: list[DroneSpawnSpec] = []
    for k, (count, r, ring) in enumerate(
        [
            (n_outer, 12.0, "outer"),
            (n_middle, 8.0, "middle"),
            (n_inner, 3.0, "inner"),
        ]
    ):
        for i in range(count):
            theta = 2 * np.pi * i / max(count, 1)
            pos = centre + np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
            sec = COVERAGE_ZONE_ID if ring == "inner" else None
            spawns.append(DroneSpawnSpec(pos=pos, sector_hvt=sec, ring=ring))

    plan = VyuhaPlan(strategy="layered_intercept", spawns=spawns)
    # Only the inner ring is sector-bound; outer + middle are free-roaming
    plan.sector_drone_ids[COVERAGE_ZONE_ID] = [
        i for i, s in enumerate(spawns) if s.sector_hvt == COVERAGE_ZONE_ID
    ]
    plan.init_metrics([COVERAGE_ZONE_ID])
    return plan


def plan_flying_cap(n_drones: int, centre: np.ndarray, radius: float = 8.0) -> VyuhaPlan:
    """Combat-air-patrol racetrack around the protected zone.

    The reflex layer is responsible for orbital following — we just spawn
    drones spaced along the racetrack midline and let the reflex drive them
    around. Each drone is free-roaming so it can break for an intercept.
    """
    spawns: list[DroneSpawnSpec] = []
    for i in range(n_drones):
        # Distribute around the orbit so the swarm is balanced
        theta = 2 * np.pi * i / max(n_drones, 1)
        pos = centre + np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0])
        # Free-roaming so any drone can engage any hostile
        spawns.append(DroneSpawnSpec(pos=pos, sector_hvt=None, ring="middle"))
    plan = VyuhaPlan(strategy="flying_cap", spawns=spawns)
    plan.init_metrics([COVERAGE_ZONE_ID])
    return plan


# ── SEAD ingress strategies — modulate role/target/metric, not spawn ─


def _spawns_for_sead(n: int) -> list[DroneSpawnSpec]:
    """Standard SEAD spawn line — friendlies along the south edge y=-32."""
    xs = np.linspace(-12.0, 12.0, n)
    return [
        DroneSpawnSpec(
            pos=np.array([float(xs[i]), -32.0 + 0.0 * i, 6.0]),
            sector_hvt=None,
            ring="central",
        )
        for i in range(n)
    ]


def plan_geodesic_direct(n_drones: int, sam_positions: list[np.ndarray]) -> VyuhaPlan:
    """All N drones plan the same Riemannian geodesic. The default β/γ."""
    plan = VyuhaPlan(
        strategy="geodesic_direct",
        spawns=_spawns_for_sead(n_drones),
        metric_override=(12.0, 3.0),
    )
    # No per-drone role overrides — everyone is "strike".
    return plan


def plan_decoy_mass(
    n_drones: int,
    sam_positions: list[np.ndarray],
    decoy_fraction: float = 0.4,
    strike_delay_s: float = 8.0,
) -> VyuhaPlan:
    """First wave (40%) = decoys flying direct through IADS centre to draw
    fire. Second wave (60%) = strike package, delayed by 8s, takes the
    clean geodesic now that radars are committed."""
    plan = VyuhaPlan(
        strategy="decoy_mass",
        spawns=_spawns_for_sead(n_drones),
        metric_override=(12.0, 3.0),
    )
    n_decoy = max(1, int(decoy_fraction * n_drones))
    # First n_decoy drones are decoys — release immediately, fly straight
    # through (override_target=None → keep north-side target, but a flatter
    # metric β=1.5 makes their path a near-straight line)
    for i in range(n_decoy):
        plan.sead_roles[i] = SeadRoleSpec(
            role="decoy",
            release_delay_s=0.0,
        )
    # Remaining drones are strike package, delayed
    for i in range(n_decoy, n_drones):
        plan.sead_roles[i] = SeadRoleSpec(
            role="strike",
            release_delay_s=strike_delay_s,
        )
    return plan


def plan_wild_weasel(
    n_drones: int,
    sam_positions: list[np.ndarray],
    weasel_fraction: float = 0.2,
) -> VyuhaPlan:
    """20% SEAD-specialised drones target the SAMs themselves. Once a SAM
    is hit, the field becomes dirty → CHANAKYA replans for the 80% strike
    package through the now-safe corridor."""
    plan = VyuhaPlan(
        strategy="wild_weasel",
        spawns=_spawns_for_sead(n_drones),
        metric_override=(12.0, 3.0),
    )
    n_weasel = max(len(sam_positions), int(weasel_fraction * n_drones))
    # Assign each weasel to a SAM in round-robin
    for i in range(min(n_weasel, n_drones)):
        sam = sam_positions[i % max(len(sam_positions), 1)] if sam_positions else None
        plan.sead_roles[i] = SeadRoleSpec(
            role="weasel",
            override_target=sam.copy() if sam is not None else None,
            release_delay_s=0.0,
        )
    # Strike package follows after weasels suppress
    for i in range(n_weasel, n_drones):
        plan.sead_roles[i] = SeadRoleSpec(
            role="strike",
            release_delay_s=6.0,
        )
    return plan


def plan_low_observable(n_drones: int, sam_positions: list[np.ndarray]) -> VyuhaPlan:
    """Bump β/γ in the Riemannian metric so geodesics hug the threat-field
    zero contour — drones fly at edge of detection rather than centre of gap."""
    return VyuhaPlan(
        strategy="low_observable",
        spawns=_spawns_for_sead(n_drones),
        metric_override=(25.0, 4.0),  # sharper conformal factor → tighter detours
    )


# ── Migration strategies — modulate corridor allocation ──────────────


@dataclass
class MigrationCorridorPlan:
    """Per-drone corridor assignment for the migration scenario.

    Stored alongside the VyuhaPlan so the migration_field logic knows how to
    initial-assign drones to corridor zones.
    """

    corridor_choice: dict[int, str] = field(default_factory=dict)  # drone_id → zone_id
    rebalance_on_hazard: bool = False


def plan_load_balanced(n_drones: int) -> VyuhaPlan:
    """Current default — drones distributed proportional to corridor capacity.

    Migration field's `initial_assignments(n)` already does this; we just tag
    the plan so the scenario knows to keep the existing assignment.
    """
    return VyuhaPlan(
        strategy="load_balanced",
        spawns=[],  # migration spawns are managed by the scenario init itself
    )


def plan_fastest_corridor(n_drones: int) -> VyuhaPlan:
    """All drones go through the fastest pass (Khardung La in Ladakh default).
    Concentration risk: if the fastest pass closes, all drones must rebalance."""
    plan = VyuhaPlan(strategy="fastest_corridor", spawns=[])
    plan.metric_override = None
    return plan


def plan_safest_corridor(n_drones: int) -> VyuhaPlan:
    """All drones go through the lowest-hazard pass (Tanglang La default)."""
    return VyuhaPlan(strategy="safest_corridor", spawns=[])


def plan_adaptive_reroute(n_drones: int) -> VyuhaPlan:
    """Drones initial-assigned load-balanced, but reroute live when hazards
    update. Uses the migration_field's `closure_events` to trigger rebalance."""
    return VyuhaPlan(strategy="adaptive_reroute", spawns=[])


# ── Public dispatcher ────────────────────────────────────────────────


def plan_for_strategy(
    strategy: VyuhaStrategy,
    n_drones: int,
    hvt_positions: dict[str, np.ndarray],
    threat_bearing_rad: float = 0.0,
) -> VyuhaPlan:
    # Trishul (multi-HVT)
    if strategy == "central":
        return plan_central(n_drones, hvt_positions)
    if strategy == "distributed":
        return plan_distributed(n_drones, hvt_positions)
    if strategy == "layered":
        return plan_layered(n_drones, hvt_positions)
    if strategy == "cap":
        return plan_cap(n_drones, hvt_positions)
    # Coverage (single protected zone — pass centre as the only HVT)
    if strategy in ("ring_uniform", "azimuth_weighted", "layered_intercept", "flying_cap"):
        centre = next(iter(hvt_positions.values()))  # caller passes a single-entry dict
        if strategy == "ring_uniform":
            return plan_ring_uniform(n_drones, centre)
        if strategy == "azimuth_weighted":
            return plan_azimuth_weighted(n_drones, centre, threat_bearing_rad=threat_bearing_rad)
        if strategy == "layered_intercept":
            return plan_layered_intercept(n_drones, centre)
        if strategy == "flying_cap":
            return plan_flying_cap(n_drones, centre)
    # SEAD ingress — hvt_positions is the SAM/radar layout (each is a defense asset)
    if strategy in ("geodesic_direct", "decoy_mass", "wild_weasel", "low_observable"):
        sam_positions = list(hvt_positions.values())
        if strategy == "geodesic_direct":
            return plan_geodesic_direct(n_drones, sam_positions)
        if strategy == "decoy_mass":
            return plan_decoy_mass(n_drones, sam_positions)
        if strategy == "wild_weasel":
            return plan_wild_weasel(n_drones, sam_positions)
        if strategy == "low_observable":
            return plan_low_observable(n_drones, sam_positions)
    # Migration — corridor routing strategies (no spawn override)
    if strategy in ("load_balanced", "fastest_corridor", "safest_corridor", "adaptive_reroute"):
        if strategy == "load_balanced":
            return plan_load_balanced(n_drones)
        if strategy == "fastest_corridor":
            return plan_fastest_corridor(n_drones)
        if strategy == "safest_corridor":
            return plan_safest_corridor(n_drones)
        if strategy == "adaptive_reroute":
            return plan_adaptive_reroute(n_drones)
    raise ValueError(f"unknown vyuha strategy: {strategy}")


# ── Sector-aware intercept filtering ─────────────────────────────────


def filter_priorities_by_sector(
    priority_matrix: np.ndarray,
    friendly_sector: list[str | None],
    hostile_target_hvt: list[str | None],
) -> np.ndarray:
    """Zero out bid entries where a sector-bound drone is being asked to engage
    a hostile bound for a different HVT. Free-roaming drones (sector_hvt=None)
    can bid on anything.

    For LAYERED strategy, also zeroes inner-ring drones unless the hostile is
    within their HVT's terminal phase (handled by the caller).
    """
    E = priority_matrix.copy()
    for i, fsec in enumerate(friendly_sector):
        if fsec is None:
            continue  # free-roaming: keep all bids
        for j, hsec in enumerate(hostile_target_hvt):
            if hsec is not None and hsec != fsec:
                E[i, j] = 0.0
    return E
