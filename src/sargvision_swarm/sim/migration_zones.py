"""Governed Migration zones — multi-corridor traversal with capacity + hazards.

Inspired by the Trilateral 'GOVERNED SWARM · MODE:MIGRATION · 100 drones'
sandbox where drones flow through named zones (VALLEY-GATE, CANYON-THREAD,
THERMAL-LIFT, HIGH-PLAIN, JETSTREAM-E …) bound by per-zone capacity, with
red storm cells they must route around.

Mapped to Indian operational reality: 100 drones moving Leh → forward LAC
positions via three high-altitude passes — **Khardung La** (north),
**Tanglang La** (south), **Zoji La** (west) — each with seasonal capacity
limits, plus glacier storms and weather fronts that close passes intermittently.

Coordinate system: same sim metres the rest of LiveSession uses. The bridge's
local_to_geo() multiplies by SIM_GEO_SCALE (80×) before mapping to lat/lon
anchored at Leh (34.1526°N, 77.5770°E). So a 30 m sim radius shows up as a
2.4 km circle on the satellite map — readable at zoom 11–13.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Zone:
    """Capacity-limited corridor cell. Drones inside `radius_m` count toward `occupancy`."""

    id: str
    name: str
    center: tuple[float, float, float]  # sim (x, y, z) metres
    radius_m: float
    capacity: int                       # max simultaneous drones (governance gate)
    kind: str = "corridor"              # corridor | start | end | thermal | rest

    @property
    def color_hex(self) -> str:
        return {
            "start": "#4AE6A0",        # green
            "end": "#FFC83D",          # amber
            "corridor": "#00C2FF",     # cyan
            "thermal": "#FF8A1F",      # saffron (helpful updraft)
            "rest": "#A78BFA",         # purple (loiter)
        }.get(self.kind, "#94A3B8")


@dataclass
class Hazard:
    """Dynamic storm cell drones must route around. Cost added inside radius."""

    id: str
    name: str
    center: tuple[float, float, float]
    radius_m: float
    severity: float                     # 0..1, scales movement penalty
    pulse_phase: float = 0.0            # for visual breathing


@dataclass
class GovernedMigrationField:
    """The full multi-zone field — zones + hazards + per-drone assignment."""

    zones: list[Zone] = field(default_factory=list)
    hazards: list[Hazard] = field(default_factory=list)
    # drone_id → zone_id currently routed to (target waypoint)
    assignment: dict[int, str] = field(default_factory=dict)
    # drone_id → zone_id currently inside (for occupancy count)
    inside: dict[int, str] = field(default_factory=dict)
    # Telemetry counters for the console (mimicking trilateral sandbox)
    violations: int = 0
    collisions: int = 0
    yields: int = 0
    cycles: dict[str, int] = field(default_factory=dict)
    # Drones who completed full loop (start → corridors → end → return)
    completed_loops: int = 0
    last_zone_of: dict[int, str] = field(default_factory=dict)
    # Per-drone trail of recent (x, y) sim positions for the console PathLayer.
    trails: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    trail_max: int = 60
    # Zone-entry events for throughput calculation — (t, zone_id, drone_id) tuples.
    entry_events: list[tuple[float, str, int]] = field(default_factory=list)
    # Storm centre velocities (sim m/s) for slow drift.
    storm_vel: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def build_ladakh(cls) -> GovernedMigrationField:
        """Build the canonical Ladakh corridor map.

        Geometry (sim metres around Leh anchor):
          - START LEH AB at (0, -25, 6) — IAF Leh airbase notional spawn
          - North corridors: Khardung La (high cap, but storm-prone)
          - West corridor: Zoji La (low cap, all-weather)
          - South corridor: Tanglang La (medium cap, thermal updraft assist)
          - END NUBRA-FWD at (0, 25, 6) — forward LAC position
          - REST loiter cells flanking the start
          - Hazards: GLACIER STORM (north), SHEAR FRONT (centre)
        """
        zones: list[Zone] = [
            Zone("START",    "LEH AIRBASE",      (0, -25, 6), 6.0, capacity=100, kind="start"),
            Zone("REST_W",   "WESTERN PLAINS",   (-15, -20, 6), 5.0, capacity=18, kind="rest"),
            Zone("REST_E",   "EASTERN APRON",    (15, -20, 6), 5.0, capacity=18, kind="rest"),

            # Northern Khardung La pair — wide capacity, but glacier storm hits
            Zone("KHARDUNG", "KHARDUNG LA · N",  (-2, 0, 7), 6.0, capacity=40, kind="corridor"),
            Zone("THERMAL_N","THERMAL UPDRAFT",  (-8, 5, 8), 5.5, capacity=24, kind="thermal"),

            # Western Zoji La — narrow but stable
            Zone("ZOJI",     "ZOJI LA · W",      (-18, 0, 6), 4.0, capacity=15, kind="corridor"),

            # Southern Tanglang La — medium, with helpful thermal
            Zone("TANGLANG", "TANGLANG LA · S",  (6, 0, 7), 5.5, capacity=30, kind="corridor"),
            Zone("THERMAL_S","HIGH PLAIN LIFT",  (12, 5, 8), 5.0, capacity=20, kind="thermal"),

            Zone("END",      "NUBRA FWD POST",   (0, 25, 6), 6.0, capacity=100, kind="end"),
        ]
        hazards: list[Hazard] = [
            Hazard("STORM_N", "GLACIER STORM",  (-5, 8, 6), 5.5, severity=0.85),
            Hazard("SHEAR",   "WIND SHEAR FRONT", (3, -2, 6), 4.0, severity=0.45),
            Hazard("WX_E",    "EASTERN WX FRONT", (12, 10, 6), 4.5, severity=0.6),
        ]
        return cls(zones=zones, hazards=hazards)

    # ── Routing ────────────────────────────────────────────────────────

    def corridor_zones(self) -> list[Zone]:
        return [z for z in self.zones if z.kind == "corridor"]

    def hazard_cost(self, pos: np.ndarray) -> float:
        cost = 0.0
        for h in self.hazards:
            d = float(np.linalg.norm(pos[:2] - np.array(h.center[:2])))
            if d < h.radius_m:
                cost += h.severity * (1.0 - d / h.radius_m)
        return cost

    def zone_at(self, pos: np.ndarray) -> str | None:
        for z in self.zones:
            d = float(np.linalg.norm(pos[:2] - np.array(z.center[:2])))
            if d < z.radius_m:
                return z.id
        return None

    def occupancy(self) -> dict[str, int]:
        counts: dict[str, int] = {z.id: 0 for z in self.zones}
        for zid in self.inside.values():
            if zid in counts:
                counts[zid] += 1
        return counts

    def pick_corridor(self, drone_id: int, drone_pos: np.ndarray) -> Zone:
        """Drone picks the lowest-cost corridor: hazard + capacity + distance.

        This is the GOVERNANCE step — each drone selects independently using
        local information (its own position + last-broadcast occupancy +
        hazard field). Result is collective load balancing without a central
        scheduler.
        """
        occ = self.occupancy()
        best: Zone | None = None
        best_score = float("inf")
        for z in self.corridor_zones():
            zpos = np.array(z.center)
            dist = float(np.linalg.norm(drone_pos[:2] - zpos[:2]))
            # Capacity penalty — hard wall as we approach the cap
            load = occ.get(z.id, 0) / max(z.capacity, 1)
            cap_penalty = 0.0 if load < 0.7 else (load - 0.7) * 40.0
            # Hazard penalty — average between drone and zone centre (rough)
            mid = (drone_pos + zpos) * 0.5
            hazard = self.hazard_cost(mid) * 25.0
            score = dist + cap_penalty + hazard
            if score < best_score:
                best = z
                best_score = score
        return best or self.zones[0]

    def step(self, drone_positions: np.ndarray, dt: float, t: float) -> None:
        """Update inside-zone tracking + assignment + dynamic storms.

        Drones near a zone they're routed to are considered 'inside' it for
        occupancy bookkeeping. When a drone reaches its END goal, it bounces
        back through the loop (turns around for return leg) — keeps the
        scenario going indefinitely.
        """
        # Pulse storms (visual breathing) + drift each storm centre slowly.
        if not self.storm_vel:
            # First call — assign each storm a small wander velocity
            self.storm_vel = {
                "STORM_N": (0.03, -0.02),
                "SHEAR":   (0.04, 0.01),
                "WX_E":    (-0.025, 0.018),
            }
        for h in self.hazards:
            h.pulse_phase = (h.pulse_phase + dt * 0.7) % (2 * math.pi)
            vx, vy = self.storm_vel.get(h.id, (0.0, 0.0))
            cx, cy, cz = h.center
            # Wander within a bounded box; reverse direction at edges
            cx_new = cx + vx * dt
            cy_new = cy + vy * dt
            if abs(cx_new) > 14:
                vx = -vx
            if abs(cy_new) > 12:
                vy = -vy
            self.storm_vel[h.id] = (vx, vy)
            h.center = (cx_new, cy_new, cz)

        # Push trail history (decimate to every 2nd sample so we don't bloat)
        if int(round(t * 10)) % 2 == 0:
            for i, p in enumerate(drone_positions):
                tr = self.trails.setdefault(i, [])
                tr.append((float(p[0]), float(p[1])))
                if len(tr) > self.trail_max:
                    self.trails[i] = tr[-self.trail_max :]

        # Determine inside-zone per drone (closest zone within radius) + emit
        # entry events when a drone transitions into a new zone (for throughput).
        new_inside: dict[int, str] = {}
        for i, p in enumerate(drone_positions):
            zid = self.zone_at(p)
            if zid is not None:
                new_inside[i] = zid
                prev = self.inside.get(i)
                if prev != zid:
                    self.entry_events.append((t, zid, i))
                    # Trim event log to last 5 minutes for throughput compute
                    cutoff = t - 300.0
                    if len(self.entry_events) > 2000:
                        self.entry_events = [e for e in self.entry_events if e[0] >= cutoff]
                self.last_zone_of[i] = zid
        self.inside = new_inside

        # Count capacity violations — once per tick per overloaded zone
        occ = self.occupancy()
        for z in self.zones:
            if occ.get(z.id, 0) > z.capacity:
                self.violations += 1

        # Count cycle increments when a drone reaches END
        for i, zid in new_inside.items():
            if zid == "END" and self.assignment.get(i) == "END":
                self.completed_loops += 1
                # Turn this drone around: assign back to START so it loops
                self.assignment[i] = "START"
                self.cycles[str(i)] = self.cycles.get(str(i), 0) + 1
            elif zid == "START" and self.assignment.get(i) == "START":
                self.assignment[i] = "END"

    def throughput_per_min(self, now_t: float, window_s: float = 60.0) -> dict[str, int]:
        """Per-zone entries in the last `window_s` seconds."""
        cutoff = now_t - window_s
        counts: dict[str, int] = {z.id: 0 for z in self.zones}
        for et, zid, _did in self.entry_events:
            if et >= cutoff and zid in counts:
                counts[zid] += 1
        # Normalise to per-min
        scale = 60.0 / max(window_s, 1e-6)
        return {k: int(round(v * scale)) for k, v in counts.items()}

    def initial_assignments(self, n_drones: int) -> None:
        """At t=0 every drone is heading from START → END through some corridor."""
        for i in range(n_drones):
            self.assignment[i] = "END"

    def goal_for_drone(self, drone_id: int, drone_pos: np.ndarray) -> np.ndarray:
        """Compute the drone's next waypoint:
          - If assigned to END: route through the lowest-cost corridor first,
            then on to END once past the pass.
          - If assigned to START: head back to start.
        """
        target_zone_id = self.assignment.get(drone_id, "END")
        target = next((z for z in self.zones if z.id == target_zone_id), None)
        if target is None:
            return drone_pos
        # If not yet past mid-line, route through a corridor; else head straight.
        y = float(drone_pos[1])
        target_y = target.center[1]
        if target_zone_id == "END" and y < target_y - 4:
            corridor = self.pick_corridor(drone_id, drone_pos)
            return np.array(corridor.center)
        if target_zone_id == "START" and y > target_y + 4:
            corridor = self.pick_corridor(drone_id, drone_pos)
            return np.array(corridor.center)
        return np.array(target.center)
