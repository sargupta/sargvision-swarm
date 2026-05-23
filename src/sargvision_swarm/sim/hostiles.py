"""Hostile drone stream — incoming threats for the counter-swarm scenario.

Spawns N hostiles around the perimeter at random bearings; they fly straight
in toward the friendly ring centroid at a fixed speed. When any friendly
drone gets within `engagement_radius_m`, the hostile is marked neutralized
(simulated kinetic / soft-kill intercept).

Drives the IAF Counter-Swarm scene (ADITI 2.0 PS-11) — what the iDEX
problem statement asks for.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np


THREAT_MIX = (("decoy", 0.40), ("kinetic", 0.50), ("nuisance", 0.10))
# Per-class observable signatures (rcs, rf_emit, traj_jerk_rate) — units 0..1.
# decoy:    Luneburg lens inflates RCS, overpowered emitters, loiter jitter.
# kinetic:  small RCS, mild emit, smooth ballistic.
# nuisance: tiny RCS, minimal emit, drift.
CLASS_SIGNATURE = {
    "decoy":    {"rcs": 0.85, "rf_emit": 0.80, "jerk_rate": 0.70},
    "kinetic":  {"rcs": 0.35, "rf_emit": 0.45, "jerk_rate": 0.15},
    "nuisance": {"rcs": 0.10, "rf_emit": 0.10, "jerk_rate": 0.45},
}


@dataclass
class Hostile:
    id: int
    pos: np.ndarray            # (3,)
    vel: np.ndarray            # (3,)
    alive: bool = True
    spawn_bearing_deg: float = 0.0
    intent_label: str = "INBOUND"
    callsign: str = ""
    assigned_to: int | None = None  # drone_id of friendly intercepting
    threat_class: str = "kinetic"   # ground truth (hidden from defender)
    rcs: float = 0.5                # noisy observable
    rf_emit: float = 0.5            # noisy observable
    traj_jerk: float = 0.2          # rolling jitter signal
    panic_level: float = 0.0        # SIR-style fear contagion I_j ∈ [0, 1]
    _prev_vel: np.ndarray | None = None


@dataclass
class HostileFleet:
    """Manages a fleet of hostile drones converging on the friendly centroid."""

    spawn_count: int = 12
    spawn_radius_m: float = 40.0     # sim units (before geo scale)
    cruise_speed_ms: float = 1.0     # sim units / sec
    engagement_radius_m: float = 2.5
    seed: int = 7

    hostiles: list[Hostile] = field(default_factory=list)
    neutralized: int = 0
    spawned: int = 0
    _next_id: int = 1000

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        self.hostiles = []
        self.neutralized = 0
        self.spawned = 0
        self._next_id = 1000

    def spawn_initial(self, center: np.ndarray) -> None:
        """Spawn the initial wave on a ring around `center`."""
        self.reset()
        for i in range(self.spawn_count):
            bearing = (i / self.spawn_count) * 2 * math.pi + self._rng.uniform(-0.1, 0.1)
            self._spawn_one(center, bearing)

    def _draw_threat_class(self) -> str:
        r = self._rng.random()
        cum = 0.0
        for cls, w in THREAT_MIX:
            cum += w
            if r < cum:
                return cls
        return THREAT_MIX[-1][0]

    def _spawn_one(self, center: np.ndarray, bearing: float) -> None:
        offset = np.array(
            [
                self.spawn_radius_m * math.cos(bearing),
                self.spawn_radius_m * math.sin(bearing),
                0.0,
            ]
        )
        pos = center + offset
        pos[2] = 6.0 + self._rng.uniform(-1.0, 1.0)
        # initial velocity vector inward
        to_center = center - pos
        to_center[2] = 0.0
        norm = float(np.linalg.norm(to_center))
        vel = (to_center / max(norm, 1e-6)) * self.cruise_speed_ms
        tcls = self._draw_threat_class()
        sig = CLASS_SIGNATURE[tcls]
        # Class-prefixed callsign so console can tell at-a-glance during dev.
        prefix = {"decoy": "DEC", "kinetic": "KIN", "nuisance": "NUI"}[tcls]
        callsign = f"{prefix}-{self._next_id - 1000 + 1001:04d}"
        self.hostiles.append(
            Hostile(
                id=self._next_id,
                pos=pos.astype(float),
                vel=vel.astype(float),
                spawn_bearing_deg=math.degrees(bearing) % 360.0,
                callsign=callsign,
                threat_class=tcls,
                rcs=float(np.clip(sig["rcs"] + self._rng.gauss(0, 0.05), 0.0, 1.0)),
                rf_emit=float(np.clip(sig["rf_emit"] + self._rng.gauss(0, 0.05), 0.0, 1.0)),
                traj_jerk=sig["jerk_rate"] * 0.5,
            )
        )
        self._next_id += 1
        self.spawned += 1

    def step(
        self,
        dt: float,
        friendly_positions: np.ndarray,
        center: np.ndarray,
    ) -> dict:
        """Advance hostiles + check engagement.

        Returns a tick summary dict with kills_this_step + new_contacts.
        """
        kills_this_step: list[dict] = []
        new_contacts = 0
        for h in self.hostiles:
            if not h.alive:
                continue
            prev_vel = h.vel.copy()
            # Recompute velocity toward center (slowly home in)
            to_center = center - h.pos
            to_center[2] = 0.0
            n = float(np.linalg.norm(to_center))
            if n > 0.1:
                desired = (to_center / n) * self.cruise_speed_ms
                # Panicked hostiles deviate perpendicular to inbound vector,
                # producing the milling-vortex circulation Couzin-Krause predicts.
                if h.panic_level > 0.2:
                    perp = np.array([-to_center[1], to_center[0], 0.0])
                    perp_norm = float(np.linalg.norm(perp))
                    if perp_norm > 1e-6:
                        perp = perp / perp_norm * self.cruise_speed_ms
                        desired = (1.0 - h.panic_level) * desired + h.panic_level * perp
                # blend
                h.vel = 0.85 * h.vel + 0.15 * desired
            # Class-driven jitter: decoys + nuisance wobble; kinetics fly straight.
            sig = CLASS_SIGNATURE.get(h.threat_class, CLASS_SIGNATURE["kinetic"])
            wobble = sig["jerk_rate"]
            h.vel = h.vel + np.array(
                [
                    self._rng.gauss(0, wobble * 0.25),
                    self._rng.gauss(0, wobble * 0.25),
                    0.0,
                ]
            )
            h.pos = h.pos + h.vel * dt
            # Update rolling jerk + observation noise
            dv = float(np.linalg.norm(h.vel - prev_vel))
            h.traj_jerk = 0.8 * h.traj_jerk + 0.2 * min(dv * 2.0, 1.0)
            h.rcs = float(np.clip(sig["rcs"] + self._rng.gauss(0, 0.04), 0.0, 1.0))
            h.rf_emit = float(np.clip(sig["rf_emit"] + self._rng.gauss(0, 0.04), 0.0, 1.0))

            # Check engagement against any friendly
            diffs = friendly_positions - h.pos[None, :]
            dists = np.linalg.norm(diffs, axis=1)
            min_d = float(dists.min()) if len(dists) else float("inf")
            if min_d < self.engagement_radius_m:
                killer_idx = int(np.argmin(dists))
                h.alive = False
                h.intent_label = "NEUTRALIZED"
                kills_this_step.append(
                    {
                        "hostile_id": h.id,
                        "callsign": h.callsign,
                        "killer_idx": killer_idx,
                        "pos": [float(h.pos[0]), float(h.pos[1]), float(h.pos[2])],
                    }
                )
                self.neutralized += 1
            elif min_d < 8.0 and h.intent_label == "INBOUND":
                h.intent_label = "TERMINAL"
                new_contacts += 1

        # Prune dead hostiles after a short visible TTL
        return {
            "kills_this_step": kills_this_step,
            "new_contacts": new_contacts,
        }

    @property
    def remaining(self) -> int:
        return sum(1 for h in self.hostiles if h.alive)

    @property
    def total(self) -> int:
        return self.spawned
