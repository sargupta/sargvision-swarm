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
        callsign = f"KAM-{self._next_id - 1000 + 1001:04d}"  # KAM-1001..
        self.hostiles.append(
            Hostile(
                id=self._next_id,
                pos=pos.astype(float),
                vel=vel.astype(float),
                spawn_bearing_deg=math.degrees(bearing) % 360.0,
                callsign=callsign,
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
            # Recompute velocity toward center (slowly home in)
            to_center = center - h.pos
            to_center[2] = 0.0
            n = float(np.linalg.norm(to_center))
            if n > 0.1:
                desired = (to_center / n) * self.cruise_speed_ms
                # blend
                h.vel = 0.85 * h.vel + 0.15 * desired
            h.pos = h.pos + h.vel * dt

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
