"""DefenseField — hostile S-400 / IADS / EW battery layout for SEAD scenarios.

Mirrors `HostileFleet`'s role but for stationary (or slowly-relocating)
defense assets. CHANAKYA's planner reads the active subset of this field
to build the Riemannian threat manifold.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from sargvision_swarm.core.threat_field import DefenseAsset


@dataclass
class DefenseField:
    """A fleet of stationary hostile defense assets.

    `seed` controls reproducible spawning. `dirty` flips True when an asset
    is added / removed / toggled so CHANAKYA knows to replan.
    """

    seed: int = 13
    assets: list[DefenseAsset] = field(default_factory=list)
    dirty: bool = True

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # ── Construction ──────────────────────────────────────────────────

    def spawn_iads_layout(
        self,
        centre: np.ndarray,
        ring_radius: float = 18.0,
        n_radars: int = 4,
        engagement_radius: float = 6.5,
    ) -> None:
        """Ring of n_radars around `centre` — a realistic mini-IADS layout."""
        self.assets = []
        for i in range(n_radars):
            theta = (i / n_radars) * 2 * math.pi + self._rng.uniform(-0.05, 0.05)
            pos = centre + np.array(
                [
                    ring_radius * math.cos(theta),
                    ring_radius * math.sin(theta),
                    4.0,
                ]
            )
            self.assets.append(
                DefenseAsset(
                    pos=pos.astype(float),
                    engagement_radius=engagement_radius,
                    active=True,
                    name=f"S400-{i + 1:02d}",
                )
            )
        self.dirty = True

    def add_pop_up(
        self, pos: np.ndarray, engagement_radius: float = 4.0, name: str = "POPUP"
    ) -> None:
        self.assets.append(
            DefenseAsset(pos=pos.astype(float), engagement_radius=engagement_radius, name=name)
        )
        self.dirty = True

    def toggle(self, idx: int, active: bool | None = None) -> None:
        if 0 <= idx < len(self.assets):
            a = self.assets[idx]
            a.active = (not a.active) if active is None else active
            self.dirty = True

    def consume_dirty(self) -> bool:
        was = self.dirty
        self.dirty = False
        return was

    # ── Accessors ─────────────────────────────────────────────────────

    @property
    def active(self) -> list[DefenseAsset]:
        return [a for a in self.assets if a.active]

    def kill_radius_check(self, drone_positions: np.ndarray) -> np.ndarray:
        """Boolean mask — True for drones inside ANY active asset's engagement radius."""
        if not self.assets:
            return np.zeros(drone_positions.shape[0], dtype=bool)
        hit = np.zeros(drone_positions.shape[0], dtype=bool)
        for a in self.assets:
            if not a.active:
                continue
            d = np.linalg.norm(drone_positions - a.pos, axis=1)
            hit = hit | (d < a.engagement_radius)
        return hit
