"""Comm topology model — who can hear whom, signal strength, packet loss.

Drones can only talk if within `range_m`. Signal strength decays inverse-square.
Random Bernoulli packet loss above a distance threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CommModel:
    range_m: float = 15.0  # Doodle Labs Mesh Rider-ish indoor range
    loss_start_m: float = 10.0  # below this, ~0% loss
    max_loss: float = 0.4  # at range_m, packet loss this high
    rng_seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.rng_seed)

    def adjacency(self, positions: np.ndarray) -> np.ndarray:
        """(N, N) bool — True if drones i, j are within range and not self."""
        n = positions.shape[0]
        diffs = positions[None, :, :] - positions[:, None, :]
        dist = np.linalg.norm(diffs, axis=-1)
        in_range = dist < self.range_m
        np.fill_diagonal(in_range, False)
        return in_range

    def signal_strength(self, positions: np.ndarray) -> np.ndarray:
        """(N, N) in [0, 1] — 1 close, 0 at range."""
        n = positions.shape[0]
        diffs = positions[None, :, :] - positions[:, None, :]
        dist = np.linalg.norm(diffs, axis=-1)
        # Clip to range, invert
        strength = np.clip(1.0 - dist / self.range_m, 0.0, 1.0)
        np.fill_diagonal(strength, 0.0)
        return strength

    def packet_loss(self, dist: float) -> float:
        """Probability a packet is dropped at distance `dist`."""
        if dist >= self.range_m:
            return 1.0
        if dist <= self.loss_start_m:
            return 0.0
        # Linear from 0 at loss_start to max_loss at range
        frac = (dist - self.loss_start_m) / (self.range_m - self.loss_start_m)
        return self.max_loss * frac

    def will_deliver(self, dist: float) -> bool:
        p_drop = self.packet_loss(dist)
        return bool(self._rng.uniform() > p_drop)
