"""GNSS spoofer model.

Replaces the simulation's binary `flags.gnss_denied` toggle with a graded,
position-drifting spoofer. Per file 52 (GNSS spoofing physics) + file 38
(adversarial ML), real GNSS spoofing produces:

  - position drift (slowly walking the victim's position estimate away from truth)
  - velocity drift (slewing the perceived velocity)
  - time drift (the most insidious — timestamps slowly shift)

The Type 0-3 taxonomy (Humphreys lab UT Austin) classifies sophistication:

  Type 0 = simple replay (easily detected by RAIM)
  Type 1 = lift-and-drag (gradual takeover from genuine signal)
  Type 2 = signal-synthesised (computational match to RAIM)
  Type 3 = ground-truth-matched (defeats most detectors)

This mock simulates Type 0-1 — sufficient to validate the resilience of CFM's
NavIC multi-frequency + multi-constellation cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GNSSSpoofer:
    """A GNSS spoofer that drifts a victim's position estimate.

    Parameters
    ----------
    drift_velocity_mps : np.ndarray (3,)
        Direction and magnitude of induced position drift.
    active : bool
        Whether the spoofer is currently transmitting.
    spoof_radius_m : float
        Spoofer effective radius — victims further than this are unaffected.
    centre_enu_m : np.ndarray (3,)
        Spoofer transmitter location.
    """

    drift_velocity_mps: np.ndarray
    active: bool = True
    spoof_radius_m: float = 5_000.0
    centre_enu_m: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.centre_enu_m is None:
            self.centre_enu_m = np.zeros(3)

    def in_range(self, victim_position_enu_m: np.ndarray) -> bool:
        return (
            float(np.linalg.norm(victim_position_enu_m - self.centre_enu_m)) <= self.spoof_radius_m
        )

    def apply(
        self,
        true_position_enu_m: np.ndarray,
        elapsed_s: float,
    ) -> np.ndarray:
        """Apply the spoof to a true position to produce the victim's perceived position.

        If the victim is outside spoof range or the spoofer is inactive, the
        true position is returned unchanged.
        """
        if not self.active:
            return true_position_enu_m
        if not self.in_range(true_position_enu_m):
            return true_position_enu_m
        return true_position_enu_m + self.drift_velocity_mps * elapsed_s

    def disable(self) -> None:
        self.active = False

    def enable(self) -> None:
        self.active = True
