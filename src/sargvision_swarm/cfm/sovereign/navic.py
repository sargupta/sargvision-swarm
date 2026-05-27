"""NavIC receiver mock — IRNSS / NavIC PNT solution simulation.

Indian Regional Navigation Satellite System (IRNSS / NavIC) provides regional
PNT coverage over India + 1500 km radius. Constellation as of 2026:

  - NavIC-1A through 1I (legacy generation, partial operational)
  - NVS-01 / NVS-02 (next-gen, L1 + L5 + S-band signals)
  - Target nominal constellation: 7 operational satellites + spares

As of early-2026 reporting, 3-5 satellites are operational; remaining
satellites are being progressively replaced (NVS-02 launched Jan 2025).

This mock simulates a multi-frequency NavIC receiver consuming L1 + L5 signals
and producing a position-velocity-time (PVT) solution with realistic noise +
graceful-degradation behaviour under reduced satellite count.

Reference: ISRO NavIC Signal-In-Space ICD, IRN-ICD-SIS-201, v1.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

#: Default nominal satellite count when constellation is healthy.
NOMINAL_NAVIC_SAT_COUNT = 7

#: Minimum satellite count for a 3D fix. Below this → 2D-only or no fix.
MIN_SATS_FOR_3D_FIX = 4

#: Position noise (1-sigma, metres) at full constellation health.
NOMINAL_POS_NOISE_M = 2.5  # typical for L1+L5 multi-freq, open-sky

#: Position noise (1-sigma, metres) under degraded constellation.
DEGRADED_POS_NOISE_M = 12.0

#: Jammer power threshold (dBm) above which NavIC L1 is denied.
NAVIC_L1_JAM_THRESHOLD_DBM = -120.0


@dataclass(frozen=True)
class PNTSolution:
    """PNT solution from a NavIC receiver.

    Attributes
    ----------
    position_m : np.ndarray of shape (3,)
        ECEF / local-tangent (x, y, z) position in metres.
    velocity_mps : np.ndarray of shape (3,)
        Velocity in m/s.
    time_unix_s : float
        UTC time of solution in Unix seconds.
    sats_used : int
        Number of satellites contributing to the solution.
    fix_type : str
        One of "no_fix", "2d", "3d", "3d_l5".
    sigma_pos_m : float
        1-sigma position uncertainty (metres).
    """

    position_m: np.ndarray
    velocity_mps: np.ndarray
    time_unix_s: float
    sats_used: int
    fix_type: Literal["no_fix", "2d", "3d", "3d_l5"]
    sigma_pos_m: float

    @property
    def has_valid_fix(self) -> bool:
        return self.fix_type != "no_fix"


@dataclass
class NavICReceiver:
    """Mock NavIC L1 + L5 dual-frequency receiver.

    Parameters
    ----------
    sats_visible : int
        Currently visible operational NavIC satellites (typically 3-7 as of
        2026).
    l5_enabled : bool
        Whether the receiver supports L5 (better ionospheric correction +
        anti-spoof). Default True for modern units.
    jammer_dbm : float
        Ambient jammer power at the antenna (dBm). Values above
        NAVIC_L1_JAM_THRESHOLD_DBM will deny L1 acquisition.
    rng_seed : int
        Random seed for noise reproducibility.
    """

    sats_visible: int = NOMINAL_NAVIC_SAT_COUNT
    l5_enabled: bool = True
    jammer_dbm: float = -140.0  # ambient floor, no jammer
    rng_seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.rng_seed)

    def get_pvt(self, true_position_m: np.ndarray, time_unix_s: float) -> PNTSolution:
        """Compute a PVT solution given the true position.

        Adds noise consistent with the current sat count + jammer state +
        L5 availability. Returns no_fix when insufficient satellites or
        jamming exceeds threshold.
        """
        true_pos = np.asarray(true_position_m, dtype=np.float64).reshape(3)

        # Jamming check (only affects L1; L5 has higher anti-jam margin)
        l1_denied = self.jammer_dbm > NAVIC_L1_JAM_THRESHOLD_DBM
        l5_denied = self.jammer_dbm > NAVIC_L1_JAM_THRESHOLD_DBM + 6.0  # L5 ~6dB more robust

        if l1_denied and (not self.l5_enabled or l5_denied):
            return PNTSolution(
                position_m=np.array([np.nan, np.nan, np.nan]),
                velocity_mps=np.zeros(3),
                time_unix_s=time_unix_s,
                sats_used=0,
                fix_type="no_fix",
                sigma_pos_m=float("inf"),
            )

        # Satellite count
        sats_used = self.sats_visible
        if l1_denied:
            sats_used = max(0, sats_used - 3)  # lose L1 → fewer usable sats

        # Fix type
        if sats_used < MIN_SATS_FOR_3D_FIX:
            return PNTSolution(
                position_m=np.array([np.nan, np.nan, np.nan]),
                velocity_mps=np.zeros(3),
                time_unix_s=time_unix_s,
                sats_used=sats_used,
                fix_type="no_fix",
                sigma_pos_m=float("inf"),
            )

        # Position noise scales inversely with sqrt(sats) and improves with L5
        sigma = NOMINAL_POS_NOISE_M * np.sqrt(NOMINAL_NAVIC_SAT_COUNT / sats_used)
        if not self.l5_enabled or l5_denied:
            sigma *= 1.8  # L1-only is ~1.8x noisier (ionospheric uncorrected)
        if sats_used < NOMINAL_NAVIC_SAT_COUNT - 2:
            sigma = max(sigma, DEGRADED_POS_NOISE_M)

        pos_noise = self._rng.normal(0.0, sigma, size=3)
        vel_noise = self._rng.normal(0.0, sigma * 0.1, size=3)

        fix_type: Literal["2d", "3d", "3d_l5"]
        if sats_used >= NOMINAL_NAVIC_SAT_COUNT and self.l5_enabled and not l5_denied:
            fix_type = "3d_l5"
        elif sats_used >= MIN_SATS_FOR_3D_FIX:
            fix_type = "3d"
        else:
            fix_type = "2d"

        return PNTSolution(
            position_m=true_pos + pos_noise,
            velocity_mps=vel_noise,
            time_unix_s=time_unix_s,
            sats_used=sats_used,
            fix_type=fix_type,
            sigma_pos_m=float(sigma),
        )

    def set_jammer_dbm(self, dbm: float) -> None:
        """Update ambient jammer power (e.g., from environment model)."""
        self.jammer_dbm = float(dbm)

    def set_sats_visible(self, count: int) -> None:
        """Update visible-satellite count (e.g., satellite outage simulation)."""
        self.sats_visible = max(0, int(count))
