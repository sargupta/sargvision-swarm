"""RF link budget — Friis path loss, jammer-degraded SNR, Shannon capacity.

Physics anchors (cross-checked vs research_2026_05/49_rf_link_budgets.md):
  - FSPL(1 km, 5.8 GHz) ≈ 107.7 dB
  - doubling distance adds exactly 6 dB
  - O₂ absorption peak at 60 GHz ≈ 15 dB/km (kills mmWave range)
  - jamming is geometrically asymmetric: a 10 mW jammer at 1 km can deny a
    1 W link at 10 km

Replaces the simulation's binary comm_range with a continuous SNR that the
coordination layer (and EWState) can consume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Speed of light (m/s).
_C = 299_792_458.0

#: Approx atmospheric absorption (dB/km) by band — coarse, for range realism.
_ABSORPTION_DB_PER_KM = {
    0.9e9: 0.01,
    2.4e9: 0.02,
    5.8e9: 0.05,
    28e9: 0.2,
    60e9: 15.0,  # O₂ peak — the reason mmWave is short-range
}


def _absorption_db_per_km(frequency_hz: float) -> float:
    """Nearest-band atmospheric absorption (dB/km)."""
    bands = sorted(_ABSORPTION_DB_PER_KM)
    nearest = min(bands, key=lambda b: abs(b - frequency_hz))
    return _ABSORPTION_DB_PER_KM[nearest]


def friis_path_loss_db(
    distance_m: float, frequency_hz: float, *, with_absorption: bool = True
) -> float:
    """Free-space path loss in dB (+ optional atmospheric absorption)."""
    if distance_m <= 0:
        return 0.0
    fspl = 20.0 * np.log10(distance_m) + 20.0 * np.log10(frequency_hz) - 147.55
    if with_absorption:
        fspl += _absorption_db_per_km(frequency_hz) * (distance_m / 1000.0)
    return float(fspl)


@dataclass(frozen=True)
class LinkBudget:
    """A drone radio link budget."""

    tx_power_dbm: float = 30.0  # 1 W
    tx_gain_dbi: float = 3.0
    rx_gain_dbi: float = 3.0
    frequency_hz: float = 5.8e9
    bandwidth_hz: float = 10e6
    noise_floor_dbm: float = -100.0  # ~10 MHz thermal + NF
    spreading_gain_db: float = 20.0  # FHSS/DSSS processing gain
    snr_threshold_db: float = 6.0  # min usable SNR

    def received_power_dbm(self, distance_m: float) -> float:
        fspl = friis_path_loss_db(distance_m, self.frequency_hz)
        return self.tx_power_dbm + self.tx_gain_dbi + self.rx_gain_dbi - fspl

    def snr_db(self, distance_m: float, jammer_power_dbm: float | None = None) -> float:
        """Effective SNR at the receiver, accounting for any jammer.

        The spreading gain is credited against the jammer (processing gain
        recovers margin); thermal noise is the floor when no jammer present.
        """
        s = self.received_power_dbm(distance_m)
        if jammer_power_dbm is None:
            noise = self.noise_floor_dbm
            return s - noise + self.spreading_gain_db * 0.0  # thermal-limited
        # Jammer present: effective noise = max(thermal, jammer), but the link
        # earns back `spreading_gain_db` against the jammer specifically.
        eff_jammer = jammer_power_dbm - self.spreading_gain_db
        noise = 10.0 * np.log10(10 ** (self.noise_floor_dbm / 10.0) + 10 ** (eff_jammer / 10.0))
        return float(s - noise)

    def is_connected(self, distance_m: float, jammer_power_dbm: float | None = None) -> bool:
        return self.snr_db(distance_m, jammer_power_dbm) >= self.snr_threshold_db

    def link_margin_db(self, distance_m: float, jammer_power_dbm: float | None = None) -> float:
        """SNR margin above threshold (feeds EWState.link_margin_db)."""
        return self.snr_db(distance_m, jammer_power_dbm) - self.snr_threshold_db

    def max_range_m(self, jammer_power_dbm: float | None = None) -> float:
        """Numerically solve for the distance where SNR == threshold."""
        lo, hi = 1.0, 1e6
        # monotone-decreasing SNR in distance → bisection
        if self.snr_db(lo, jammer_power_dbm) < self.snr_threshold_db:
            return 0.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.snr_db(mid, jammer_power_dbm) >= self.snr_threshold_db:
                lo = mid
            else:
                hi = mid
        return float(lo)


def link_snr_db(
    distance_m: float,
    *,
    tx_power_dbm: float = 30.0,
    frequency_hz: float = 5.8e9,
    noise_floor_dbm: float = -100.0,
    jammer_power_dbm: float | None = None,
    spreading_gain_db: float = 20.0,
) -> float:
    """Convenience wrapper for a one-off SNR computation."""
    lb = LinkBudget(
        tx_power_dbm=tx_power_dbm,
        frequency_hz=frequency_hz,
        noise_floor_dbm=noise_floor_dbm,
        spreading_gain_db=spreading_gain_db,
    )
    return lb.snr_db(distance_m, jammer_power_dbm)


def link_capacity_mbps(
    snr_db: float, bandwidth_hz: float = 10e6, *, efficiency: float = 0.65
) -> float:
    """Shannon capacity (Mbps) at the given SNR, scaled by a practical efficiency.

    C = B·log2(1 + SNR_linear). Real radios hit ~50-80% of Shannon — default 0.65.
    """
    snr_lin = 10 ** (snr_db / 10.0)
    c = bandwidth_hz * np.log2(1.0 + snr_lin) * efficiency
    return float(c / 1e6)
