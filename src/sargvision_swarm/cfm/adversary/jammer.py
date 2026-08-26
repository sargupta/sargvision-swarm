"""Friis-based RF jammer model.

Replaces the simulation's existing binary `flags.jamming` toggle with a
physics-grounded jammer entity. Jamming effect on a victim radio is a function
of:

  - jammer transmit power (dBm)
  - jammer antenna gain (dBi)
  - victim antenna gain (dBi)
  - operating frequency (Hz)
  - distance from jammer to victim (m)
  - victim receiver sensitivity / SNR threshold (dB)

The Friis transmission equation gives free-space path loss:

    FSPL(dB) = 20·log10(4π·d / λ)
             = 20·log10(d) + 20·log10(f) − 147.55

where d is metres, f is Hz, λ = c/f.

Jammer power at victim antenna:

    J_rx(dBm) = J_tx + G_j + G_v − FSPL

If J_rx exceeds the victim's noise floor by more than its spreading-gain
margin, the link is denied. Realistic small-radio noise floor: ~-100 dBm at
1 MHz bandwidth. Drone tactical radios typically have 10-30 dB spreading
margin from FHSS/DSSS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def friis_path_loss_db(distance_m: float, frequency_hz: float) -> float:
    """Free-space path loss in dB."""
    if distance_m <= 0:
        return 0.0
    # 20·log10(4π/c) ≈ -147.55 (with c = 299_792_458 m/s)
    return 20.0 * np.log10(distance_m) + 20.0 * np.log10(frequency_hz) - 147.55


@dataclass
class FriisJammer:
    """A physical RF jammer.

    Parameters
    ----------
    position_enu_m : np.ndarray (3,)
        Jammer location in ENU frame.
    tx_power_dbm : float
        Effective transmit power (dBm). 1 W = 30 dBm. Krasukha-class
        military jammers reach 60+ dBm EIRP.
    antenna_gain_dbi : float
        Jammer antenna gain (dBi). Omni-directional ~3 dBi; sector ~15 dBi.
    frequency_hz : float
        Centre frequency of jamming (Hz). Defaults to 5.8 GHz (common drone
        C2 band).
    bandwidth_hz : float
        Jamming bandwidth (Hz). Wider = more spectrum denied but lower
        power density per Hz.
    """

    position_enu_m: np.ndarray
    tx_power_dbm: float = 30.0  # 1 W
    antenna_gain_dbi: float = 3.0
    frequency_hz: float = 5.8e9
    bandwidth_hz: float = 20e6

    def power_at_victim_dbm(
        self,
        victim_position_enu_m: np.ndarray,
        victim_antenna_gain_dbi: float = 3.0,
    ) -> float:
        """Power received at the victim antenna (dBm)."""
        d = float(np.linalg.norm(victim_position_enu_m - self.position_enu_m))
        fspl = friis_path_loss_db(d, self.frequency_hz)
        return self.tx_power_dbm + self.antenna_gain_dbi + victim_antenna_gain_dbi - fspl

    def is_link_denied(
        self,
        victim_position_enu_m: np.ndarray,
        *,
        victim_noise_floor_dbm: float = -100.0,
        spreading_gain_db: float = 20.0,
        victim_antenna_gain_dbi: float = 3.0,
    ) -> bool:
        """Decide whether the victim's link is denied by this jammer.

        Returns True when jammer power at the victim exceeds the noise floor
        by more than the victim's spreading-gain margin.
        """
        j_rx = self.power_at_victim_dbm(
            victim_position_enu_m, victim_antenna_gain_dbi=victim_antenna_gain_dbi
        )
        # Link is denied when J/N ratio exceeds spreading gain.
        return j_rx > victim_noise_floor_dbm + spreading_gain_db

    def snr_degradation_db(
        self,
        victim_position_enu_m: np.ndarray,
        *,
        victim_noise_floor_dbm: float = -100.0,
        victim_antenna_gain_dbi: float = 3.0,
    ) -> float:
        """Effective SNR reduction (dB) caused by this jammer at the victim.

        Returns 0.0 if the jammer is too weak to affect the link.
        """
        j_rx = self.power_at_victim_dbm(
            victim_position_enu_m, victim_antenna_gain_dbi=victim_antenna_gain_dbi
        )
        # Effective noise = max(jammer_power, original_noise)
        effective_noise_dbm = max(j_rx, victim_noise_floor_dbm)
        return float(effective_noise_dbm - victim_noise_floor_dbm)

    def deny_radius_m(
        self,
        *,
        victim_noise_floor_dbm: float = -100.0,
        spreading_gain_db: float = 20.0,
        victim_antenna_gain_dbi: float = 3.0,
    ) -> float:
        """Radius at which this jammer denies a typical victim link.

        Solves J_rx(d) = noise_floor + spreading_gain for d.
        """
        # J_rx = tx + Gj + Gv − FSPL(d) ≥ floor + spreading_gain
        # FSPL(d) ≤ tx + Gj + Gv − floor − spreading_gain
        margin_db = (
            self.tx_power_dbm
            + self.antenna_gain_dbi
            + victim_antenna_gain_dbi
            - victim_noise_floor_dbm
            - spreading_gain_db
        )
        if margin_db <= 0:
            return 0.0
        # FSPL(d) = 20·log10(d) + 20·log10(f) − 147.55
        # → 20·log10(d) = margin − 20·log10(f) + 147.55
        log10_d = (margin_db - 20.0 * np.log10(self.frequency_hz) + 147.55) / 20.0
        return float(10.0**log10_d)
