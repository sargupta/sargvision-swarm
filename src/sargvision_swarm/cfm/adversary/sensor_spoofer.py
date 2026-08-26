"""Sensor-spoofing attacker — injects false sensor reports.

Per file 38 (adversarial ML), camera + LIDAR + radar injection attacks are
documented in academic literature (USENIX 2015 Son/Shin acoustic-injection
on MEMS gyros; CCS 2019 LIDAR injection; USENIX 2023 TPatch). This mock
simulates a compromised drone reporting falsified observations to its peers.

CFM's TWSL fusion + cross-modal consistency should detect this by:
  1. The compromised drone's reports disagree with multi-modal independent
     observers of the same target (cross-modal consistency check fails).
  2. The compromised drone fails attestation (TPM-signed report missing or
     IMU triple-redundancy fails).
  3. The TWSL Dirichlet residual on the compromised node climbs, causing the
     trust score to converge low, eventually triggering the kill-switch.

This attacker is used to validate that the defence layers actually defend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sensors.attestation import AttestationFlags
from ..sensors.report import Modality, SensorReport


@dataclass
class SensorSpoofingAttacker:
    """Compromised drone that injects false reports into its peers' streams.

    Parameters
    ----------
    compromised_node_id : str
        Reporter ID that this attacker has hijacked.
    spoof_offset_enu_m : np.ndarray (3,)
        Position offset added to every reported target (a constant bias).
    spoof_modalities : tuple[Modality, ...]
        Which modalities are being spoofed.
    """

    compromised_node_id: str
    spoof_offset_enu_m: np.ndarray
    spoof_modalities: tuple[Modality, ...] = (Modality.RADAR,)

    def craft_false_report(
        self,
        true_target_pos_enu_m: np.ndarray,
        true_target_vel_mps: np.ndarray | None,
        timestamp_s: float,
        modality: Modality | None = None,
    ) -> SensorReport:
        """Return a fabricated SensorReport with biased position."""
        mod = modality or self.spoof_modalities[0]
        # Compromised drone tries to look attested, but IMU triple-redundancy
        # fails (because injection bypasses sensor chip). This is the
        # signature CFM uses to detect.
        false_attestation = AttestationFlags(
            tpm_attested=True,  # adversary stole TPM key
            imu_triple_redundancy_pass=False,  # cannot fake — gives away spoof
            rf_tamper_flag=False,
            secure_boot_chain_valid=True,
        )
        return SensorReport(
            reporter_id=self.compromised_node_id,
            modality=mod,
            target_position_enu_m=true_target_pos_enu_m + self.spoof_offset_enu_m,
            target_velocity_mps=true_target_vel_mps,
            confidence=0.95,  # adversary asserts high confidence to manipulate fusion
            timestamp_s=timestamp_s,
            attestation=false_attestation,
            payload={"spoofed": True},  # diagnostic flag; real attacker wouldn't set this
        )
