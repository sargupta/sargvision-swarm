"""Hardware-attestation flags for sensor reports.

Per the §101 / §3(k) patent verdict (47_PATENT_READINESS_VERDICT.md), the
defensible patent feature for TWSL is hardware-attested gating — every sensor
report must carry verifiable flags that the trust-weighted fusion can use to
gate or weight the report.

Three classes of attestation flag:

  1. **TPM-attested**: the report was signed by a TPM-rooted key bound to the
     drone hardware. Verifies the reporter has not been firmware-tampered.

  2. **IMU triple-redundancy pass**: three independent IMU chips on the drone
     agreed within tolerance on the platform's own attitude / acceleration —
     defends against acoustic-injection MEMS spoofing (USENIX 2015 Son/Shin).

  3. **RF tamper flag**: the drone's radio link did not exhibit anomalous
     timing / latency / power signatures suggestive of MITM or replay.

This module simulates these flags for the CFM sandbox. Production hardware
generates these via actual TPM (e.g., Microchip ATECC608A) + sensor cross-
check + RF side-channel monitor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AttestationFlags:
    """Cryptographic + hardware integrity flags accompanying a sensor report.

    Attributes
    ----------
    tpm_attested : bool
        Report was signed by the drone's TPM-bound key.
    imu_triple_redundancy_pass : bool
        Three IMU chips agreed (within tolerance) at observation time.
    rf_tamper_flag : bool
        True if RF link showed tamper-suggestive signature (BAD — flag set
        means tamper SUSPECTED; downstream should distrust).
    secure_boot_chain_valid : bool
        Drone booted from trusted firmware chain (UEFI / Secure Boot equivalent).
    """

    tpm_attested: bool = True
    imu_triple_redundancy_pass: bool = True
    rf_tamper_flag: bool = False
    secure_boot_chain_valid: bool = True

    @property
    def all_pass(self) -> bool:
        """True iff every attestation check passed."""
        return (
            self.tpm_attested
            and self.imu_triple_redundancy_pass
            and not self.rf_tamper_flag
            and self.secure_boot_chain_valid
        )

    @property
    def trust_multiplier(self) -> float:
        """Scalar multiplier for downstream trust weighting.

        Returns:
          1.0 if all flags pass
          0.5 if one flag fails (degraded but usable)
          0.1 if two flags fail (heavily downweighted)
          0.0 if three or more fail (effectively rejected)
        """
        failed = sum(
            [
                not self.tpm_attested,
                not self.imu_triple_redundancy_pass,
                bool(self.rf_tamper_flag),
                not self.secure_boot_chain_valid,
            ]
        )
        return {0: 1.0, 1: 0.5, 2: 0.1, 3: 0.0, 4: 0.0}[failed]


def generate_attestation(
    rng: np.random.Generator | None = None,
    *,
    fail_tpm_prob: float = 0.0,
    fail_imu_prob: float = 0.0,
    rf_tamper_prob: float = 0.0,
    fail_secure_boot_prob: float = 0.0,
) -> AttestationFlags:
    """Generate a synthetic attestation flag set.

    The default produces all-pass flags. For adversarial simulation, dial up
    the failure probabilities to model compromised drones.
    """
    if rng is None:
        rng = np.random.default_rng()
    return AttestationFlags(
        tpm_attested=bool(rng.random() >= fail_tpm_prob),
        imu_triple_redundancy_pass=bool(rng.random() >= fail_imu_prob),
        rf_tamper_flag=bool(rng.random() < rf_tamper_prob),
        secure_boot_chain_valid=bool(rng.random() >= fail_secure_boot_prob),
    )


def imu_triple_redundancy_check(
    imu_a: np.ndarray,
    imu_b: np.ndarray,
    imu_c: np.ndarray,
    *,
    tolerance: float = 0.05,
) -> bool:
    """Triple-redundancy consistency check on three IMU readings.

    Returns True if all three IMUs agreed within `tolerance` on every axis.
    Used to detect acoustic-injection or single-chip-MEMS-spoofing attacks
    that compromise one IMU at a time.

    Args
    ----
    imu_a, imu_b, imu_c : np.ndarray
        Three independent IMU readings (e.g., 3-axis acceleration in m/s^2).
    tolerance : float
        Per-axis maximum allowed deviation between any two IMUs.
    """
    a = np.asarray(imu_a, dtype=np.float64)
    b = np.asarray(imu_b, dtype=np.float64)
    c = np.asarray(imu_c, dtype=np.float64)
    if not (a.shape == b.shape == c.shape):
        raise ValueError("all three IMU readings must have the same shape")
    diffs = np.stack(
        [
            np.abs(a - b),
            np.abs(b - c),
            np.abs(a - c),
        ]
    )
    return bool(np.all(diffs <= tolerance))
