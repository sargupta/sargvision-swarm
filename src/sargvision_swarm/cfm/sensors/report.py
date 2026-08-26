"""SensorReport — the canonical multi-modal sensor report carrier.

Every sensor (radar, RF, EO, IR, acoustic, IMU) produces SensorReport instances
that flow into TWSL fusion. The dataclass is intentionally narrow + immutable:

  - one modality per report
  - one observation per report
  - cryptographic + hardware attestation flags carried inline
  - timestamp + reporter ID for graph construction

Downstream consumers (TWSL fusion, cross-modal consistency, console telemetry)
treat reports uniformly regardless of modality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .attestation import AttestationFlags


class Modality(Enum):
    """Sensor modality."""

    RADAR = "radar"
    RF_DIRECTION_FINDING = "rf_df"
    ELECTRO_OPTICAL = "eo"
    INFRARED = "ir"
    ACOUSTIC = "acoustic"
    IMU = "imu"
    GNSS = "gnss"
    HYPERSPECTRAL = "hyperspectral"


@dataclass(frozen=True)
class SensorReport:
    """A single sensor observation with attestation context.

    Attributes
    ----------
    reporter_id : str
        ID of the drone or ground sensor that produced the report.
    modality : Modality
        Which sensor produced the observation.
    target_position_enu_m : np.ndarray (3,) or None
        Best estimate of observed target position (None if non-localizing).
    target_velocity_mps : np.ndarray (3,) or None
        Best estimate of target velocity (None if not measured).
    confidence : float
        Reporter-internal confidence in (0, 1].
    timestamp_s : float
        Unix timestamp of observation.
    attestation : AttestationFlags
        Hardware + cryptographic integrity flags. Critical for trust gating.
    payload : dict
        Modality-specific extras (e.g., radar Doppler, RF frequency, EO image hash).
    """

    reporter_id: str
    modality: Modality
    target_position_enu_m: np.ndarray | None
    target_velocity_mps: np.ndarray | None
    confidence: float
    timestamp_s: float
    attestation: AttestationFlags
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def has_localization(self) -> bool:
        return self.target_position_enu_m is not None

    @property
    def is_trusted(self) -> bool:
        """Quick trust check — pass-through if all attestation flags pass."""
        return self.attestation.all_pass

    def with_confidence(self, conf: float) -> SensorReport:
        """Return a copy with updated confidence (frozen-friendly mutator)."""
        return SensorReport(
            reporter_id=self.reporter_id,
            modality=self.modality,
            target_position_enu_m=self.target_position_enu_m,
            target_velocity_mps=self.target_velocity_mps,
            confidence=conf,
            timestamp_s=self.timestamp_s,
            attestation=self.attestation,
            payload=self.payload,
        )
