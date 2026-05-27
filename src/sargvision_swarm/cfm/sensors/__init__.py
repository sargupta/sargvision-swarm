"""Sensor abstraction + hardware-attestation interfaces.

Multi-modal sensor reports flow through this layer before reaching TWSL. Each
SensorReport carries cryptographic + hardware integrity attestation flags that
let the trust-weighted fusion downstream gate or weight the report.

The attestation flags are the §101-defensible feature that converts TWSL from a
pure-mathematical object into a hardware-grounded apparatus claim per the
patent verdict in `47_PATENT_READINESS_VERDICT.md`.

Modules:
  - report      — SensorReport dataclass + modality enum
  - attestation — TPM / IMU triple-redundancy / RF tamper flag generation
  - fusion      — multi-modal cross-modal consistency check
"""

from __future__ import annotations

from .attestation import (
    AttestationFlags,
    generate_attestation,
    imu_triple_redundancy_check,
)
from .report import Modality, SensorReport
from .fusion import cross_modal_consistency, fuse_reports

__all__ = [
    "AttestationFlags",
    "generate_attestation",
    "imu_triple_redundancy_check",
    "Modality",
    "SensorReport",
    "cross_modal_consistency",
    "fuse_reports",
]
