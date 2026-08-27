"""CFM scenarios — civilianised C-UAS use cases.

These scenarios are deliberately framed in civilian/critical-infrastructure
language to avoid §35 secrecy direction triggers and ITAR/EAR sensitivity.

Each scenario describes the defender swarm's task: protect a Critical
Infrastructure Asset (CIA) against an incoming hostile drone wave.

Scenarios:
  - c_uas_defence — generic C-UAS defence of a fixed asset against a
    mixed-threat wave (FPV + OWA + cruise-class)
"""

from __future__ import annotations

from .c_uas_defence import (
    CriticalInfrastructureAsset,
    CUASDefenceScenario,
    HostileWave,
    HostileWaveParams,
    ThreatClass,
)

__all__ = [
    "CUASDefenceScenario",
    "CriticalInfrastructureAsset",
    "HostileWave",
    "HostileWaveParams",
    "ThreatClass",
]
