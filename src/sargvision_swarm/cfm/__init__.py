"""CFM — C-UAS Fusion Mesh.

SARGVISION CFM is the sovereign-context coordination middleware that sits above
heterogeneous sensor hardware (radar, RF, EO/IR, acoustic) and below kinetic +
EW effectors. It provides:

  - sovereign-stack integration (NavIC PNT, Bhuvan tiles, IACCS air picture,
    Akashteer tactical cues)
  - hardware-attested multi-modal sensor fusion via TWSL trust operator
  - saturation-aware track allocation under cost-exchange constraints
  - adversarial-resilient coordination under jamming + spoofing + Sybil
  - DEW-aware engagement planning (route around directed-energy zones,
    persistent low-detectability)

This package is the commercial product surface. The civilian/research mathematics
remain in `sargvision_swarm.core.*`; the public scenarios in
`sargvision_swarm.sim.*`. CFM is the bridge between core math and operational
deployment.

Sub-packages:
  - sovereign  — NavIC / Bhuvan / IACCS / Akashteer integration mocks
  - sensors    — SensorReport + hardware-attestation interfaces
  - scenarios  — civilianised C-UAS defence scenarios
  - strategies — 4 C-UAS posture strategies (LAYERED / POINT / AREA / MOBILE_CAP)
  - adversary  — sensor-spoofing, GNSS-spoofing, Sybil, Friis-jamming attacker models
"""

from __future__ import annotations

__version__ = "0.1.0-dev"
