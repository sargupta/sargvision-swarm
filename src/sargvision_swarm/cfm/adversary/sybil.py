"""Sybil attacker — physical drone impersonation.

A Sybil attack adds N adversary drones that appear physically valid to peers
(they fly, they emit RF) but are not part of the legitimate fleet. TWSL trust
+ hardware attestation should detect this:

  1. Sybil drones lack TPM-bound keys → attestation fails → trust → 0.
  2. Sybil drone reports disagree with the legitimate fleet's consensus
     position estimates → TWSL Dirichlet residual climbs.

Per file 38 (adversarial ML), Sybil is one of the 5 named attack vectors
SARGVISION must defend against. The TWSL operator alone is insufficient —
Sybil defence requires the cryptographic identity layer
(see ARCHITECTURE_2046.md §1 Hook 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SybilDrone:
    """A Sybil-attacker-introduced fake drone."""

    sybil_id: str
    position_enu_m: np.ndarray
    velocity_mps: np.ndarray
    claims_node_id: str  # ID it pretends to be (may collide with real drone)


@dataclass
class SybilAttacker:
    """Spawn + control a population of Sybil drones."""

    sybil_drones: list[SybilDrone] = field(default_factory=list)

    def add_sybil(
        self,
        sybil_id: str,
        position_enu_m: np.ndarray,
        velocity_mps: np.ndarray | None = None,
        claims_node_id: str | None = None,
    ) -> None:
        """Inject a Sybil into the population."""
        self.sybil_drones.append(
            SybilDrone(
                sybil_id=sybil_id,
                position_enu_m=position_enu_m,
                velocity_mps=velocity_mps if velocity_mps is not None else np.zeros(3),
                claims_node_id=claims_node_id or sybil_id,
            )
        )

    def step(self, dt_s: float) -> None:
        """Advance Sybil positions one timestep."""
        for s in self.sybil_drones:
            s.position_enu_m = s.position_enu_m + s.velocity_mps * dt_s

    def count(self) -> int:
        return len(self.sybil_drones)
