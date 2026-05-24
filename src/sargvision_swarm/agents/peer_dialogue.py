"""Peer-to-peer drone dialogue using A2A-shaped JSON-RPC.

In real deployment: A2A SDK over HTTPS+SSE. Here: in-memory shim with the
same envelope shape so swap-out is trivial.

Methods:
- share.intent   — broadcast current intent
- negotiate.yield — request neighbor yield right-of-way
- claim.task     — assert task ownership (ED-CBBA tie-break)
- share.health   — gossip battery + healthy bit
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from sargvision_swarm.comms.protocols import (
    A2ACard,
    A2AMessage,
    HeartbeatPayload,
    PosePayload,
    Protocol,
    WireMessage,
)
from sargvision_swarm.comms.topology import CommModel
from sargvision_swarm.core.state import DroneState


def agent_card(drone: DroneState) -> A2ACard:
    """Build the drone's advertised A2A capability card."""
    return A2ACard(
        agent_id=drone.id,
        name=f"sargvision-drone-{drone.id:03d}",
        capabilities=[
            "share.intent",
            "negotiate.yield",
            "claim.task",
            "share.health",
        ],
        transports=["HTTP+SSE", "Zenoh"],
    )


@dataclass
class PeerDialogue:
    """Drives A2A-shaped exchanges between drones each cycle."""

    drones: list[DroneState]
    comm: CommModel
    rng_seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.rng_seed)

    def emit_round(self, t: float, intents: dict[int, str]) -> list[WireMessage]:
        """Produce one round of inter-drone messages.

        Returns the list of WireMessage envelopes that successfully delivered
        (after packet-loss filter).
        """
        import numpy as np

        positions = np.vstack([d.pos for d in self.drones])
        adjacency = self.comm.adjacency(positions)
        messages: list[WireMessage] = []
        n = len(self.drones)
        for i in range(n):
            # 1. Broadcast pose via Zenoh-shaped pubsub (always to all in range)
            pose_msg = WireMessage.make(
                src=self.drones[i].id,
                protocol=Protocol.ZENOH,
                topic=f"swarm/{self.drones[i].id}/pose",
                payload=PosePayload(
                    pos=self.drones[i].pos.tolist(),
                    vel=self.drones[i].vel.tolist(),
                ),
                t=t,
            )
            messages.append(pose_msg)

            # 2. Per-drone A2A intent share (sample 1-2 neighbors)
            neighbors = [j for j in range(n) if adjacency[i, j]]
            if not neighbors:
                continue
            sample = self._rng.sample(neighbors, k=min(2, len(neighbors)))
            for j in sample:
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                if not self.comm.will_deliver(dist):
                    continue
                intent = intents.get(self.drones[i].id, "hold_formation")
                a2a = A2AMessage(
                    method="share.intent",
                    params={
                        "from": self.drones[i].id,
                        "intent": intent,
                        "yaw": 0.0,
                    },
                )
                messages.append(
                    WireMessage.make(
                        src=self.drones[i].id,
                        dst=self.drones[j].id,
                        protocol=Protocol.A2A,
                        topic=f"a2a/drone-{self.drones[j].id:03d}/share.intent",
                        payload=a2a,
                        t=t,
                    )
                )

                # 3. Negotiate yield if drones are very close
                if dist < 2.0:
                    yield_req = A2AMessage(
                        method="negotiate.yield",
                        params={"requester": self.drones[i].id, "reason": "proximity"},
                    )
                    messages.append(
                        WireMessage.make(
                            src=self.drones[i].id,
                            dst=self.drones[j].id,
                            protocol=Protocol.A2A,
                            topic=f"a2a/drone-{self.drones[j].id:03d}/negotiate.yield",
                            payload=yield_req,
                            t=t,
                        )
                    )

            # 4. Heartbeat every cycle (MAVLink shape)
            hb = HeartbeatPayload(
                battery=self.drones[i].battery,
                healthy=self.drones[i].healthy,
                role=self.drones[i].role.value,
            )
            messages.append(
                WireMessage.make(
                    src=self.drones[i].id,
                    protocol=Protocol.MAVLINK,
                    topic=f"mavlink/drone-{self.drones[i].id:03d}/heartbeat",
                    payload=hb,
                    t=t,
                )
            )

        return messages
