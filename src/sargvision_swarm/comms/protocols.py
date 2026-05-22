"""Wire-level message types — what actually flies across the swarm bus.

Each message records:
- src / dst (or 'broadcast')
- protocol name (A2A / MAVLink / Zenoh-DDS / gRPC / BFT)
- payload (typed via Pydantic)
- byte size (so we can model bandwidth)

A2A spec recap (Google, Apr 2025, Apache-2.0, Linux Foundation):
- JSON-RPC 2.0 over HTTP + SSE
- Agent Cards advertise capabilities
- Sync request/response + streaming + async push notifications
- Complements MCP (which is agent-to-tool, A2A is agent-to-agent)

References:
- https://a2a-protocol.org/latest/specification/
- https://github.com/a2aproject/A2A
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Protocol(str, Enum):
    A2A = "A2A"                # agent-to-agent JSON-RPC 2.0 over HTTP+SSE
    MCP = "MCP"                # model context protocol (agent-to-tool)
    MAVLINK = "MAVLink"        # autopilot bus, signed v2
    ZENOH = "Zenoh"            # ROS 2 alt middleware, gossip discovery
    DDS = "DDS"                # ROS 2 default middleware (Fast/Cyclone)
    GRPC = "gRPC"              # LLM-to-LLM streaming
    BFT = "BFT"                # SwarmRaft Byzantine fault tolerance


# ── Typed payloads ─────────────────────────────────────────────────────


class PosePayload(BaseModel):
    pos: list[float]   # 3
    vel: list[float]   # 3
    yaw: float = 0.0
    bytes_size: int = 32   # MAVLink-ish position telem


class HeartbeatPayload(BaseModel):
    battery: float
    healthy: bool
    role: str
    bytes_size: int = 16


class IntentPayload(BaseModel):
    intent: str
    rationale: str = ""
    bytes_size: int = 96


class A2AMessage(BaseModel):
    """JSON-RPC 2.0 envelope (subset) per the A2A spec.

    Real wire: HTTPS POST + SSE stream. Here: in-memory dict.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str            # e.g. "negotiate.yield", "claim.task", "share.intent"
    params: dict[str, Any]
    bytes_size: int = 256


class A2ACard(BaseModel):
    """A2A Agent Card — advertised capability set."""

    agent_id: int
    name: str
    capabilities: list[str]
    transports: list[str] = ["HTTP+SSE"]
    bytes_size: int = 192


class BFTVote(BaseModel):
    """SwarmRaft committee vote on mission state."""

    voter_id: int
    term: int
    proposal: str         # e.g. "advance_phase:engage" / "abort" / "rtl"
    decision: Literal["yes", "no"]
    bytes_size: int = 64


class CBBABid(BaseModel):
    """Event-Driven CBBA bid on a task slot."""

    bidder_id: int
    task_id: str
    bid_score: float
    bundle: list[str]
    bytes_size: int = 128


# ── Envelope ────────────────────────────────────────────────────────────


class WireMessage(BaseModel):
    """The single envelope dropped onto the swarm bus."""

    t: float
    src: int                       # drone id; -1 = ground station
    dst: int | None = None         # None = broadcast
    protocol: Protocol
    topic: str                     # 'swarm/<id>/intent' shape
    payload: dict[str, Any]
    bytes_size: int

    @classmethod
    def make(
        cls,
        src: int,
        protocol: Protocol,
        topic: str,
        payload: BaseModel,
        dst: int | None = None,
        t: float | None = None,
    ) -> WireMessage:
        d = payload.model_dump()
        return cls(
            t=t if t is not None else time.time(),
            src=src,
            dst=dst,
            protocol=protocol,
            topic=topic,
            payload=d,
            bytes_size=int(d.get("bytes_size", 64)),
        )


class MessageLog:
    """Bounded ring buffer of WireMessages — fed into Gradio table."""

    def __init__(self, capacity: int = 20000) -> None:
        from collections import deque
        self._buffer: deque[WireMessage] = deque(maxlen=capacity)

    def append(self, msg: WireMessage) -> None:
        self._buffer.append(msg)

    def recent(self, k: int = 200) -> list[WireMessage]:
        return list(self._buffer)[-k:]

    def by_protocol(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self._buffer:
            counts[m.protocol.value] = counts.get(m.protocol.value, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._buffer)
