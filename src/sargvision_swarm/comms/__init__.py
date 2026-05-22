"""Comms layer — in-memory pubsub w/ Zenoh-shaped topic API.

On Linux + ROS 2 + Zenoh, swap `InMemoryBus` for `ZenohBus` (TODO).
Topics are namespaced as `<scope>/<drone_id>/<channel>` to mirror real deploy.
"""

from sargvision_swarm.comms.bandwidth import BandwidthTracker
from sargvision_swarm.comms.bus import Channels, InMemoryBus, Topic
from sargvision_swarm.comms.protocols import (
    A2ACard,
    A2AMessage,
    BFTVote,
    CBBABid,
    HeartbeatPayload,
    IntentPayload,
    MessageLog,
    PosePayload,
    Protocol,
    WireMessage,
)
from sargvision_swarm.comms.topology import CommModel

__all__ = [
    "A2ACard",
    "A2AMessage",
    "BFTVote",
    "BandwidthTracker",
    "CBBABid",
    "Channels",
    "CommModel",
    "HeartbeatPayload",
    "InMemoryBus",
    "IntentPayload",
    "MessageLog",
    "PosePayload",
    "Protocol",
    "Topic",
    "WireMessage",
]
