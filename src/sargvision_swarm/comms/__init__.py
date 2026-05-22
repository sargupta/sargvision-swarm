"""Comms layer — in-memory pubsub w/ Zenoh-shaped topic API.

On Linux + ROS 2 + Zenoh, swap `InMemoryBus` for `ZenohBus` (TODO).
Topics are namespaced as `<scope>/<drone_id>/<channel>` to mirror real deploy.
"""

from sargvision_swarm.comms.bus import Channels, InMemoryBus, Topic

__all__ = ["Channels", "InMemoryBus", "Topic"]
