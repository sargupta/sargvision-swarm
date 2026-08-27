"""IACCS message bus mock — Integrated Air Command and Control System schema.

IACCS is the IAF tri-service air-defence networking backbone — it ingests
radar tracks from BEL Aslesha/Rohini/Atulya + Akashteer node feeds, fuses
into a single air picture, and disseminates to combat elements.

For SARGVISION CFM, IACCS provides:
  - external air picture (tracks of aircraft, missiles, drones the wider AD
    network has detected) — used to prime CFM's own track allocation
  - airspace deconfliction (no-fly zones, friendly-aircraft corridors)
  - cross-cuing — feed CFM's own track output back into IACCS so wider AD
    can engage targets CFM cannot reach kinetically

This mock provides the publish/subscribe message schema. Production integration
swaps the in-memory bus for the actual IACCS TCP/UDP message gateway, which is
typically reached through a service-side proxy on a classified network.

This is a schema mock only — no operational interface secrets are embedded.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

#: Standard IACCS track message types.
MessageType = Literal[
    "track_update",  # external sensor track (radar/IRST/EO)
    "track_drop",  # external track lost
    "airspace_corridor",  # friendly aircraft corridor declaration
    "no_fly_zone",  # NFZ activation (civilian airliner, friendly base)
    "engagement_handoff",  # CFM declares it will engage a track
    "engagement_release",  # CFM hands target back to wider AD
    "heartbeat",  # peer liveness
]

TrackKind = Literal["aircraft", "missile", "drone", "unknown"]


@dataclass(frozen=True)
class IACCSTrack:
    """A single track in the wider AD air picture.

    Attributes
    ----------
    track_id : str
        Globally unique track ID (e.g., "IACCS-XYZ-A12-001234").
    position_enu_m : np.ndarray (3,)
        East-North-Up position relative to a declared local origin.
    velocity_mps : np.ndarray (3,)
        Velocity in ENU frame.
    track_kind : str
        Best-guess classification (aircraft / missile / drone / unknown).
    confidence : float
        Classifier confidence in (0, 1].
    source_sensor : str
        Origin sensor identifier (e.g., "ROHINI-NODE-04").
    timestamp_s : float
        Unix timestamp of the track update.
    hostility : str
        "friend" / "neutral" / "unknown" / "hostile".
    """

    track_id: str
    position_enu_m: np.ndarray
    velocity_mps: np.ndarray
    track_kind: TrackKind
    confidence: float
    source_sensor: str
    timestamp_s: float
    hostility: Literal["friend", "neutral", "unknown", "hostile"] = "unknown"


@dataclass
class IACCSMessage:
    """An IACCS message envelope."""

    msg_type: MessageType
    payload: dict
    timestamp_s: float
    sender_id: str


@dataclass
class IACCSMessageBus:
    """In-memory mock of the IACCS publish/subscribe message bus.

    Subscribers register a callback for one or more message types and receive
    every message of those types published to the bus.

    Parameters
    ----------
    max_history : int
        Number of recent messages to retain for late subscribers / debugging.
    """

    max_history: int = 256
    _subscribers: dict[MessageType, list[Callable[[IACCSMessage], None]]] = field(
        default_factory=dict
    )
    _history: deque[IACCSMessage] = field(default_factory=lambda: deque(maxlen=256))
    _tracks: dict[str, IACCSTrack] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Re-init deque maxlen from max_history (deque-init in field default uses literal 256)
        self._history = deque(maxlen=self.max_history)

    def subscribe(
        self,
        msg_type: MessageType,
        callback: Callable[[IACCSMessage], None],
    ) -> None:
        """Register a callback for messages of the given type."""
        self._subscribers.setdefault(msg_type, []).append(callback)

    def publish(self, msg: IACCSMessage) -> None:
        """Publish a message to all subscribers."""
        self._history.append(msg)
        # Maintain live track registry on track_update / track_drop
        if msg.msg_type == "track_update":
            track = msg.payload.get("track")
            if isinstance(track, IACCSTrack):
                self._tracks[track.track_id] = track
        elif msg.msg_type == "track_drop":
            track_id = msg.payload.get("track_id")
            if isinstance(track_id, str):
                self._tracks.pop(track_id, None)
        # Fan out to subscribers
        for cb in self._subscribers.get(msg.msg_type, []):
            cb(msg)

    def publish_track(self, track: IACCSTrack, sender_id: str = "IACCS-MOCK") -> None:
        """Convenience: publish a track_update."""
        self.publish(
            IACCSMessage(
                msg_type="track_update",
                payload={"track": track},
                timestamp_s=track.timestamp_s,
                sender_id=sender_id,
            )
        )

    def current_tracks(
        self, hostility: str | None = None, kind: TrackKind | None = None
    ) -> list[IACCSTrack]:
        """Return the current live track set, optionally filtered."""
        tracks = list(self._tracks.values())
        if hostility is not None:
            tracks = [t for t in tracks if t.hostility == hostility]
        if kind is not None:
            tracks = [t for t in tracks if t.track_kind == kind]
        return tracks

    def history(self, n: int | None = None) -> list[IACCSMessage]:
        """Return the most recent N messages (default: all retained)."""
        if n is None:
            return list(self._history)
        return list(self._history)[-n:]

    def heartbeat(self, sender_id: str = "SARGVISION-CFM") -> None:
        """Emit a liveness heartbeat."""
        self.publish(
            IACCSMessage(
                msg_type="heartbeat",
                payload={},
                timestamp_s=time.time(),
                sender_id=sender_id,
            )
        )
