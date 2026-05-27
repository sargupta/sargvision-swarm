"""Tests for the IACCS message bus mock."""

from __future__ import annotations

import time

import numpy as np

from sargvision_swarm.cfm.sovereign.iaccs import (
    IACCSMessage,
    IACCSMessageBus,
    IACCSTrack,
)


def _track(
    track_id: str = "T-001",
    hostility: str = "hostile",
    kind: str = "drone",
) -> IACCSTrack:
    return IACCSTrack(
        track_id=track_id,
        position_enu_m=np.array([1000.0, 2000.0, 100.0]),
        velocity_mps=np.array([-30.0, 0.0, 0.0]),
        track_kind=kind,  # type: ignore[arg-type]
        confidence=0.85,
        source_sensor="ROHINI-NODE-04",
        timestamp_s=time.time(),
        hostility=hostility,  # type: ignore[arg-type]
    )


def test_publish_track_registers_in_live_set():
    bus = IACCSMessageBus()
    track = _track()
    bus.publish_track(track, sender_id="IACCS-MOCK")
    live = bus.current_tracks()
    assert len(live) == 1
    assert live[0].track_id == "T-001"


def test_track_drop_removes_from_live_set():
    bus = IACCSMessageBus()
    bus.publish_track(_track("T-001"))
    bus.publish_track(_track("T-002"))
    assert len(bus.current_tracks()) == 2
    bus.publish(
        IACCSMessage(
            msg_type="track_drop",
            payload={"track_id": "T-001"},
            timestamp_s=time.time(),
            sender_id="IACCS-MOCK",
        )
    )
    live = bus.current_tracks()
    assert len(live) == 1
    assert live[0].track_id == "T-002"


def test_filter_by_hostility_and_kind():
    bus = IACCSMessageBus()
    bus.publish_track(_track("T-friend", hostility="friend"))
    bus.publish_track(_track("T-host-aircraft", hostility="hostile", kind="aircraft"))
    bus.publish_track(_track("T-host-drone", hostility="hostile", kind="drone"))
    hostiles = bus.current_tracks(hostility="hostile")
    assert {t.track_id for t in hostiles} == {"T-host-aircraft", "T-host-drone"}
    drones = bus.current_tracks(kind="drone")
    assert {t.track_id for t in drones} == {"T-friend", "T-host-drone"}


def test_subscriber_receives_published_messages():
    bus = IACCSMessageBus()
    received: list[IACCSMessage] = []
    bus.subscribe("track_update", received.append)
    bus.publish_track(_track())
    assert len(received) == 1
    assert received[0].msg_type == "track_update"


def test_heartbeat_publishes_envelope():
    bus = IACCSMessageBus()
    bus.heartbeat(sender_id="SARGVISION-CFM")
    hist = bus.history()
    assert any(m.msg_type == "heartbeat" for m in hist)


def test_history_bounded_by_max_history():
    bus = IACCSMessageBus(max_history=4)
    for i in range(10):
        bus.publish_track(_track(f"T-{i:03d}"))
    hist = bus.history()
    assert len(hist) == 4  # only last 4 retained
