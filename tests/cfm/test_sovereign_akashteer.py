"""Tests for the Akashteer cue channel mock."""

from __future__ import annotations

import time

import numpy as np

from sargvision_swarm.cfm.sovereign.akashteer import (
    AkashteerCueChannel,
    TacticalCue,
)


def _cue(
    cue_id: str = "AKA-CUE-1",
    priority: str = "P1_high",
    timestamp_s: float | None = None,
    window: tuple[float, float] = (0.0, 120.0),
) -> TacticalCue:
    return TacticalCue(
        cue_id=cue_id,
        target_position_enu_m=np.array([500.0, 1200.0, 80.0]),
        target_velocity_mps=np.array([-20.0, -5.0, 0.0]),
        target_kind="fpv",
        priority=priority,  # type: ignore[arg-type]
        engagement_window_s=window,
        authority="AKA-NODE-7",
        timestamp_s=timestamp_s if timestamp_s is not None else time.time(),
        confidence=0.9,
    )


def test_push_and_subscribe():
    ch = AkashteerCueChannel()
    received: list[TacticalCue] = []
    ch.subscribe(received.append)
    ch.push_cue(_cue())
    assert len(received) == 1


def test_priority_filter():
    ch = AkashteerCueChannel()
    ch.push_cue(_cue("c1", priority="P0_emergency"))
    ch.push_cue(_cue("c2", priority="P2_normal"))
    ch.push_cue(_cue("c3", priority="P3_observation"))
    high = ch.active_cues(min_priority="P1_high")
    assert {c.cue_id for c in high} == {"c1"}
    normal_or_better = ch.active_cues(min_priority="P2_normal")
    assert {c.cue_id for c in normal_or_better} == {"c1", "c2"}


def test_acknowledge_removes_cue():
    ch = AkashteerCueChannel()
    ch.push_cue(_cue("c1"))
    assert ch.acknowledge("c1") is True
    assert ch.acknowledge("c1") is False  # second ack returns False
    assert ch.active_cues() == []


def test_expire_old_cues():
    ch = AkashteerCueChannel()
    now = time.time()
    # cue that expired 10 s ago (window ended at now - 10)
    ch.push_cue(_cue("expired", timestamp_s=now - 130.0, window=(0.0, 120.0)))
    # cue still alive
    ch.push_cue(_cue("alive", timestamp_s=now, window=(0.0, 120.0)))
    expired_count = ch.expire(now_s=now)
    assert expired_count == 1
    assert {c.cue_id for c in ch.active_cues()} == {"alive"}
