"""Akashteer cue channel mock — tactical air-defence cueing input.

Akashteer is the Indian Army's automated tactical air-defence command-and-
control + cue network. It ingests radar tracks from forward radars, computes
engagement solutions, and pushes time-critical cues to tactical AD batteries.

For SARGVISION CFM, Akashteer is an INPUT — it provides high-priority cues
("engage this track immediately") that the CFM coordination layer must honour
within latency budgets, rather than re-deriving from scratch.

This mock provides the cue-channel input schema. Production integration swaps
the in-memory channel for the actual Akashteer interface (typically a
classified tactical-radio mesh).

Schema-only mock — no operational secrets embedded.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np


#: Priority levels emitted by Akashteer for cued tracks.
CuePriority = Literal[
    "P0_emergency",  # incoming, < 30 s to engagement window close
    "P1_high",       # high-priority engagement, < 2 min
    "P2_normal",     # normal cue
    "P3_observation",  # observe-only, no engagement authority
]


@dataclass(frozen=True)
class TacticalCue:
    """A single Akashteer tactical cue.

    Attributes
    ----------
    cue_id : str
        Cue identifier (e.g., "AKA-CUE-2026-05-24T14:32:01.123").
    target_position_enu_m : np.ndarray (3,)
        Predicted intercept point in local ENU frame, metres.
    target_velocity_mps : np.ndarray (3,)
        Predicted target velocity at intercept time.
    target_kind : str
        Best-guess threat category (fpv / owa / cruise / aircraft / unknown).
    priority : str
        One of CuePriority levels.
    engagement_window_s : tuple of (start, end) seconds-from-now
        Time window in which engagement is authorised.
    authority : str
        Authorising element identifier (e.g., "AKA-NODE-7").
    timestamp_s : float
        Unix timestamp of cue generation.
    confidence : float
        Classifier confidence in (0, 1].
    """

    cue_id: str
    target_position_enu_m: np.ndarray
    target_velocity_mps: np.ndarray
    target_kind: Literal["fpv", "owa", "cruise", "aircraft", "unknown"]
    priority: CuePriority
    engagement_window_s: tuple[float, float]
    authority: str
    timestamp_s: float
    confidence: float = 0.8


@dataclass
class AkashteerCueChannel:
    """In-memory mock of the Akashteer tactical cue channel.

    Receives high-priority engagement cues from upstream tactical AD nodes
    and dispatches them to CFM subscribers.

    Parameters
    ----------
    max_history : int
        Number of recent cues retained for inspection.
    """

    max_history: int = 64
    _subscribers: list[Callable[[TacticalCue], None]] = field(default_factory=list)
    _history: deque[TacticalCue] = field(default_factory=lambda: deque(maxlen=64))
    _active_cues: dict[str, TacticalCue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.max_history)

    def subscribe(self, callback: Callable[[TacticalCue], None]) -> None:
        """Register a callback to receive every incoming cue."""
        self._subscribers.append(callback)

    def push_cue(self, cue: TacticalCue) -> None:
        """Push a tactical cue into the channel."""
        self._history.append(cue)
        self._active_cues[cue.cue_id] = cue
        for cb in self._subscribers:
            cb(cue)

    def expire(self, now_s: float | None = None) -> int:
        """Expire any cues whose engagement window has passed.

        Returns the number expired.
        """
        now = float(now_s if now_s is not None else time.time())
        expired = [
            cue_id
            for cue_id, cue in self._active_cues.items()
            if cue.timestamp_s + cue.engagement_window_s[1] < now
        ]
        for cue_id in expired:
            self._active_cues.pop(cue_id, None)
        return len(expired)

    def active_cues(self, min_priority: CuePriority = "P3_observation") -> list[TacticalCue]:
        """Return active cues at or above the given priority level."""
        order = {
            "P0_emergency": 0,
            "P1_high": 1,
            "P2_normal": 2,
            "P3_observation": 3,
        }
        threshold = order[min_priority]
        return [c for c in self._active_cues.values() if order[c.priority] <= threshold]

    def acknowledge(self, cue_id: str) -> bool:
        """CFM acknowledges + accepts engagement responsibility for a cue."""
        return self._active_cues.pop(cue_id, None) is not None

    def history(self, n: int | None = None) -> list[TacticalCue]:
        """Return recent cue history."""
        if n is None:
            return list(self._history)
        return list(self._history)[-n:]
