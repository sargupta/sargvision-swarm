"""Bandwidth accounting — bytes per link per second, rolling rate."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class BandwidthTracker:
    """Tracks rolling byte / message rates per protocol."""

    window_s: float = 5.0
    _events: deque[tuple[float, str, int]] = field(default_factory=lambda: deque(maxlen=10_000))

    def record(self, t: float, protocol: str, bytes_n: int) -> None:
        self._events.append((t, protocol, bytes_n))
        self._prune(t)

    def _prune(self, now: float) -> None:
        while self._events and (now - self._events[0][0]) > self.window_s:
            self._events.popleft()

    def rates_by_protocol(self, now: float) -> dict[str, dict[str, float]]:
        """Per protocol: {bytes_per_s, msgs_per_s}."""
        self._prune(now)
        per: dict[str, list[int]] = {}
        for _, proto, b in self._events:
            per.setdefault(proto, []).append(b)
        out: dict[str, dict[str, float]] = {}
        denom = max(self.window_s, 1e-6)
        for proto, byts in per.items():
            out[proto] = {
                "bytes_per_s": sum(byts) / denom,
                "msgs_per_s": len(byts) / denom,
                "total_bytes": sum(byts),
                "total_msgs": len(byts),
            }
        return out

    def total_bytes_per_s(self, now: float) -> float:
        self._prune(now)
        return sum(b for _, _, b in self._events) / max(self.window_s, 1e-6)
