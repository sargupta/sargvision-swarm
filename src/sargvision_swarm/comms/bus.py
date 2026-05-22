"""In-memory pubsub. Mirrors Zenoh `Session.declare_subscriber` shape."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass
class Topic:
    """A namespaced topic. Latest value cached for late subscribers."""

    name: str
    history: int = 16  # ring buffer length

    def __post_init__(self) -> None:
        self._buffer: deque[Any] = deque(maxlen=self.history)
        self._latest: Any = None
        self._subscribers: list[Callable[[Any], None]] = []
        self._lock = RLock()

    def publish(self, payload: Any) -> None:
        with self._lock:
            self._buffer.append(payload)
            self._latest = payload
            subscribers = list(self._subscribers)
        for cb in subscribers:
            try:
                cb(payload)
            except Exception:  # pragma: no cover — best-effort
                pass

    def subscribe(self, callback: Callable[[Any], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def latest(self) -> Any:
        with self._lock:
            return self._latest

    def history_list(self) -> list[Any]:
        with self._lock:
            return list(self._buffer)


@dataclass
class InMemoryBus:
    """Hub for all topics. Single instance per process."""

    topics: dict[str, Topic] = field(default_factory=lambda: defaultdict(lambda: None))  # type: ignore

    def __post_init__(self) -> None:
        # Use real dict (typed) — defaultdict above is only for first-touch creation pattern
        self.topics = {}
        self._lock = RLock()

    def topic(self, name: str, history: int = 16) -> Topic:
        with self._lock:
            if name not in self.topics:
                self.topics[name] = Topic(name=name, history=history)
            return self.topics[name]

    def publish(self, name: str, payload: Any) -> None:
        self.topic(name).publish(payload)

    def subscribe(self, name: str, callback: Callable[[Any], None]) -> Callable[[], None]:
        return self.topic(name).subscribe(callback)

    def latest(self, name: str) -> Any:
        return self.topic(name).latest()


# Standard channel names mirroring planned Zenoh deployment.
class Channels:
    """Stable topic-name builders."""

    @staticmethod
    def pose(drone_id: int) -> str:
        return f"swarm/{drone_id}/pose"

    @staticmethod
    def health(drone_id: int) -> str:
        return f"swarm/{drone_id}/health"

    @staticmethod
    def intent(drone_id: int) -> str:
        return f"swarm/{drone_id}/intent"

    @staticmethod
    def mission_state() -> str:
        return "swarm/mission/state"

    @staticmethod
    def task_bundle(drone_id: int) -> str:
        return f"swarm/{drone_id}/task_bundle"
