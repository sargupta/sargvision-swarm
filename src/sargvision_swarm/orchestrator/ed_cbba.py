"""Event-Driven CBBA — Consensus-Based Bundle Algorithm with event triggers.

ED-CBBA cuts radio traffic ~52% vs vanilla CBBA by only re-bidding when a
neighbor's known winning bid changes, not on every cycle.

Reference: arXiv 2509.06481.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sargvision_swarm.comms.protocols import CBBABid


@dataclass
class Task:
    id: str
    pos: list[float]
    reward: float = 1.0


@dataclass
class EDCBBA:
    drones_pos: np.ndarray  # (N, 3)
    tasks: list[Task]
    bundle_size: int = 3

    _winning_bid: dict[str, float] = field(default_factory=dict)
    _winning_bidder: dict[str, int] = field(default_factory=dict)
    _bids_history: list[CBBABid] = field(default_factory=list)

    def _bid_value(self, drone_id: int, task: Task) -> float:
        pos = self.drones_pos[drone_id]
        d = float(np.linalg.norm(pos - np.array(task.pos)))
        return task.reward / max(d, 0.5)

    def round(self) -> list[CBBABid]:
        """One ED-CBBA round. Returns the bids emitted this round."""
        emitted: list[CBBABid] = []
        for drone_id in range(self.drones_pos.shape[0]):
            scored = sorted(
                ((self._bid_value(drone_id, t), t) for t in self.tasks),
                key=lambda x: x[0],
                reverse=True,
            )
            bundle = scored[: self.bundle_size]
            for score, task in bundle:
                current_best = self._winning_bid.get(task.id, -1.0)
                if score > current_best + 1e-6:
                    bid = CBBABid(
                        bidder_id=drone_id,
                        task_id=task.id,
                        bid_score=float(score),
                        bundle=[t.id for _, t in bundle],
                    )
                    self._winning_bid[task.id] = score
                    self._winning_bidder[task.id] = drone_id
                    emitted.append(bid)
                    self._bids_history.append(bid)
        return emitted

    def assignment(self) -> dict[str, int]:
        return dict(self._winning_bidder)
