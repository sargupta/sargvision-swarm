"""SwarmRaft K=7 BFT committee for mission state decisions.

Tolerates ⌊(K-1)/2⌋ = 3 Byzantine drones, incl. GNSS-spoofed.
Irreversible decisions (engage, RTL, abort) require ⅔ quorum.

This is a simulation — real impl would use a Raft library + signed votes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from sargvision_swarm.comms.protocols import BFTVote


@dataclass
class SwarmRaft:
    committee_size: int = 7
    quorum_fraction: float = 2 / 3
    rng_seed: int = 0
    _term: int = 0
    _state: dict[str, str] = field(default_factory=dict)
    _votes_for_proposal: dict[str, list[BFTVote]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.rng_seed)

    def pick_committee(self, n_drones: int) -> list[int]:
        """Pick K committee members by id. Stable across phase changes."""
        k = min(self.committee_size, n_drones)
        ids = list(range(n_drones))
        # Deterministic-ish committee — every Nth drone for spread.
        step = max(1, n_drones // k)
        return ids[::step][:k]

    def propose(self, proposal: str, byzantine_ids: set[int] | None = None) -> tuple[bool, list[BFTVote]]:
        """Run a vote round on `proposal`.

        Byzantine drones flip their vote randomly. Returns (passed?, vote-list).
        """
        self._term += 1
        if byzantine_ids is None:
            byzantine_ids = set()
        committee = self.pick_committee(max(self.committee_size, 7))
        votes: list[BFTVote] = []
        yes = 0
        for voter in committee:
            if voter in byzantine_ids:
                # Byzantine: vote randomly or against
                decision = self._rng.choice(["yes", "no"])
            else:
                decision = "yes"
            votes.append(
                BFTVote(
                    voter_id=voter,
                    term=self._term,
                    proposal=proposal,
                    decision=decision,
                )
            )
            if decision == "yes":
                yes += 1
        passed = (yes / len(committee)) >= self.quorum_fraction
        self._votes_for_proposal[proposal] = votes
        if passed:
            key, val = (proposal.split(":", 1) + [""])[:2]
            self._state[key] = val
        return passed, votes

    def state(self) -> dict[str, str]:
        return dict(self._state)
