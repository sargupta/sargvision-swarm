"""LiveSession — one-tick-at-a-time swarm runner for streaming demos.

Encapsulates the same physics + protocols as `runner.rollout()` but exposes
`step()` so the Gradio app can render a frame between ticks.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from sargvision_swarm.agents.backends import MockBackend
from sargvision_swarm.agents.peer_dialogue import PeerDialogue
from sargvision_swarm.comms import (
    BandwidthTracker,
    Channels,
    CommModel,
    InMemoryBus,
    MessageLog,
    Protocol,
    WireMessage,
)
from sargvision_swarm.comms.protocols import IntentPayload
from sargvision_swarm.core import ReflexParams, SwarmState, compose_reflex
from sargvision_swarm.core.bvc import bvc_safe_velocity
from sargvision_swarm.orchestrator import EDCBBA, MissionGoal, MissionPlanner, SwarmRaft, Task
from sargvision_swarm.sim import SimConfig, SimpleSim
from sargvision_swarm.viz.live_frame import FloatingEvent, RecentMessage, TrailHistory


@dataclass
class TickResult:
    """What changed during the latest tick — fed to the renderer."""

    new_messages: list[WireMessage] = field(default_factory=list)
    floating_events: list[FloatingEvent] = field(default_factory=list)
    bft_flash_voters: set[int] = field(default_factory=set)
    cbba_flash_cells: dict[str, tuple[float, float]] = field(default_factory=dict)
    event_log_lines: list[str] = field(default_factory=list)


class LiveSession:
    """One swarm scenario as a step-able generator."""

    def __init__(
        self,
        n_drones: int = 30,
        scenario: str = "coverage",
        seed: int | None = 42,
        comm_range_m: float = 15.0,
        intent_refresh_every: int = 10,
    ) -> None:
        self.n_drones = n_drones
        self.scenario = scenario
        self.seed = seed
        self.intent_refresh_every = intent_refresh_every
        self._rng = random.Random(seed)

        planner = MissionPlanner()
        self.plan = planner.plan(MissionGoal(goal_text="live", n_drones=n_drones, scenario=scenario))
        self.swarm = SwarmState.random_init(n_drones, seed=seed)
        self.sim = SimpleSim(SimConfig(dt=0.05), seed=seed)
        self.bus = InMemoryBus()
        self.comm = CommModel(range_m=comm_range_m, rng_seed=seed or 0)
        self.peer = PeerDialogue(drones=self.swarm.drones, comm=self.comm, rng_seed=seed or 0)
        self.backend = MockBackend(seed=seed)
        self.raft = SwarmRaft(rng_seed=seed or 0)
        self.intents: dict[int, str] = {d.id: "hold_formation" for d in self.swarm.drones}

        self.cbba_tasks: list[Task] = []
        if scenario == "coverage":
            for i, b in enumerate(self.plan.bundles):
                if b.goal_pos:
                    self.cbba_tasks.append(Task(id=f"cell-{i:02d}", pos=b.goal_pos))
        self.cbba = EDCBBA(drones_pos=self.swarm.positions, tasks=self.cbba_tasks)

        self.message_log = MessageLog(capacity=20000)
        self.bandwidth = BandwidthTracker(window_s=5.0)
        self.trails = TrailHistory(capacity=12)
        self.recent_msgs: deque[RecentMessage] = deque(maxlen=200)
        self.floating: deque[FloatingEvent] = deque(maxlen=40)
        self.event_log: deque[str] = deque(maxlen=300)
        self.bft_flash: set[int] = set()
        self.bft_flash_ttl = 0
        self.cbba_flash: dict[str, tuple[float, float]] = {}
        self.cbba_flash_ttl = 0
        self.bft_count = 0
        self.cbba_count = 0
        # Cross-step history surfaces in the frame for the console.
        self.bft_history: list[dict] = []
        self.cbba_history: list[dict] = []
        self.step_i = 0
        self.goal_pos = np.array(self.plan.goal_pos)

    # ── Public API ─────────────────────────────────────────────────────

    def step(self) -> TickResult:
        """Advance the swarm one tick. Returns events for the renderer."""
        result = TickResult()
        positions = self.swarm.positions
        velocities = self.swarm.velocities

        # ── Reflex ──
        if self.plan.scenario == "formation_v":
            per_goal = np.array(
                [b.goal_pos if b.goal_pos else self.goal_pos.tolist() for b in self.plan.bundles]
            )
            slot_pull = (per_goal - positions) * 1.4 - velocities * 0.6
            v_cmd = bvc_safe_velocity(positions, slot_pull, safety_radius=0.9, dt=0.1)
        elif self.plan.scenario == "coverage":
            per_goal = np.array(
                [b.goal_pos if b.goal_pos else self.goal_pos.tolist() for b in self.plan.bundles]
            )
            slot_pull = (per_goal - positions) * 1.2 - velocities * 0.5
            v_cmd = bvc_safe_velocity(positions, slot_pull, safety_radius=1.0, dt=0.1)
        elif self.plan.scenario == "hover":
            v_cmd = (self.goal_pos - positions) * 1.0 - velocities * 0.8
        else:
            v_cmd = compose_reflex(positions, velocities, algorithm=self.plan.algorithm, goal_pos=self.goal_pos)

        self.sim.step(self.swarm, v_cmd)
        t = self.swarm.t

        # Update trails
        for d in self.swarm.drones:
            self.trails.push(d.id, float(d.pos[0]), float(d.pos[1]))

        # ── LLM intent refresh ──
        if self.step_i % self.intent_refresh_every == 0:
            for d in self.swarm.drones:
                user = f"id={d.id} pos={d.pos.tolist()} battery={d.battery:.2f}"
                raw = self.backend.complete("terse", user, max_tokens=64)
                try:
                    import json
                    self.intents[d.id] = json.loads(raw).get("intent", "hold_formation")
                except Exception:
                    self.intents[d.id] = "hold_formation"

                m = WireMessage.make(
                    src=d.id,
                    protocol=Protocol.GRPC,
                    topic=f"grpc/cognition/{d.id}/intent",
                    payload=IntentPayload(intent=self.intents[d.id], rationale="mock"),
                    t=t,
                )
                result.new_messages.append(m)

        # ── Peer dialogue (A2A + Zenoh + MAVLink) ──
        wire = self.peer.emit_round(t, self.intents)
        result.new_messages.extend(wire)

        # ── BFT vote at step 30 and 100 ──
        if self.step_i == 30:
            passed, votes = self.raft.propose("advance_phase:engage")
            for v in votes:
                m = WireMessage.make(
                    src=v.voter_id,
                    protocol=Protocol.BFT,
                    topic="bft/swarm-raft/vote",
                    payload=v,
                    t=t,
                )
                result.new_messages.append(m)
            self.bft_flash = {v.voter_id for v in votes}
            self.bft_flash_ttl = 12
            self.bft_count += 1
            tag = f"BFT PASSED · advance_phase:engage  ({sum(1 for v in votes if v.decision=='yes')}/7)"
            result.floating_events.append(FloatingEvent(text=tag, x=0.0, y=24.0, color=(252, 211, 77)))
            result.event_log_lines.append(f"t={t:.2f}s  ⚖  BFT vote PASSED — advance_phase:engage")
            self.bft_history.append(
                {
                    "t": float(t),
                    "proposal": "advance_phase:engage",
                    "passed": bool(passed),
                    "yes": sum(1 for v in votes if v.decision == "yes"),
                    "no": sum(1 for v in votes if v.decision == "no"),
                    "voters": [int(v.voter_id) for v in votes],
                    "byzantine": [],
                }
            )
        if self.step_i == 100 and self.scenario != "hover":
            byz = set(self._rng.sample(range(self.n_drones), k=min(2, self.n_drones)))
            passed, votes = self.raft.propose("re-plan:wind-gust", byzantine_ids=byz)
            for v in votes:
                m = WireMessage.make(
                    src=v.voter_id,
                    protocol=Protocol.BFT,
                    topic="bft/swarm-raft/vote",
                    payload=v,
                    t=t,
                )
                result.new_messages.append(m)
            self.bft_flash = {v.voter_id for v in votes}
            self.bft_flash_ttl = 14
            self.bft_count += 1
            outcome = "PASSED" if passed else "FAILED"
            tag = f"BFT {outcome} · re-plan w/ {len(byz)} spoofed drones"
            color = (252, 211, 77) if passed else (248, 113, 113)
            result.floating_events.append(FloatingEvent(text=tag, x=0.0, y=-24.0, color=color))
            result.event_log_lines.append(
                f"t={t:.2f}s  ⚖  BFT vote {outcome} — re-plan (byzantine={sorted(byz)})"
            )
            self.bft_history.append(
                {
                    "t": float(t),
                    "proposal": "re-plan:wind-gust",
                    "passed": bool(passed),
                    "yes": sum(1 for v in votes if v.decision == "yes"),
                    "no": sum(1 for v in votes if v.decision == "no"),
                    "voters": [int(v.voter_id) for v in votes],
                    "byzantine": [int(b) for b in byz],
                }
            )

        # ── ED-CBBA bidding (coverage scenario, every 25 steps) ──
        if self.cbba_tasks and self.step_i % 25 == 0:
            self.cbba.drones_pos = self.swarm.positions
            bids = self.cbba.round()
            for bid in bids:
                m = WireMessage.make(
                    src=bid.bidder_id,
                    protocol=Protocol.ZENOH,
                    topic="cbba/bid",
                    payload=bid,
                    t=t,
                )
                result.new_messages.append(m)
            if bids:
                self.cbba_count += 1
                for bid in bids:
                    self.cbba_history.append(
                        {
                            "t": float(t),
                            "task_id": str(bid.task_id),
                            "bidder_id": int(bid.bidder_id),
                            "bid_score": float(bid.bid_score),
                        }
                    )
                self.cbba_flash = {}
                for bid in bids[:6]:
                    task_pos = next((t.pos for t in self.cbba_tasks if t.id == bid.task_id), None)
                    if task_pos:
                        self.cbba_flash[bid.task_id] = (task_pos[0], task_pos[1])
                self.cbba_flash_ttl = 10
                result.cbba_flash_cells = dict(self.cbba_flash)
                result.event_log_lines.append(
                    f"t={t:.2f}s  📋 CBBA · {len(bids)} bids · drone {bids[0].bidder_id} claims {bids[0].task_id}"
                )

        # Decay BFT / CBBA flashes
        if self.bft_flash_ttl > 0:
            result.bft_flash_voters = set(self.bft_flash)
            self.bft_flash_ttl -= 1
            if self.bft_flash_ttl == 0:
                self.bft_flash = set()
        if self.cbba_flash_ttl > 0:
            result.cbba_flash_cells = dict(self.cbba_flash)
            self.cbba_flash_ttl -= 1
            if self.cbba_flash_ttl == 0:
                self.cbba_flash = {}

        # ── Bookkeeping ──
        for m in result.new_messages:
            self.message_log.append(m)
            self.bandwidth.record(t, m.protocol.value, m.bytes_size)

        # Record RecentMessage for renderer
        for m in result.new_messages:
            src_xy = (float(self.swarm.drones[m.src].pos[0]), float(self.swarm.drones[m.src].pos[1]))
            if m.dst is not None and 0 <= m.dst < self.n_drones:
                dst_xy = (float(self.swarm.drones[m.dst].pos[0]), float(self.swarm.drones[m.dst].pos[1]))
            else:
                dst_xy = None
            self.recent_msgs.append(
                RecentMessage(
                    src_id=m.src,
                    dst_id=m.dst,
                    protocol=m.protocol,
                    age=0,
                    src_xy=src_xy,
                    dst_xy=dst_xy,
                )
            )
            if m.protocol == Protocol.A2A and m.dst is not None:
                # Surface negotiate.yield as a floating event
                payload = m.payload
                if isinstance(payload, dict) and payload.get("method") == "negotiate.yield":
                    self.event_log.append(
                        f"t={t:.2f}s  ↻ drone {m.src} → drone {m.dst}: negotiate.yield"
                    )

        # Age recent messages
        for rm in list(self.recent_msgs):
            rm.age += 1
        # Age floating events
        new_floats = []
        for fe in list(self.floating):
            fe.age += 1
            if fe.age < 8:
                new_floats.append(fe)
        self.floating = deque(new_floats, maxlen=40)
        for fe in result.floating_events:
            self.floating.append(fe)

        # Append event log lines
        for line in result.event_log_lines:
            self.event_log.append(line)

        self.step_i += 1
        return result

    # ── Snapshot accessors for renderer ────────────────────────────────

    def comm_adjacency(self) -> np.ndarray:
        return self.comm.adjacency(self.swarm.positions)

    def render_stats(self) -> dict:
        now = self.swarm.t
        rates = self.bandwidth.rates_by_protocol(now)
        total_msgs = len(self.message_log)
        msgs_per_s = sum(r["msgs_per_s"] for r in rates.values())
        by_proto = {p: int(r["total_msgs"]) for p, r in rates.items()}
        return {
            "t": now,
            "total_msgs": total_msgs,
            "msgs_per_s": msgs_per_s,
            "bft_count": self.bft_count,
            "cbba_count": self.cbba_count,
            "by_proto": by_proto,
        }

    def floating_for_render(self) -> list[FloatingEvent]:
        return list(self.floating)

    def recent_for_render(self) -> list[RecentMessage]:
        return list(self.recent_msgs)
