"""Headless rollout — now emits A2A / Zenoh / MAVLink / BFT / CBBA messages each tick."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

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
from sargvision_swarm.core import SwarmState, compose_reflex
from sargvision_swarm.orchestrator import EDCBBA, MissionGoal, MissionPlanner, SwarmRaft, Task
from sargvision_swarm.sim import SimConfig, SimpleSim


@dataclass
class RolloutResult:
    states: list[SwarmState] = field(default_factory=list)
    plan_rationale: str = ""
    message_log: MessageLog = field(default_factory=MessageLog)
    bandwidth: BandwidthTracker = field(default_factory=lambda: BandwidthTracker(window_s=5.0))
    intents_by_drone: dict[int, str] = field(default_factory=dict)
    bft_events: list[dict] = field(default_factory=list)
    cbba_events: list[dict] = field(default_factory=list)
    comm_model: CommModel | None = None

    @property
    def positions_history(self) -> np.ndarray:
        return np.stack([s.positions for s in self.states], axis=0)


def rollout(
    n_drones: int = 30,
    scenario: str = "flock",
    steps: int = 200,
    seed: int | None = 42,
    snapshot_every: int = 4,
    comm_range_m: float = 15.0,
    intent_refresh_every: int = 10,
) -> RolloutResult:
    """Run a scenario w/ full comms simulation. Records every wire message."""
    rng = random.Random(seed)
    planner = MissionPlanner()
    plan = planner.plan(MissionGoal(goal_text="demo", n_drones=n_drones, scenario=scenario))

    swarm = SwarmState.random_init(n_drones, seed=seed)
    sim = SimpleSim(SimConfig(dt=0.05), seed=seed)
    bus = InMemoryBus()
    comm = CommModel(range_m=comm_range_m, rng_seed=seed or 0)
    peer = PeerDialogue(drones=swarm.drones, comm=comm, rng_seed=seed or 0)
    backend = MockBackend(seed=seed)
    raft = SwarmRaft(rng_seed=seed or 0)
    intents: dict[int, str] = {d.id: "hold_formation" for d in swarm.drones}

    # Coverage scenario: real tasks for ED-CBBA bidding
    cbba_tasks: list[Task] = []
    if scenario == "coverage":
        for i, b in enumerate(plan.bundles):
            if b.goal_pos:
                cbba_tasks.append(Task(id=f"cell-{i:02d}", pos=b.goal_pos))
    cbba = EDCBBA(drones_pos=swarm.positions, tasks=cbba_tasks)

    result = RolloutResult(plan_rationale=plan.rationale, comm_model=comm)
    goal_pos = np.array(plan.goal_pos)

    for step in range(steps):
        positions = swarm.positions
        velocities = swarm.velocities

        # ── Reflex layer (velocity command) ─────────────────────────────
        if plan.scenario == "formation_v":
            per_drone_goal = np.array(
                [b.goal_pos if b.goal_pos else goal_pos.tolist() for b in plan.bundles]
            )
            slot_pull = (per_drone_goal - positions) * 1.4 - velocities * 0.6
            from sargvision_swarm.core.bvc import bvc_safe_velocity

            v_cmd = bvc_safe_velocity(positions, slot_pull, safety_radius=0.9, dt=0.1)
        elif plan.scenario == "coverage":
            per_drone_goal = np.array(
                [b.goal_pos if b.goal_pos else goal_pos.tolist() for b in plan.bundles]
            )
            slot_pull = (per_drone_goal - positions) * 1.2 - velocities * 0.5
            from sargvision_swarm.core.bvc import bvc_safe_velocity

            v_cmd = bvc_safe_velocity(positions, slot_pull, safety_radius=1.0, dt=0.1)
        elif plan.scenario == "hover":
            v_cmd = (goal_pos - positions) * 1.0 - velocities * 0.8
        else:
            v_cmd = compose_reflex(
                positions, velocities, algorithm=plan.algorithm, goal_pos=goal_pos
            )

        sim.step(swarm, v_cmd)

        t = swarm.t

        # ── LLM intent refresh (slow loop, ~every 0.5s) ─────────────────
        if step % intent_refresh_every == 0:
            for drone in swarm.drones:
                neighbors = [d for d in swarm.drones if d.id != drone.id]
                # Drone-local observation summary
                user = (
                    f"id={drone.id} pos={drone.pos.tolist()} battery={drone.battery:.2f} "
                    f"phase={plan.scenario}"
                )
                raw = backend.complete("be terse", user, max_tokens=64)
                try:
                    import json

                    data = json.loads(raw)
                    intent = data.get("intent", "hold_formation")
                except Exception:
                    intent = "hold_formation"
                intents[drone.id] = intent

                # Also log intent broadcast as gRPC (cognition pipe to ground)
                intent_msg = WireMessage.make(
                    src=drone.id,
                    protocol=Protocol.GRPC,
                    topic=f"grpc/cognition/{drone.id}/intent",
                    payload=IntentPayload(
                        intent=intent,
                        rationale="mock-LLM cycle",
                    ),
                    t=t,
                )
                result.message_log.append(intent_msg)
                result.bandwidth.record(t, intent_msg.protocol.value, intent_msg.bytes_size)
                bus.publish(Channels.intent(drone.id), {"intent": intent})

        # ── Peer dialogue (A2A + Zenoh + MAVLink) ───────────────────────
        wire_msgs = peer.emit_round(t, intents)
        for m in wire_msgs:
            result.message_log.append(m)
            result.bandwidth.record(t, m.protocol.value, m.bytes_size)

        # ── BFT vote at phase transitions ──────────────────────────────
        if step == 30:
            passed, votes = raft.propose("advance_phase:engage")
            for v in votes:
                bft_msg = WireMessage.make(
                    src=v.voter_id,
                    protocol=Protocol.BFT,
                    topic="bft/swarm-raft/vote",
                    payload=v,
                    t=t,
                )
                result.message_log.append(bft_msg)
                result.bandwidth.record(t, bft_msg.protocol.value, bft_msg.bytes_size)
            result.bft_events.append(
                {
                    "t": round(t, 2),
                    "proposal": "advance_phase:engage",
                    "passed": passed,
                    "yes": sum(1 for v in votes if v.decision == "yes"),
                    "no": sum(1 for v in votes if v.decision == "no"),
                    "voters": [v.voter_id for v in votes],
                }
            )

        if step == 100 and scenario != "hover":
            byz = set(rng.sample(range(n_drones), k=min(2, n_drones)))
            passed, votes = raft.propose("re-plan:wind-gust", byzantine_ids=byz)
            for v in votes:
                bft_msg = WireMessage.make(
                    src=v.voter_id,
                    protocol=Protocol.BFT,
                    topic="bft/swarm-raft/vote",
                    payload=v,
                    t=t,
                )
                result.message_log.append(bft_msg)
                result.bandwidth.record(t, bft_msg.protocol.value, bft_msg.bytes_size)
            result.bft_events.append(
                {
                    "t": round(t, 2),
                    "proposal": "re-plan:wind-gust",
                    "passed": passed,
                    "yes": sum(1 for v in votes if v.decision == "yes"),
                    "no": sum(1 for v in votes if v.decision == "no"),
                    "byzantine": list(byz),
                }
            )

        # ── ED-CBBA bidding (every 25 steps for coverage scenario) ─────
        if cbba_tasks and step % 25 == 0:
            cbba.drones_pos = swarm.positions
            bids = cbba.round()
            for bid in bids:
                bid_msg = WireMessage.make(
                    src=bid.bidder_id,
                    protocol=Protocol.ZENOH,
                    topic="cbba/bid",
                    payload=bid,
                    t=t,
                )
                result.message_log.append(bid_msg)
                result.bandwidth.record(t, bid_msg.protocol.value, bid_msg.bytes_size)
            if bids:
                result.cbba_events.append(
                    {
                        "t": round(t, 2),
                        "n_bids": len(bids),
                        "assignment": dict(cbba.assignment()),
                    }
                )

        # ── Snapshot ────────────────────────────────────────────────────
        if step % snapshot_every == 0 or step == steps - 1:
            snap = SwarmState(
                drones=[
                    type(d)(
                        id=d.id,
                        pos=d.pos.copy(),
                        vel=d.vel.copy(),
                        role=d.role,
                        battery=d.battery,
                        healthy=d.healthy,
                    )
                    for d in swarm.drones
                ],
                t=swarm.t,
            )
            result.states.append(snap)

    result.intents_by_drone = dict(intents)
    return result
