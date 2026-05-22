"""Headless rollout of a scenario — used by Gradio app + CLI."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sargvision_swarm.comms import Channels, InMemoryBus
from sargvision_swarm.core import ReflexParams, SwarmState, compose_reflex
from sargvision_swarm.orchestrator import MissionGoal, MissionPlanner
from sargvision_swarm.sim import SimConfig, SimpleSim


@dataclass
class RolloutResult:
    states: list[SwarmState] = field(default_factory=list)
    plan_rationale: str = ""

    @property
    def positions_history(self) -> np.ndarray:
        """(T, N, 3) array of positions over time."""
        return np.stack([s.positions for s in self.states], axis=0)


def rollout(
    n_drones: int = 30,
    scenario: str = "flock",
    steps: int = 200,
    seed: int | None = 42,
    snapshot_every: int = 4,
) -> RolloutResult:
    """Run a scenario headlessly. Returns snapshots for animation."""
    planner = MissionPlanner()
    plan = planner.plan(MissionGoal(goal_text="demo", n_drones=n_drones, scenario=scenario))

    swarm = SwarmState.random_init(n_drones, seed=seed)
    sim = SimpleSim(SimConfig(dt=0.05), seed=seed)
    bus = InMemoryBus()

    result = RolloutResult(plan_rationale=plan.rationale)
    goal_pos = np.array(plan.goal_pos)

    for step in range(steps):
        positions = swarm.positions
        velocities = swarm.velocities

        if plan.scenario == "formation_v":
            # Per-drone slot pull + BVC collision filter. No flocking attraction —
            # we want each drone at its own assigned slot, not centroid.
            per_drone_goal = np.array(
                [
                    bundle.goal_pos if bundle.goal_pos else goal_pos.tolist()
                    for bundle in plan.bundles
                ]
            )
            slot_pull = (per_drone_goal - positions) * 1.4 - velocities * 0.6
            from sargvision_swarm.core.bvc import bvc_safe_velocity
            v_cmd = bvc_safe_velocity(positions, slot_pull, safety_radius=0.9, dt=0.1)
        elif plan.scenario == "coverage":
            # Each drone heads to its own goal cell on a ring. No global γ-agent.
            per_drone_goal = np.array(
                [
                    bundle.goal_pos if bundle.goal_pos else goal_pos.tolist()
                    for bundle in plan.bundles
                ]
            )
            slot_pull = (per_drone_goal - positions) * 1.2 - velocities * 0.5
            from sargvision_swarm.core.bvc import bvc_safe_velocity
            v_cmd = bvc_safe_velocity(positions, slot_pull, safety_radius=1.0, dt=0.1)
        elif plan.scenario == "hover":
            target = goal_pos
            v_cmd = (target - positions) * 1.0 - velocities * 0.8
        else:
            v_cmd = compose_reflex(positions, velocities, algorithm=plan.algorithm, goal_pos=goal_pos)

        sim.step(swarm, v_cmd)

        # Publish position to the bus (Zenoh-shaped) — exercises comms path even in demo.
        for drone in swarm.drones:
            bus.publish(Channels.pose(drone.id), {"pos": drone.pos.tolist(), "t": swarm.t})

        if step % snapshot_every == 0 or step == steps - 1:
            # Deep-ish copy of state for animation history
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

    return result
