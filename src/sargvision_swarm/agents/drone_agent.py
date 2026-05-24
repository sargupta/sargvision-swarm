"""Per-drone LLM agent. Loop: observe → reason → broadcast → act.

The LLM produces a high-level **intent**. The reflex layer (boids/olfati-saber/BVC)
converts that intent into actual velocity commands. The LLM does NOT close the
fast control loop — that's the Brooks-subsumption discipline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from sargvision_swarm.agents.backends import LLMBackend, make_backend
from sargvision_swarm.comms.bus import Channels, InMemoryBus
from sargvision_swarm.core.state import DroneState


class Observation(BaseModel):
    """What a drone sees each cycle."""

    drone_id: int
    pos: list[float]
    vel: list[float]
    battery: float
    healthy: bool
    n_neighbors_in_range: int
    nearest_distance: float
    mission_phase: str = "idle"


class AgentDecision(BaseModel):
    """High-level intent emitted by LLM agent."""

    intent: str = Field(
        description="One of: hold_formation, advance_to_goal, yield_to_neighbor, rotate_role, report_health"
    )
    rationale: str = Field(default="")


SYSTEM_PROMPT = """You are a single drone in a swarm of N intelligent quadcopters. \
The reflex layer keeps you safe — you only emit a HIGH-LEVEL INTENT each cycle. \
Reply with strict JSON: {"intent": <one of: hold_formation, advance_to_goal, yield_to_neighbor, rotate_role, report_health>, "rationale": <short>}."""


@dataclass
class DroneAgent:
    """Per-drone LLM agent. Cheap to construct, reused across cycles."""

    drone_id: int
    bus: InMemoryBus
    backend: LLMBackend | None = None

    def __post_init__(self) -> None:
        if self.backend is None:
            self.backend = make_backend()

    def observe(
        self, drone: DroneState, neighbors: list[DroneState], phase: str = "idle"
    ) -> Observation:
        if neighbors:
            distances = [float(np.linalg.norm(drone.pos - n.pos)) for n in neighbors]
            nearest = min(distances)
        else:
            nearest = float("inf")
        return Observation(
            drone_id=self.drone_id,
            pos=drone.pos.tolist(),
            vel=drone.vel.tolist(),
            battery=drone.battery,
            healthy=drone.healthy,
            n_neighbors_in_range=len(neighbors),
            nearest_distance=nearest,
            mission_phase=phase,
        )

    def decide(self, obs: Observation) -> AgentDecision:
        user_prompt = (
            f"State: pos={obs.pos}, vel={obs.vel}, battery={obs.battery:.2f}, "
            f"neighbors={obs.n_neighbors_in_range}, nearest={obs.nearest_distance:.2f}m, "
            f"phase={obs.mission_phase}. Decide intent."
        )
        raw = self.backend.complete(SYSTEM_PROMPT, user_prompt, max_tokens=128)
        try:
            data = json.loads(raw)
            return AgentDecision(**data)
        except Exception:
            # Fall back to a safe default
            return AgentDecision(intent="hold_formation", rationale=f"parse-fail: {raw[:80]}")

    def broadcast(self, decision: AgentDecision, obs: Observation) -> None:
        """Publish intent on the swarm bus."""
        self.bus.publish(Channels.pose(self.drone_id), obs.model_dump())
        self.bus.publish(Channels.intent(self.drone_id), decision.model_dump())
