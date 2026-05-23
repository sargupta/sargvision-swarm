"""Mission planner. LangGraph DAG with deterministic fallback.

LangGraph is optional at import time so the demo runs even if the dep isn't
fully resolved. The deterministic path covers the demo scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from sargvision_swarm.core.state import Role


class MissionGoal(BaseModel):
    """High-level mission spec."""

    goal_text: str
    n_drones: int
    scenario: str = "flock"   # flock | formation_v | coverage | hover | sead_ingress


class TaskBundle(BaseModel):
    drone_id: int
    role: str
    goal_pos: list[float] | None = None
    keepout_radius: float = 0.8


class MissionPlan(BaseModel):
    """Output of planner — fed downstream into reflex + agents."""

    scenario: str
    algorithm: str   # boids | olfati_saber
    goal_pos: list[float]
    bundles: list[TaskBundle]
    rationale: str = ""


@dataclass
class MissionPlanner:
    """Translates MissionGoal → executable MissionPlan.

    Deterministic by default (no LLM). LangGraph wiring left as a TODO when
    the orchestrator escalates beyond demo scenarios.
    """

    use_langgraph: bool = False
    _lg_graph: object | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.use_langgraph:
            self._build_langgraph()

    def _build_langgraph(self) -> None:  # pragma: no cover — optional dep
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            self.use_langgraph = False
            return

        class State(BaseModel):
            goal: MissionGoal
            plan: MissionPlan | None = None

        g: StateGraph = StateGraph(State)
        g.add_node("decide", lambda s: {"plan": self._decide(s.goal)})
        g.set_entry_point("decide")
        g.add_edge("decide", END)
        self._lg_graph = g.compile()

    def plan(self, goal: MissionGoal) -> MissionPlan:
        if self.use_langgraph and self._lg_graph is not None:  # pragma: no cover
            out = self._lg_graph.invoke({"goal": goal})  # type: ignore[attr-defined]
            return out["plan"]
        return self._decide(goal)

    # ----- deterministic scenario library --------------------------------

    def _decide(self, goal: MissionGoal) -> MissionPlan:
        scenario = goal.scenario
        if scenario == "flock":
            return MissionPlan(
                scenario=scenario,
                algorithm="boids",
                goal_pos=[15.0, 0.0, 6.0],
                bundles=[
                    TaskBundle(drone_id=i, role=Role.WORKER.value)
                    for i in range(goal.n_drones)
                ],
                rationale="Boids reflex + BVC. No explicit goal; emergent flock cohesion.",
            )
        if scenario == "formation_v":
            return MissionPlan(
                scenario=scenario,
                algorithm="olfati_saber",
                goal_pos=[20.0, 0.0, 6.0],
                bundles=self._v_formation_bundles(goal.n_drones),
                rationale="Olfati-Saber Algorithm 3 toward V-shaped goal formation.",
            )
        if scenario == "coverage":
            return MissionPlan(
                scenario=scenario,
                algorithm="olfati_saber",
                goal_pos=[0.0, 0.0, 8.0],
                bundles=self._coverage_bundles(goal.n_drones),
                rationale="Distributed coverage — drones disperse to assigned cells.",
            )
        if scenario == "sead_ingress":
            return MissionPlan(
                scenario=scenario,
                algorithm="chanakya_geodesic",
                goal_pos=[0.0, 32.0, 6.0],
                bundles=[
                    TaskBundle(drone_id=i, role=Role.WORKER.value)
                    for i in range(goal.n_drones)
                ],
                rationale="CHANAKYA Riemannian geodesic ingress across hostile IADS.",
            )
        if scenario == "migration":
            return MissionPlan(
                scenario=scenario,
                algorithm="governed_migration",
                goal_pos=[0.0, 25.0, 6.0],
                bundles=[
                    TaskBundle(drone_id=i, role=Role.WORKER.value)
                    for i in range(goal.n_drones)
                ],
                rationale=(
                    "GOVERNED MIGRATION Leh → forward LAC. 100 drones load-balance "
                    "across Khardung La / Zoji La / Tanglang La passes, avoiding glacier "
                    "storms + wind shear. Per-zone capacity, hazard-aware routing."
                ),
            )
        # Default: hover in place
        return MissionPlan(
            scenario="hover",
            algorithm="olfati_saber",
            goal_pos=[0.0, 0.0, 5.0],
            bundles=[TaskBundle(drone_id=i, role=Role.WORKER.value) for i in range(goal.n_drones)],
            rationale="Hover. No movement intent.",
        )

    @staticmethod
    def _v_formation_bundles(n: int) -> list[TaskBundle]:
        bundles: list[TaskBundle] = []
        half = n // 2
        for i in range(n):
            # left wing or right wing
            row = (i // 2) + 1
            x = -row * 1.5
            y = (1 if i % 2 == 0 else -1) * row * 2.0
            z = 6.0
            role = Role.LEADER.value if i == 0 else Role.WORKER.value
            if i == 0:
                x, y = 0.0, 0.0
            bundles.append(
                TaskBundle(drone_id=i, role=role, goal_pos=[x, y, z], keepout_radius=1.0)
            )
        _ = half  # not used; kept for readability
        return bundles

    @staticmethod
    def _coverage_bundles(n: int) -> list[TaskBundle]:
        import math

        bundles: list[TaskBundle] = []
        radius = max(5.0, n * 0.3)
        for i in range(n):
            theta = 2 * math.pi * i / n
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            bundles.append(
                TaskBundle(
                    drone_id=i,
                    role=Role.WORKER.value,
                    goal_pos=[x, y, 8.0],
                    keepout_radius=0.8,
                )
            )
        return bundles
