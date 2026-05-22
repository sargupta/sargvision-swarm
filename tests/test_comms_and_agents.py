"""Comms bus + LLM agent scaffold (mock backend)."""

from sargvision_swarm.agents.backends import MockBackend, make_backend
from sargvision_swarm.agents.drone_agent import DroneAgent
from sargvision_swarm.comms.bus import Channels, InMemoryBus
from sargvision_swarm.core import SwarmState


def test_bus_publish_subscribe_roundtrip():
    bus = InMemoryBus()
    received = []
    bus.subscribe("test/topic", received.append)
    bus.publish("test/topic", {"x": 1})
    assert received == [{"x": 1}]


def test_bus_caches_latest():
    bus = InMemoryBus()
    bus.publish("t/1", {"v": 42})
    assert bus.latest("t/1") == {"v": 42}


def test_channels_names_are_stable():
    assert Channels.pose(7) == "swarm/7/pose"
    assert Channels.intent(7) == "swarm/7/intent"


def test_mock_backend_returns_json():
    backend = MockBackend(seed=0)
    out = backend.complete("system", "user")
    assert "intent" in out


def test_make_backend_default_is_mock():
    backend = make_backend()
    assert isinstance(backend, MockBackend)


def test_drone_agent_decide_returns_intent():
    bus = InMemoryBus()
    swarm = SwarmState.random_init(3, seed=0)
    agent = DroneAgent(drone_id=0, bus=bus)
    obs = agent.observe(swarm.drones[0], swarm.drones[1:])
    decision = agent.decide(obs)
    assert decision.intent
    agent.broadcast(decision, obs)
    assert bus.latest(Channels.intent(0)) is not None


def test_orchestrator_plans_each_scenario():
    from sargvision_swarm.orchestrator import MissionGoal, MissionPlanner

    planner = MissionPlanner()
    for scenario in ("flock", "formation_v", "coverage", "hover"):
        plan = planner.plan(MissionGoal(goal_text="t", n_drones=12, scenario=scenario))
        assert plan.scenario == scenario
        assert len(plan.bundles) == 12
