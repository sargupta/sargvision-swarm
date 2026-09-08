"""Gates for claims the demonstration makes on screen.

Class gate for the 2026-08-29 red-team findings. Each of these guards a
statement the console asserts to a viewer. The failure mode they exist to catch
is not a crash -- it is the console displaying a claim that the simulation
underneath contradicts, which is discoverable by anyone who opens the recording
and is far more damaging than a visible bug.
"""

import pathlib

import pytest
from fastapi.testclient import TestClient

from sargvision_swarm.cfm.trust import DEFAULT_KILL_THRESHOLD
from sargvision_swarm.demo.live_session import DECOY_WITHHOLD_MIN_P, LiveSession
from sargvision_swarm.orchestrator.shield import ShieldParams
from sargvision_swarm.server.bridge import app
from sargvision_swarm.sim.border_strike import BorderStrikeField


def _run(steps: int) -> LiveSession:
    session = LiveSession(n_drones=24, scenario="border_strike", seed=42, comm_range_m=15.0)
    for _ in range(steps):
        session.step()
    return session


def test_protected_assets_are_civilian_and_tagged_consistently() -> None:
    """Display name and `kind` must agree, at source.

    The earlier layout named real military sites and relied on a string-rewrite
    to civilianise the recording, which changed names but not kinds — leaving a
    water plant tagged `command`. A half-applied rename reads as concealment.
    """
    expected = {
        "PRIMARY SUBSTATION": "energy",
        "METRO DATA CENTRE": "data",
        "CITY WATER PLANT": "water",
    }
    hvts = {h.name: h.kind for h in BorderStrikeField.build_default().hvts}
    assert hvts == expected, f"asset names and kinds disagree: {hvts}"


def test_no_real_place_names_in_the_scenario() -> None:
    """Nothing downstream should need to rewrite a place name out of a recording."""
    banned = ("LEH", "KARU", "DBO", "LADAKH", "NUBRA", "KHARDUNG")
    field = BorderStrikeField.build_default()
    blob = " ".join(f"{h.id} {h.name}" for h in field.hvts).upper()
    found = [b for b in banned if b in blob]
    assert not found, f"real place names still in the scenario source: {found}"


def test_trust_kill_threshold_has_a_single_source_of_truth() -> None:
    """Two subsystems disagreeing about a safety constant is itself the defect."""
    assert ShieldParams().trust_kill_threshold == DEFAULT_KILL_THRESHOLD


@pytest.mark.parametrize(
    "endpoint",
    ["/scenario/coverage", "/vyuha/wall", "/jam", "/gnss/toggle", "/hijack/toggle"],
)
def test_control_endpoints_refuse_without_a_token(endpoint: str) -> None:
    """The bridge is publicly reachable. Mutating it must not be.

    Previously these were unauthenticated with a CORS policy admitting any
    *.pages.dev origin, so a third party could change what a live demonstration
    was showing while it was being shown.
    """
    response = TestClient(app).post(endpoint)
    assert response.status_code in (401, 503), (
        f"{endpoint} accepted an unauthenticated state change (HTTP {response.status_code})"
    )


def test_the_flagship_scenario_actually_spawns_hostiles() -> None:
    """A counter-UAS demo with no hostiles is not a counter-UAS demo.

    Renaming the protected-asset ids silently broke the spawner, which targets
    assets by id: every recording came out with zero hostiles and a mission that
    could never start. All 218 tests passed, because none of them asserted that
    the flagship scenario has anything to defend against.
    """
    session = LiveSession(n_drones=24, scenario="border_strike", seed=42, comm_range_m=15.0)
    for _ in range(5):
        session.step()
    assert len(session.hostile_fleet.hostiles) > 0, (
        "no hostiles spawned — the scenario has nothing to defend against"
    )


def test_every_hostile_is_aimed_at_a_real_asset() -> None:
    """The spawner resolves assets by id; a rename must not orphan an axis."""
    field = BorderStrikeField.build_default()
    ids = {h.id for h in field.hvts}
    session = LiveSession(n_drones=24, scenario="border_strike", seed=42, comm_range_m=15.0)
    for _ in range(5):
        session.step()
    orphans = [
        h.callsign
        for h in session.hostile_fleet.hostiles
        if getattr(h, "target_id", None) is not None and h.target_id not in ids
    ]
    assert not orphans, f"hostiles aimed at assets that do not exist: {orphans}"
