"""Governed migration must actually deliver drones to the destination.

Class gate for the 2026-08-28 bug: `goal_for_drone` routed a drone to a
corridor whenever it was short of the DESTINATION's latitude. A drone sitting
inside the corridor is also short of the destination, so it re-targeted a
corridor every tick and oscillated between corridors forever -- 0 arrivals in
300 s of simulation, with `violations` climbing past 1400.

Every existing test passed throughout, because none of them asserted that the
migration ever finishes. These do. They fail for ANY regression that stops
drones completing the route, not just the corridor-gate instance.
"""

from sargvision_swarm.demo.live_session import LiveSession
from sargvision_swarm.server.bridge import build_frame


def _run(steps: int) -> dict:
    session = LiveSession(n_drones=24, scenario="migration", seed=42, comm_range_m=15.0)
    for _ in range(steps):
        session.step()
    return build_frame(session)


def test_migration_completes_loops_within_60s() -> None:
    """Drones must reach the destination, not circle the corridors forever."""
    migration = _run(600).get("migration") or {}
    assert migration.get("completed_loops", 0) > 0, (
        "no drone completed the migration in 60 s of simulation — "
        "the route is not delivering anyone to the destination"
    )


def test_migration_does_not_accumulate_violations() -> None:
    """Oscillating drones rack up corridor violations; a working route does not."""
    migration = _run(600).get("migration") or {}
    violations = migration.get("violations", 0)
    assert violations < 50, (
        f"{violations} corridor violations in 60 s — drones are being bounced "
        "between corridors instead of transiting them"
    )


def test_migration_drones_leave_the_start_zone() -> None:
    """A stalled route can also look like everyone still queued at the origin."""
    zones = {z["id"]: z["occupancy"] for z in (_run(300).get("migration") or {}).get("zones", [])}
    assert zones.get("START", 0) < 24, "every drone is still sat in the START zone"
