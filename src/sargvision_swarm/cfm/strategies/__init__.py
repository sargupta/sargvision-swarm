"""C-UAS posture strategies — 4 named tactics for the defender swarm.

Each strategy returns per-defender-drone goal positions given the asset(s) to
protect and the current hostile wave state. Strategies are pure functions —
state lives in the scenario + wave, not in the strategy.

Strategies:

  - LAYERED       : layered defence — concentric rings, outer drones engage
                    earliest, inner drones backstop leakage. Best vs mixed wave.

  - POINT_DEFENCE : tight bubble immediately around asset, accept some leakage
                    in exchange for guaranteed last-ring kill. Best vs OWA/cruise
                    where outer engagement is limited by interceptor speed.

  - AREA_DEFENCE  : defenders spread broadly across azimuth sector, engage on
                    first detection. Best vs FPV mass where dispersion + early
                    intercept matters most.

  - MOBILE_CAP    : Combat Air Patrol — defenders orbit at intermediate radius
                    and dynamically converge on threats. Best vs unknown-axis
                    waves where the threat azimuth shifts unpredictably.
"""

from __future__ import annotations

from .c_uas import (
    CUASStrategy,
    area_defence_goals,
    layered_defence_goals,
    mobile_cap_goals,
    point_defence_goals,
    plan_goals_for_strategy,
)

__all__ = [
    "CUASStrategy",
    "area_defence_goals",
    "layered_defence_goals",
    "mobile_cap_goals",
    "point_defence_goals",
    "plan_goals_for_strategy",
]
