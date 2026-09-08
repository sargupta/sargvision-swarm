"""Shared safety-relevant constants.

Deliberately dependency-free. These values are read by both the coordination
orchestrator and the CFM trust pipeline, and they must not diverge: two
subsystems disagreeing about a safety constant is read as evidence about the
engineering rather than about the number.

They live here, rather than in either subsystem, so that importing a constant
does not drag one subsystem's module graph onto the other's import path. The
first attempt had `orchestrator.shield` import from `cfm.trust`, which pulled
`core.twsl` and the sensor stack onto the API server's critical import path and
pushed its cold start past the boot probe in CI.
"""

from __future__ import annotations

#: Loyalty trust below which a node is isolated from tasking and from voting.
#: Not derived from measured data -- it is a tuned constant, and should be
#: replaced by a value derived from a false-eviction budget once real flight
#: data exists to measure one against.
DEFAULT_KILL_THRESHOLD = 0.35
