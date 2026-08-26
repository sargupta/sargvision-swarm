"""Physics-realism layer — RF link budgets, battery/endurance, environment, PNT fusion.

Round-1 brutal-honesty audit (file 64) + Round-2 (files 81/90/94) flagged the
simulation as point-mass: perfect comms, no battery, GNSS assumed, no wind. This
package adds physics-correct models the coordination layer must respect.

Modules:
  - rf_link      — Friis link budget, jammer-degraded SNR, Shannon capacity
  - battery      — pack-level energy, endurance with cold + altitude derate,
                   chemistry presets incl. 2026 Li-metal upside
  - environment  — ISA density, momentum-theory hover power, wind tolerance,
                   altitude derate (Ladakh / high-altitude operations)
  - pnt_fusion   — sovereign MULTI-SOURCE PNT fusion (GPS+NavIC+VIO+MagNav+
                   swarm-relative), inverse-variance weighting + spoof-outlier
                   rejection (Round-2 reframe: fusion, not NavIC-primary)

These are deliberately parameterised + physics-anchored, not tuned to flatter the
demo — several existing sim strategies will become infeasible under them, which
is the point.
"""

from __future__ import annotations

from .battery import BatteryModel, Chemistry, CHEMISTRY_PRESETS
from .environment import hover_power_w, isa_density, altitude_derate_factor, wind_tolerated
from .pnt_fusion import PNTSource, PNTFusionResult, fuse_pnt
from .rf_link import (
    friis_path_loss_db,
    link_snr_db,
    link_capacity_mbps,
    LinkBudget,
)

__all__ = [
    "BatteryModel",
    "Chemistry",
    "CHEMISTRY_PRESETS",
    "hover_power_w",
    "isa_density",
    "altitude_derate_factor",
    "wind_tolerated",
    "PNTSource",
    "PNTFusionResult",
    "fuse_pnt",
    "friis_path_loss_db",
    "link_snr_db",
    "link_capacity_mbps",
    "LinkBudget",
]
