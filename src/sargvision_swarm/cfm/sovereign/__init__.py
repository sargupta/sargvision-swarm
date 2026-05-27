"""Sovereign-stack integration mocks.

These modules provide interface-compatible mocks for Indian sovereign defence
infrastructure. They are NOT production integrations — they are documented
interface schemas + simulated responses suitable for demo + development.

Production integration with actual NavIC receivers, Bhuvan tile servers, IACCS
message buses, and Akashteer tactical networks happens through these same
interface contracts, swapping the mock implementation for the real one.

Modules:
  - navic       — NavICReceiver (IRNSS L1+L5 PNT solution mock)
  - bhuvan      — BhuvanTileService (ISRO geospatial tile mock)
  - iaccs       — IACCSMessageBus (air picture + airspace coordination mock)
  - akashteer   — AkashteerCueChannel (tactical AD cueing mock)
"""

from __future__ import annotations

from .akashteer import AkashteerCueChannel, TacticalCue
from .bhuvan import BhuvanTileService, TileRequest, TileResponse
from .iaccs import IACCSMessageBus, IACCSTrack, IACCSMessage
from .navic import NavICReceiver, PNTSolution

__all__ = [
    "AkashteerCueChannel",
    "TacticalCue",
    "BhuvanTileService",
    "TileRequest",
    "TileResponse",
    "IACCSMessageBus",
    "IACCSTrack",
    "IACCSMessage",
    "NavICReceiver",
    "PNTSolution",
]
