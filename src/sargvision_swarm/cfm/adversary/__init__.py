"""Adversarial robustness layer — named attacker models that run against the swarm.

Replaces binary jam_toggle / gnss_toggle simulations with progressive,
physics-grounded attacker models. Per file 38 (adversarial ML research) +
file 64 (brutal honesty audit) + file 47 (patent verdict), the existing
sim's binary attack flags do not validate Byzantine-resilience claims.

Five attacker classes (subset shipped in v0.1):

  - FriisJammer            — RF link degradation as function of (distance,
                              power, frequency). Friis-physics-grounded,
                              continuous SNR not binary.
  - GNSSSpoofer            — drifts target position estimates with
                              configurable accuracy + spoofing-window.
  - SensorSpoofingAttacker — injects false sensor reports into a target
                              drone's input stream.
  - SybilAttacker          — adds physically-impersonating fake drones
                              that appear valid to neighbours' comms but
                              fail attestation if challenged.
  - AdaptiveJammer         — DARPA-SC2-style cognitive EW that observes
                              swarm RF behaviour + adapts.
                              (Placeholder; defer until 2028.)

Each attacker has configurable strength / duration / target subset and exposes
a step() interface compatible with the sim event loop.
"""

from __future__ import annotations

from .gnss_spoofer import GNSSSpoofer
from .jammer import FriisJammer, friis_path_loss_db
from .sensor_spoofer import SensorSpoofingAttacker
from .sybil import SybilAttacker

__all__ = [
    "FriisJammer",
    "GNSSSpoofer",
    "SensorSpoofingAttacker",
    "SybilAttacker",
    "friis_path_loss_db",
]
