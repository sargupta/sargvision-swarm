"""EWState — Byzantine-hardened electronic-warfare state as a fusion input.

Per the Round-2 cognitive-EW finding (research_2026_05/87_cognitive_ew.md), the
coordination layer must consume the local EW picture as a *sensor input*, not
ignore it. Crucially, EW degradation is distinct from Byzantine compromise:

  - A JAMMED loyal node is still loyal — its observations are merely less
    informative. EWState reduces its fusion *confidence*, it does NOT lower its
    loyalty trust (that would let an adversary evict honest nodes by jamming
    them — a denial-of-trust attack).

  - A SPOOFED / Sybil node is Byzantine — handled by the loyalty trust pipeline
    (graph residual × hardware attestation), not by EWState.

The EWState is itself Byzantine-hardened: an attacker who forges a "no jamming"
EWState cannot inflate a node's fusion weight beyond what its attestation +
graph-agreement already permit, because confidence is multiplied by, never
substituted for, those gates.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EWState:
    """Local electronic-warfare picture at one node.

    Attributes
    ----------
    jammer_power_dbm : float
        Strongest detected jammer power at the node antenna (dBm). Ambient
        floor ~ -100 dBm; above the link margin the node's comms degrade.
    link_margin_db : float
        Remaining SNR margin on the node's primary link (dB). Falls as
        jamming rises; <= 0 means link denied.
    gnss_spoof_suspected : bool
        Multi-constellation / RAIM consistency check flagged a possible
        GNSS spoof (PNT confidence should drop).
    cognitive_adaptation_detected : bool
        The node observed an adversary emitter adapting its waveform in
        near-real-time (cognitive EW signature).
    """

    jammer_power_dbm: float = -100.0
    link_margin_db: float = 20.0
    gnss_spoof_suspected: bool = False
    cognitive_adaptation_detected: bool = False

    @property
    def confidence_factor(self) -> float:
        """Multiplier in (0, 1] applied to this node's fusion confidence.

        Degrades smoothly with shrinking link margin and on GNSS-spoof
        suspicion. Never zero — a jammed node still contributes weakly,
        and never flags the node as Byzantine.
        """
        # Link-margin term: full confidence at >=15 dB, linearly down to a
        # 0.2 floor as margin -> 0, and 0.1 if the link is fully denied.
        if self.link_margin_db <= 0:
            margin_term = 0.1
        else:
            margin_term = 0.2 + 0.8 * min(1.0, self.link_margin_db / 15.0)
        # GNSS-spoof suspicion halves PNT-derived confidence.
        spoof_term = 0.5 if self.gnss_spoof_suspected else 1.0
        # Cognitive-adaptation is a warning, mild extra discount.
        cog_term = 0.85 if self.cognitive_adaptation_detected else 1.0
        return float(margin_term * spoof_term * cog_term)

    @property
    def link_denied(self) -> bool:
        return self.link_margin_db <= 0
