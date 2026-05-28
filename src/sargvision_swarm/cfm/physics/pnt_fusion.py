"""Sovereign multi-source PNT fusion with spoof-outlier rejection.

Round-2 reframe (files 81/94): NavIC is NOT primary PNT — it's degraded (3 sats
operational Mar-2026) and, in the contested LAC/LoC geography, the engagement
point is GNSS-denied regardless. The defensible moat is robust FUSION of many
degraded sources: GPS + NavIC (anti-spoof diversity) + visual-inertial odometry
+ magnetic-anomaly nav (the small-drone-portable quantum-adjacent piece) +
swarm-relative ranging.

This module fuses heterogeneous PNT sources by inverse-variance weighting and
rejects a spoofed/faulty source via robust-consensus outlier detection — so a
single spoofed GPS cannot pull the fused fix, exactly as multi-constellation
RAIM defeats a single-constellation spoofer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PNTSource:
    """One PNT source's position estimate + its 1-sigma uncertainty.

    name : "gps" / "navic" / "vio" / "magnav" / "swarm_relative" / ...
    position_enu_m : (3,) estimated position.
    sigma_m : 1-sigma position uncertainty (metres). Larger = trusted less.
    available : whether the source currently has a valid fix.
    """

    name: str
    position_enu_m: np.ndarray
    sigma_m: float
    available: bool = True


@dataclass
class PNTFusionResult:
    fused_position_enu_m: np.ndarray
    fused_sigma_m: float
    used_sources: list[str]
    rejected_sources: list[str] = field(default_factory=list)

    @property
    def has_fix(self) -> bool:
        return len(self.used_sources) > 0


def fuse_pnt(
    sources: list[PNTSource],
    *,
    outlier_k: float = 3.0,
    min_sources_for_rejection: int = 3,
) -> PNTFusionResult:
    """Fuse PNT sources by inverse-variance weighting with outlier rejection.

    Algorithm:
      1. Take available sources.
      2. Form a robust consensus (inverse-variance weighted mean).
      3. Reject any source whose distance from consensus exceeds
         outlier_k × (its own sigma + consensus sigma) — a spoofed source
         disagrees far beyond its claimed precision.
      4. Re-fuse the survivors.

    Rejection only runs with >= min_sources_for_rejection available sources
    (you cannot outvote a spoofer with only two sources).
    """
    avail = [s for s in sources if s.available and s.sigma_m > 0]
    if not avail:
        return PNTFusionResult(
            fused_position_enu_m=np.array([np.nan, np.nan, np.nan]),
            fused_sigma_m=float("inf"),
            used_sources=[],
            rejected_sources=[s.name for s in sources],
        )

    def _weighted(srcs: list[PNTSource]) -> tuple[np.ndarray, float]:
        w = np.array([1.0 / (s.sigma_m**2) for s in srcs])
        P = np.stack([np.asarray(s.position_enu_m, dtype=np.float64).reshape(3) for s in srcs])
        mean = (P * w[:, None]).sum(axis=0) / w.sum()
        fused_sigma = float(np.sqrt(1.0 / w.sum()))
        return mean, fused_sigma

    rejected: list[str] = []
    survivors = avail
    if len(avail) >= min_sources_for_rejection:
        # Robust consensus for outlier detection: the coordinate-wise MEDIAN is
        # resistant to a single spoofer EVEN IF it claims a tiny sigma (an
        # inverse-variance mean would be pulled toward such a spoofer and let it
        # escape — the bug this guards against).
        P_all = np.stack(
            [np.asarray(s.position_enu_m, dtype=np.float64).reshape(3) for s in avail]
        )
        robust_consensus = np.median(P_all, axis=0)
        median_sigma = float(np.median([s.sigma_m for s in avail]))
        survivors = []
        for s in avail:
            d = float(np.linalg.norm(np.asarray(s.position_enu_m).reshape(3) - robust_consensus))
            gate = outlier_k * (s.sigma_m + median_sigma)
            if d <= gate:
                survivors.append(s)
            else:
                rejected.append(s.name)
        if not survivors:  # everyone rejected → fall back to all available
            survivors = avail
            rejected = []

    fused, fused_sigma = _weighted(survivors)
    return PNTFusionResult(
        fused_position_enu_m=fused,
        fused_sigma_m=fused_sigma,
        used_sources=[s.name for s in survivors],
        rejected_sources=rejected,
    )
