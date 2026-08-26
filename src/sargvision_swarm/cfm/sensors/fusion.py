"""Multi-modal sensor fusion + cross-modal consistency.

For SARGVISION CFM, multiple modalities (radar / RF / EO / IR / acoustic) may
report the same target. Cross-modal consistency check identifies reports that
DISAGREE across modalities — a strong indicator of sensor spoofing on one
modality (e.g., LIDAR-injection attack that does not also appear in radar
or RF detection).

The cross-modal check is the multi-modal counterpart to the TWSL Dirichlet
residual — both detect inconsistency, but TWSL operates on the swarm GRAPH
while this operates on the per-target MODALITY-stack.
"""

from __future__ import annotations

import numpy as np

from .report import Modality, SensorReport


def cross_modal_consistency(
    reports: list[SensorReport],
    *,
    position_tolerance_m: float = 30.0,
    velocity_tolerance_mps: float = 5.0,
    min_modalities_agreeing: int = 2,
) -> tuple[float, list[SensorReport]]:
    """Score how consistently multiple modalities describe a target.

    Args
    ----
    reports : list of SensorReport
        Reports all believed to describe the same target.
    position_tolerance_m : float
        Maximum pairwise position discrepancy considered "agreement".
    velocity_tolerance_mps : float
        Maximum pairwise velocity discrepancy.
    min_modalities_agreeing : int
        Minimum number of modalities that must mutually agree for the target
        to be considered "consistent".

    Returns
    -------
    consistency_score : float in [0, 1]
        Fraction of reports that fall in the agreeing-cluster.
    agreeing_reports : list of SensorReport
        Reports that form the largest mutually-consistent cluster.
    """
    localised = [r for r in reports if r.has_localization]
    if len(localised) < min_modalities_agreeing:
        return 0.0, []

    # Pairwise agreement matrix
    n = len(localised)
    agree = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = localised[i], localised[j]
            assert ri.target_position_enu_m is not None  # narrowed by `has_localization`
            assert rj.target_position_enu_m is not None
            pos_ok = (
                float(np.linalg.norm(ri.target_position_enu_m - rj.target_position_enu_m))
                <= position_tolerance_m
            )
            vel_ok = True
            if ri.target_velocity_mps is not None and rj.target_velocity_mps is not None:
                vel_ok = (
                    float(np.linalg.norm(ri.target_velocity_mps - rj.target_velocity_mps))
                    <= velocity_tolerance_mps
                )
            agree[i, j] = agree[j, i] = pos_ok and vel_ok

    # Find largest mutually-agreeing clique (greedy — N is small here)
    best_cluster: list[int] = []
    for seed in range(n):
        cluster = [seed]
        for j in range(n):
            if j == seed:
                continue
            if all(agree[j, k] for k in cluster):
                cluster.append(j)
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    if len(best_cluster) < min_modalities_agreeing:
        return 0.0, []
    return len(best_cluster) / n, [localised[k] for k in best_cluster]


def fuse_reports(
    reports: list[SensorReport],
    *,
    position_tolerance_m: float = 30.0,
) -> SensorReport | None:
    """Fuse a cluster of reports into a single best-estimate report.

    Weighted by per-report confidence × attestation trust multiplier.
    Returns None if no consistent cluster.
    """
    score, agreeing = cross_modal_consistency(
        reports, position_tolerance_m=position_tolerance_m
    )
    if not agreeing:
        return None

    # Weighted mean of positions / velocities
    positions = np.stack(
        [r.target_position_enu_m for r in agreeing if r.target_position_enu_m is not None]
    )
    weights = np.array(
        [r.confidence * r.attestation.trust_multiplier for r in agreeing]
    )
    if weights.sum() <= 0:
        return None
    weights = weights / weights.sum()
    fused_pos = (positions * weights[:, None]).sum(axis=0)

    vels = [r.target_velocity_mps for r in agreeing if r.target_velocity_mps is not None]
    fused_vel = None
    if vels:
        vel_stack = np.stack(vels)
        # use weights only for the reports that had velocity
        vel_weights = np.array(
            [
                r.confidence * r.attestation.trust_multiplier
                for r in agreeing
                if r.target_velocity_mps is not None
            ]
        )
        vel_weights = vel_weights / max(vel_weights.sum(), 1e-9)
        fused_vel = (vel_stack * vel_weights[:, None]).sum(axis=0)

    avg_conf = float(np.mean([r.confidence for r in agreeing]) * score)

    # The fused report inherits the BEST attestation among agreeing reports —
    # if any contributing report was hardware-attested, the fused result is
    # treated as attested (since at least one modality verified integrity).
    best_attestation = max(
        (r.attestation for r in agreeing),
        key=lambda a: a.trust_multiplier,
    )

    # Use the first agreeing report's modality + reporter as nominal carrier;
    # downstream consumers should treat fused reports as composite.
    head = agreeing[0]
    return SensorReport(
        reporter_id=f"fused:{','.join(r.reporter_id for r in agreeing)}",
        modality=head.modality,
        target_position_enu_m=fused_pos,
        target_velocity_mps=fused_vel,
        confidence=avg_conf,
        timestamp_s=max(r.timestamp_s for r in agreeing),
        attestation=best_attestation,
        payload={"fused_from": [r.modality.value for r in agreeing], "score": score},
    )
