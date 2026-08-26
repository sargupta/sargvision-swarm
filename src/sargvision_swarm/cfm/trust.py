"""Trust bridge — wire hardware-attested multi-modal sensor reports into TWSL.

This is the integration the §101 patent verdict (file 47) identified as the
defensible differentiator: the per-node trust score is a function of BOTH

  (a) the graph-structured Dirichlet residual of the TWSL operator — i.e. does
      this node AGREE with its neighbours about what it observes? — and
  (b) the node's HARDWARE-ATTESTATION flags (TPM + IMU triple-redundancy + RF
      tamper + secure boot), via AttestationFlags.trust_multiplier.

Final loyalty trust = graph_trust × attestation_multiplier.

The kill-switch excludes any node whose loyalty trust falls below a configurable
threshold. EWState (separate) reduces a node's fusion CONFIDENCE without
touching its loyalty — a jammed honest node is downweighted, never evicted.

Round-2 finding (file 86): a churning 3D-printed fleet creates a Sybil-injection
surface. `attestation_on_join` is the differentiating gate — a newly-joined node
cannot earn trust above a floor until it passes hardware attestation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.twsl import twsl_self_consistent_iteration
from .sensors.attestation import AttestationFlags
from .sensors.ewstate import EWState
from .sensors.report import SensorReport

#: Default loyalty-trust threshold below which the kill-switch fires.
DEFAULT_KILL_THRESHOLD = 0.35

#: Trust ceiling for a node that has NOT passed attestation-on-join.
UNATTESTED_JOIN_FLOOR = 0.2


@dataclass(frozen=True)
class TrustResult:
    """Output of the combined trust pipeline."""

    node_ids: list[str]
    graph_trust: np.ndarray          # (N,) TWSL residual-based loyalty in (0,1]
    attestation_mult: np.ndarray     # (N,) hardware-attestation multiplier in [0,1]
    loyalty_trust: np.ndarray        # (N,) graph_trust × attestation_mult
    confidence: np.ndarray           # (N,) EWState fusion-confidence in (0,1]
    killed: np.ndarray               # (N,) bool — excluded by kill-switch

    def excluded_ids(self) -> list[str]:
        return [nid for nid, k in zip(self.node_ids, self.killed) if k]

    def retained_ids(self) -> list[str]:
        return [nid for nid, k in zip(self.node_ids, self.killed) if not k]


def _cochain_matrix(reports_by_node: list[SensorReport | None]) -> np.ndarray:
    """Build an (N, 3) matrix of each node's reported target ENU position.

    Nodes with no localising report contribute their row as NaN; callers should
    have already substituted a neighbourhood prior or dropped them. Here we
    replace NaN rows with the column-mean so they neither agree nor disagree
    artificially.
    """
    rows: list[np.ndarray] = []
    for r in reports_by_node:
        if r is not None and r.target_position_enu_m is not None:
            rows.append(np.asarray(r.target_position_enu_m, dtype=np.float64).reshape(3))
        else:
            rows.append(np.array([np.nan, np.nan, np.nan]))
    X = np.stack(rows)
    # Replace NaN rows with column means (a non-committal "agree with consensus").
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X


def graph_trust_from_reports(
    adjacency: np.ndarray,
    reports_by_node: list[SensorReport | None],
    *,
    sigma_m: float = 50.0,
    iters: int = 30,
) -> np.ndarray:
    """TWSL residual-based loyalty trust from multi-node target observations.

    Runs the TWSL self-consistent iteration once per spatial dimension (x, y, z)
    using each node's reported target coordinate (in metres) as the test
    cochain, and takes the per-node MINIMUM trust across dimensions — so a node
    spoofed in any one axis is caught.

    The TWSL Dirichlet residual (L_T x)_i measures how far node i's report is
    from the trust-weighted consensus of its neighbours. Because the Laplacian
    annihilates the constant component, only DIFFERENCES matter — the absolute
    target range is irrelevant. We mean-centre each axis purely for numerical
    conditioning (targets are at km-scale), and do NOT normalise by the spread:
    dividing by the per-axis std would amplify honest sensor noise into apparent
    disagreement and depress loyal-node trust.

    `sigma_m` is the residual scale in METRES. Choose it a few × the expected
    honest sensor noise (single-to-tens of metres) and well below a credible
    spoof offset (hundreds+ of metres): honest residuals ≪ σ → trust ≈ 1;
    spoof residuals ≫ σ → trust → floor.

    Loyal nodes agree with neighbours → low residual → trust ≈ 1.
    A node reporting a biased target → high residual → trust → low.
    """
    A = np.asarray(adjacency, dtype=np.float64)
    X = _cochain_matrix(reports_by_node)
    trust_per_dim = []
    for d in range(3):
        x = X[:, d]
        x_centered = x - np.mean(x)  # numerical conditioning only
        if np.allclose(x_centered, 0.0):
            trust_per_dim.append(np.ones(A.shape[0]))
            continue
        T = twsl_self_consistent_iteration(A, x_centered, sigma=sigma_m, iters=iters)
        trust_per_dim.append(T)
    return np.min(np.stack(trust_per_dim), axis=0)


def combined_trust(
    adjacency: np.ndarray,
    node_ids: list[str],
    reports_by_node: list[SensorReport | None],
    attestations: list[AttestationFlags],
    ew_states: list[EWState] | None = None,
    *,
    joined_attested: list[bool] | None = None,
    sigma_m: float = 50.0,
    kill_threshold: float = DEFAULT_KILL_THRESHOLD,
) -> TrustResult:
    """Full pipeline: graph residual × hardware attestation → loyalty + kill-switch.

    Parameters
    ----------
    adjacency : (N, N) comm-graph adjacency (symmetric, zero diagonal).
    node_ids : per-node identifiers.
    reports_by_node : each node's current localising SensorReport (or None).
    attestations : each node's hardware-attestation flags.
    ew_states : optional per-node EWState (defaults to nominal — full confidence).
    joined_attested : optional per-node flag — did the node pass attestation
        when it joined the fleet? A node that did NOT is capped at
        UNATTESTED_JOIN_FLOOR loyalty trust (Sybil-on-join defence).
    """
    n = len(node_ids)
    if not (len(reports_by_node) == len(attestations) == n):
        raise ValueError("node_ids, reports_by_node, attestations must align")

    graph_t = graph_trust_from_reports(adjacency, reports_by_node, sigma_m=sigma_m)
    att_mult = np.array([a.trust_multiplier for a in attestations], dtype=np.float64)
    loyalty = graph_t * att_mult

    # Attestation-on-join cap (Sybil defence on a churning fleet).
    if joined_attested is not None:
        for i, ok in enumerate(joined_attested):
            if not ok:
                loyalty[i] = min(loyalty[i], UNATTESTED_JOIN_FLOOR)

    if ew_states is None:
        confidence = np.ones(n, dtype=np.float64)
    else:
        confidence = np.array([e.confidence_factor for e in ew_states], dtype=np.float64)

    killed = loyalty < kill_threshold
    return TrustResult(
        node_ids=list(node_ids),
        graph_trust=graph_t,
        attestation_mult=att_mult,
        loyalty_trust=loyalty,
        confidence=confidence,
        killed=killed,
    )


def attestation_on_join(attestation: AttestationFlags) -> bool:
    """Gate a node joining the fleet — must pass core hardware attestation.

    A node that fails TPM or secure-boot, or trips the RF-tamper flag, is NOT
    admitted to full trust on join (capped at UNATTESTED_JOIN_FLOOR by
    combined_trust). IMU triple-redundancy can fail transiently in flight and
    is handled by the in-flight loyalty pipeline, so it is not a join gate.
    """
    return (
        attestation.tpm_attested
        and attestation.secure_boot_chain_valid
        and not attestation.rf_tamper_flag
    )
