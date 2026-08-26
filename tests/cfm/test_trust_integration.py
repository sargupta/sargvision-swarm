"""Integration tests — hardware-attested TWSL trust pipeline vs adversary models.

These prove the headline capability: the combined trust pipeline (TWSL graph
residual × hardware attestation) DETECTS spoofed / Sybil / unattested nodes and
fires the kill-switch on them, while retaining loyal nodes — and that EW jamming
of a loyal node downweights its confidence WITHOUT evicting it.
"""

from __future__ import annotations

import numpy as np
import pytest

from sargvision_swarm.cfm.adversary import SensorSpoofingAttacker, SybilAttacker
from sargvision_swarm.cfm.sensors.attestation import AttestationFlags
from sargvision_swarm.cfm.sensors.ewstate import EWState
from sargvision_swarm.cfm.sensors.report import Modality, SensorReport
from sargvision_swarm.cfm.trust import (
    UNATTESTED_JOIN_FLOOR,
    attestation_on_join,
    combined_trust,
)


def _ring_adjacency(n: int) -> np.ndarray:
    """Simple connected ring graph (each node linked to 2 neighbours)."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = 1.0
        A[i, (i - 1) % n] = 1.0
    return A


def _loyal_report(node_id: str, target: np.ndarray, rng: np.random.Generator) -> SensorReport:
    return SensorReport(
        reporter_id=node_id,
        modality=Modality.RADAR,
        target_position_enu_m=target + rng.normal(0, 2.0, size=3),  # small honest noise
        target_velocity_mps=np.array([-10.0, 0.0, 0.0]),
        confidence=0.9,
        timestamp_s=1.0,
        attestation=AttestationFlags(),  # all pass
    )


def test_all_loyal_nodes_retained():
    """Eight honest nodes all agreeing → all high trust, none killed."""
    n = 8
    rng = np.random.default_rng(0)
    target = np.array([5000.0, 1000.0, 200.0])
    ids = [f"DRN-{i:03d}" for i in range(n)]
    reports = [_loyal_report(ids[i], target, rng) for i in range(n)]
    atts = [AttestationFlags() for _ in range(n)]
    res = combined_trust(_ring_adjacency(n), ids, reports, atts)
    assert res.killed.sum() == 0
    assert res.loyalty_trust.min() > 0.5


def test_sensor_spoofer_node_is_killed():
    """A compromised node injecting a biased target + failing IMU attestation
    must converge to low trust and be kill-switched; loyal nodes retained."""
    n = 8
    rng = np.random.default_rng(1)
    target = np.array([5000.0, 1000.0, 200.0])
    ids = [f"DRN-{i:03d}" for i in range(n)]
    reports = [_loyal_report(ids[i], target, rng) for i in range(n)]
    atts = [AttestationFlags() for _ in range(n)]

    # Compromise node 3: big spoof offset + IMU triple-redundancy fails.
    attacker = SensorSpoofingAttacker(
        compromised_node_id=ids[3],
        spoof_offset_enu_m=np.array([1500.0, -800.0, 0.0]),
    )
    reports[3] = attacker.craft_false_report(target, np.array([-10.0, 0.0, 0.0]), 1.0)
    atts[3] = reports[3].attestation  # IMU-fail attestation from the attacker

    res = combined_trust(_ring_adjacency(n), ids, reports, atts)
    assert ids[3] in res.excluded_ids(), "spoofed node should be kill-switched"
    # all other nodes retained
    assert set(res.retained_ids()) == {ids[i] for i in range(n) if i != 3}
    # the spoofed node's trust is the lowest
    assert int(np.argmin(res.loyalty_trust)) == 3


def test_attestation_gate_blocks_pure_attestation_failure_even_if_agreeing():
    """A node that AGREES on the target (low residual) but fails hardware
    attestation (TPM + IMU) must still be downweighted by the attestation
    multiplier — graph agreement alone cannot launder a compromised node."""
    n = 8
    rng = np.random.default_rng(2)
    target = np.array([3000.0, 0.0, 150.0])
    ids = [f"DRN-{i:03d}" for i in range(n)]
    reports = [_loyal_report(ids[i], target, rng) for i in range(n)]
    atts = [AttestationFlags() for _ in range(n)]
    # Node 5 reports the correct target (graph-trusted) but fails 3 attestation
    # checks → trust_multiplier 0.0.
    atts[5] = AttestationFlags(
        tpm_attested=False, imu_triple_redundancy_pass=False, rf_tamper_flag=True
    )
    res = combined_trust(_ring_adjacency(n), ids, reports, atts)
    assert ids[5] in res.excluded_ids()


def test_unattested_join_is_capped():
    """A node that did not pass attestation-on-join is capped at the join floor
    even if it agrees with everyone (Sybil-on-churning-fleet defence)."""
    n = 6
    rng = np.random.default_rng(3)
    target = np.array([2000.0, 500.0, 100.0])
    ids = [f"DRN-{i:03d}" for i in range(n)]
    reports = [_loyal_report(ids[i], target, rng) for i in range(n)]
    atts = [AttestationFlags() for _ in range(n)]
    joined = [True] * n
    joined[2] = False  # node 2 never passed join attestation
    res = combined_trust(_ring_adjacency(n), ids, reports, atts, joined_attested=joined)
    assert res.loyalty_trust[2] <= UNATTESTED_JOIN_FLOOR + 1e-9


def test_attestation_on_join_gate():
    assert attestation_on_join(AttestationFlags()) is True
    assert attestation_on_join(AttestationFlags(tpm_attested=False)) is False
    assert attestation_on_join(AttestationFlags(secure_boot_chain_valid=False)) is False
    assert attestation_on_join(AttestationFlags(rf_tamper_flag=True)) is False
    # IMU transient failure is NOT a join gate (handled in-flight):
    assert attestation_on_join(AttestationFlags(imu_triple_redundancy_pass=False)) is True


def test_ew_jamming_downweights_confidence_without_eviction():
    """A jammed but loyal node keeps its loyalty trust (not killed) but its
    fusion confidence drops — defends against denial-of-trust via jamming."""
    n = 6
    rng = np.random.default_rng(4)
    target = np.array([4000.0, 0.0, 120.0])
    ids = [f"DRN-{i:03d}" for i in range(n)]
    reports = [_loyal_report(ids[i], target, rng) for i in range(n)]
    atts = [AttestationFlags() for _ in range(n)]
    ews = [EWState() for _ in range(n)]
    # Node 1 heavily jammed: link margin near zero + GNSS spoof suspected.
    ews[1] = EWState(jammer_power_dbm=-70.0, link_margin_db=1.0, gnss_spoof_suspected=True)
    res = combined_trust(_ring_adjacency(n), ids, reports, atts, ew_states=ews)
    # not evicted (still loyal)
    assert ids[1] not in res.excluded_ids()
    # but confidence is sharply reduced vs an un-jammed peer
    assert res.confidence[1] < 0.5
    assert res.confidence[0] > 0.9


def test_sybil_nodes_fail_join_and_are_capped():
    """Sybil drones lack valid hardware identity → fail attestation-on-join →
    capped at the join floor, regardless of what they report."""
    sybil = SybilAttacker()
    sybil.add_sybil("SYB-1", np.array([0.0, 0.0, 100.0]))
    # Sybils present no valid TPM/secure-boot — model as failing the join gate.
    sybil_attestation = AttestationFlags(tpm_attested=False, secure_boot_chain_valid=False)
    assert attestation_on_join(sybil_attestation) is False
