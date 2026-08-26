"""Tests for sensor attestation flags + multi-modal fusion."""

from __future__ import annotations

import numpy as np
import pytest

from sargvision_swarm.cfm.sensors.attestation import (
    AttestationFlags,
    generate_attestation,
    imu_triple_redundancy_check,
)
from sargvision_swarm.cfm.sensors.fusion import (
    cross_modal_consistency,
    fuse_reports,
)
from sargvision_swarm.cfm.sensors.report import Modality, SensorReport


def test_attestation_all_pass_multiplier_is_one():
    flags = AttestationFlags()
    assert flags.all_pass
    assert flags.trust_multiplier == 1.0


def test_attestation_one_fail_halves_multiplier():
    flags = AttestationFlags(tpm_attested=False)
    assert not flags.all_pass
    assert flags.trust_multiplier == 0.5


def test_attestation_two_failures_heavy_downweight():
    flags = AttestationFlags(tpm_attested=False, imu_triple_redundancy_pass=False)
    assert flags.trust_multiplier == 0.1


def test_attestation_three_failures_rejects():
    flags = AttestationFlags(
        tpm_attested=False,
        imu_triple_redundancy_pass=False,
        rf_tamper_flag=True,
    )
    assert flags.trust_multiplier == 0.0


def test_generate_attestation_default_all_pass():
    flags = generate_attestation(np.random.default_rng(0))
    assert flags.all_pass


def test_imu_triple_redundancy_pass_within_tolerance():
    a = np.array([1.0, 0.0, 9.8])
    b = np.array([1.01, 0.0, 9.81])
    c = np.array([1.005, -0.005, 9.795])
    assert imu_triple_redundancy_check(a, b, c, tolerance=0.05)


def test_imu_triple_redundancy_fail_outside_tolerance():
    a = np.array([1.0, 0.0, 9.8])
    b = np.array([1.5, 0.0, 9.8])  # outlier
    c = np.array([1.005, 0.0, 9.8])
    assert not imu_triple_redundancy_check(a, b, c, tolerance=0.05)


def test_imu_triple_redundancy_shape_mismatch():
    with pytest.raises(ValueError):
        imu_triple_redundancy_check(np.zeros(3), np.zeros(2), np.zeros(3))


# ── Cross-modal fusion tests ────────────────────────────────────────────────


def _report(modality: Modality, pos: np.ndarray, conf: float = 0.8) -> SensorReport:
    return SensorReport(
        reporter_id=f"drone-{modality.value}",
        modality=modality,
        target_position_enu_m=pos,
        target_velocity_mps=np.array([-10.0, 0.0, 0.0]),
        confidence=conf,
        timestamp_s=1_700_000_000.0,
        attestation=AttestationFlags(),
    )


def test_cross_modal_consistency_full_agreement():
    pos = np.array([1000.0, 0.0, 100.0])
    reports = [
        _report(Modality.RADAR, pos),
        _report(Modality.RF_DIRECTION_FINDING, pos + np.array([5.0, 5.0, 0.0])),
        _report(Modality.ELECTRO_OPTICAL, pos + np.array([10.0, 0.0, 0.0])),
    ]
    score, agreeing = cross_modal_consistency(reports, position_tolerance_m=30.0)
    assert score == pytest.approx(1.0)
    assert len(agreeing) == 3


def test_cross_modal_consistency_one_outlier():
    pos = np.array([1000.0, 0.0, 100.0])
    reports = [
        _report(Modality.RADAR, pos),
        _report(Modality.RF_DIRECTION_FINDING, pos + np.array([5.0, 5.0, 0.0])),
        _report(Modality.ELECTRO_OPTICAL, np.array([1500.0, 200.0, 100.0])),  # outlier
    ]
    score, agreeing = cross_modal_consistency(reports, position_tolerance_m=30.0)
    assert score == pytest.approx(2 / 3, rel=1e-3)
    assert len(agreeing) == 2


def test_fuse_reports_produces_weighted_mean():
    pos = np.array([1000.0, 0.0, 100.0])
    reports = [
        _report(Modality.RADAR, pos, conf=1.0),
        _report(Modality.ELECTRO_OPTICAL, pos + np.array([20.0, 0.0, 0.0]), conf=0.5),
    ]
    fused = fuse_reports(reports, position_tolerance_m=30.0)
    assert fused is not None
    # higher-confidence radar pulls fused position toward pos
    assert fused.target_position_enu_m[0] < pos[0] + 10.0


def test_fuse_reports_returns_none_when_no_agreement():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([10_000.0, 0.0, 0.0])  # very far
    reports = [
        _report(Modality.RADAR, p1),
        _report(Modality.ELECTRO_OPTICAL, p2),
    ]
    fused = fuse_reports(reports, position_tolerance_m=30.0)
    assert fused is None
