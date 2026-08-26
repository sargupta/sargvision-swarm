"""Tests for the adversarial robustness layer."""

from __future__ import annotations

import numpy as np
import pytest

from sargvision_swarm.cfm.adversary import (
    FriisJammer,
    GNSSSpoofer,
    SensorSpoofingAttacker,
    SybilAttacker,
    friis_path_loss_db,
)
from sargvision_swarm.cfm.sensors.report import Modality


# ── Friis jammer tests ─────────────────────────────────────────────────────


def test_friis_path_loss_at_1km_5p8ghz_matches_published_value():
    """Free-space path loss at 1 km, 5.8 GHz should be ≈ 107.7 dB.

    Cross-checked against file 49 (RF link budgets):
       FSPL(1km, 5.8 GHz) = 20·log10(1000) + 20·log10(5.8e9) − 147.55
                          ≈ 60 + 195.27 − 147.55
                          ≈ 107.72 dB
    """
    loss = friis_path_loss_db(1000.0, 5.8e9)
    assert 107.0 <= loss <= 108.5


def test_friis_path_loss_doubles_distance_adds_6db():
    """Doubling distance adds exactly 6 dB to FSPL."""
    a = friis_path_loss_db(100.0, 2.4e9)
    b = friis_path_loss_db(200.0, 2.4e9)
    assert b == pytest.approx(a + 6.0, abs=0.05)


def test_jammer_denies_link_at_close_range():
    """1W jammer at 100m should deny a typical drone link."""
    jammer = FriisJammer(
        position_enu_m=np.array([0.0, 0.0, 50.0]),
        tx_power_dbm=30.0,  # 1 W
        antenna_gain_dbi=3.0,
        frequency_hz=2.4e9,
    )
    victim = np.array([100.0, 0.0, 50.0])
    assert jammer.is_link_denied(victim)


def test_jammer_does_not_deny_link_at_far_range():
    """1W jammer at 50km should NOT deny a typical drone link."""
    jammer = FriisJammer(
        position_enu_m=np.array([0.0, 0.0, 50.0]),
        tx_power_dbm=30.0,
        antenna_gain_dbi=3.0,
        frequency_hz=2.4e9,
    )
    victim = np.array([50_000.0, 0.0, 50.0])
    assert not jammer.is_link_denied(victim)


def test_jammer_deny_radius_monotone_in_power():
    """Higher jammer power → larger deny radius."""
    low = FriisJammer(
        position_enu_m=np.zeros(3),
        tx_power_dbm=20.0,
        frequency_hz=2.4e9,
    )
    high = FriisJammer(
        position_enu_m=np.zeros(3),
        tx_power_dbm=40.0,
        frequency_hz=2.4e9,
    )
    assert high.deny_radius_m() > low.deny_radius_m()


def test_jammer_higher_frequency_smaller_deny_radius():
    """Higher frequency suffers more path loss → smaller deny radius."""
    low_freq = FriisJammer(
        position_enu_m=np.zeros(3),
        tx_power_dbm=30.0,
        frequency_hz=900e6,
    )
    high_freq = FriisJammer(
        position_enu_m=np.zeros(3),
        tx_power_dbm=30.0,
        frequency_hz=5.8e9,
    )
    assert low_freq.deny_radius_m() > high_freq.deny_radius_m()


# ── GNSS spoofer tests ─────────────────────────────────────────────────────


def test_gnss_spoofer_drifts_position_over_time():
    spoofer = GNSSSpoofer(
        drift_velocity_mps=np.array([1.0, 0.0, 0.0]),
        spoof_radius_m=10_000.0,
        centre_enu_m=np.zeros(3),
    )
    true_pos = np.array([100.0, 0.0, 50.0])
    after_60s = spoofer.apply(true_pos, elapsed_s=60.0)
    # 1 m/s drift × 60 s = 60 m offset
    assert after_60s[0] == pytest.approx(true_pos[0] + 60.0)


def test_gnss_spoofer_out_of_range_no_effect():
    spoofer = GNSSSpoofer(
        drift_velocity_mps=np.array([1.0, 0.0, 0.0]),
        spoof_radius_m=1000.0,
        centre_enu_m=np.zeros(3),
    )
    far_victim = np.array([10_000.0, 0.0, 0.0])
    after = spoofer.apply(far_victim, elapsed_s=60.0)
    assert np.allclose(after, far_victim)


def test_gnss_spoofer_disabled_no_effect():
    spoofer = GNSSSpoofer(drift_velocity_mps=np.array([1.0, 0.0, 0.0]))
    spoofer.disable()
    victim = np.array([0.0, 0.0, 0.0])
    after = spoofer.apply(victim, elapsed_s=60.0)
    assert np.allclose(after, victim)


# ── Sensor-spoofing attacker tests ─────────────────────────────────────────


def test_sensor_spoofer_returns_biased_report():
    attacker = SensorSpoofingAttacker(
        compromised_node_id="drone-007",
        spoof_offset_enu_m=np.array([500.0, 0.0, 0.0]),
    )
    true_pos = np.array([1000.0, 1000.0, 100.0])
    rep = attacker.craft_false_report(true_pos, None, timestamp_s=1.0)
    assert rep.target_position_enu_m is not None
    assert rep.target_position_enu_m[0] == pytest.approx(1500.0)


def test_sensor_spoofer_fails_imu_attestation():
    attacker = SensorSpoofingAttacker(
        compromised_node_id="drone-007",
        spoof_offset_enu_m=np.array([100.0, 0.0, 0.0]),
    )
    rep = attacker.craft_false_report(np.zeros(3), None, timestamp_s=0.0)
    # Spoofed reports must NOT pass attestation (this is the defence signature)
    assert not rep.attestation.all_pass
    assert not rep.attestation.imu_triple_redundancy_pass


def test_sensor_spoofer_custom_modality():
    attacker = SensorSpoofingAttacker(
        compromised_node_id="drone-007",
        spoof_offset_enu_m=np.array([10.0, 0.0, 0.0]),
        spoof_modalities=(Modality.ELECTRO_OPTICAL,),
    )
    rep = attacker.craft_false_report(np.zeros(3), None, timestamp_s=0.0)
    assert rep.modality == Modality.ELECTRO_OPTICAL


# ── Sybil attacker tests ───────────────────────────────────────────────────


def test_sybil_attacker_adds_and_advances():
    sybil = SybilAttacker()
    sybil.add_sybil(
        sybil_id="sybil-1",
        position_enu_m=np.array([0.0, 0.0, 100.0]),
        velocity_mps=np.array([10.0, 0.0, 0.0]),
    )
    assert sybil.count() == 1
    sybil.step(dt_s=2.0)
    assert sybil.sybil_drones[0].position_enu_m[0] == pytest.approx(20.0)
