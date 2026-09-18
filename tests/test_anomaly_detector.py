"""
Unit tests for RFAnomalyDetector.
"""

import pytest
from smart_ew.knowledge.anomaly_detector import RFAnomalyDetector, AnomalyRecord
from smart_ew.receiver.schema import ReceiverObservation


def test_anomaly_detector_initialization():
    detector = RFAnomalyDetector(num_bands=10)
    assert detector.num_bands == 10
    summary = detector.get_anomaly_summary()
    assert summary["total_evaluations"] == 0
    assert summary["total_anomalies"] == 0


def test_anomaly_detector_evaluates_normal_and_anomalous_events():
    detector = RFAnomalyDetector(num_bands=5, quiet_burst_threshold=5)

    # 1. Quiet scan on band 0 -> Normal
    obs1 = ReceiverObservation(
        band_id=0, timestamp=0, detected=False, measured_power_dbm=-100.0,
        estimated_snr_db=0.0, confidence=0.9, tuning_cost_paid=0.0, is_false_alarm=False
    )
    rec1 = detector.evaluate_observation(obs1, predicted_prob=0.1)
    assert not rec1.is_anomaly
    assert rec1.anomaly_category == "NORMAL"

    # 2. Unexpected burst on quiet band 3 -> Unexpected burst anomaly
    obs2 = ReceiverObservation(
        band_id=3, timestamp=10, detected=True, measured_power_dbm=-85.0,
        estimated_snr_db=15.0, confidence=0.8, tuning_cost_paid=1.0, is_false_alarm=False
    )
    rec2 = detector.evaluate_observation(obs2, predicted_prob=0.1)
    assert rec2.is_anomaly
    assert rec2.anomaly_category == "UNEXPECTED_BURST"

    # 3. High power SNR spike -> Power spike anomaly
    obs3 = ReceiverObservation(
        band_id=1, timestamp=11, detected=True, measured_power_dbm=-65.0,
        estimated_snr_db=35.0, confidence=0.95, tuning_cost_paid=0.0, is_false_alarm=False
    )
    rec3 = detector.evaluate_observation(obs3, predicted_prob=0.8)
    assert rec3.is_anomaly
    assert rec3.anomaly_category == "POWER_SPIKE"

    summary = detector.get_anomaly_summary()
    assert summary["total_evaluations"] == 3
    assert summary["total_anomalies"] == 2


def test_anomaly_detector_reset():
    detector = RFAnomalyDetector(num_bands=5)
    obs = ReceiverObservation(
        band_id=1, timestamp=1, detected=True, measured_power_dbm=-85.0,
        estimated_snr_db=15.0, confidence=0.8, tuning_cost_paid=0.0, is_false_alarm=False
    )
    detector.evaluate_observation(obs, predicted_prob=0.05)
    assert len(detector.anomaly_records) == 1

    detector.reset()
    assert len(detector.anomaly_records) == 0
    assert detector.get_anomaly_summary()["total_evaluations"] == 0
