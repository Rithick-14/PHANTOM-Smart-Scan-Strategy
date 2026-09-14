"""
Unit tests for Detection Model and ES Receiver Model.
"""

import pytest

from smart_ew.environment.schema import SimConfig, EmitterConfig
from smart_ew.environment.simulator import RFEnvironment
from smart_ew.emitters.continuous import ContinuousEmitter
from smart_ew.detection.model import DetectionModel
from smart_ew.receiver.schema import ReceiverConfig
from smart_ew.receiver.model import ESReceiver


def test_detection_model_deterministic():
    det_model = DetectionModel(
        snr_threshold_db=10.0, p_false_alarm=0.0, noise_std_db=0.0, probabilistic=False
    )

    # Signal 15 dB above noise -> SNR = 15 dB >= 10 dB -> detected
    res1 = det_model.evaluate(true_power_dbm=-85.0, noise_floor_dbm=-100.0, is_truly_active=True)
    assert res1.detected
    assert pytest.approx(res1.estimated_snr_db, abs=0.01) == 15.0
    assert not res1.is_false_alarm

    # Signal 5 dB above noise -> SNR = 5 dB < 10 dB -> missed
    res2 = det_model.evaluate(true_power_dbm=-95.0, noise_floor_dbm=-100.0, is_truly_active=True)
    assert not res2.detected


def test_detection_model_probabilistic_curve():
    det_model = DetectionModel(snr_threshold_db=10.0, probabilistic=True)

    # Low SNR (2 dB) -> low Pd
    pd_low = det_model.compute_pd(2.0)
    # High SNR (20 dB) -> high Pd
    pd_high = det_model.compute_pd(20.0)

    assert pd_low < 0.1
    assert pd_high > 0.95


def test_receiver_retuning_cost():
    rx_config = ReceiverConfig(retuning_time_cost=1.5)
    receiver = ESReceiver(config=rx_config)

    sim_config = SimConfig(num_bands=3)
    env = RFEnvironment(sim_config)
    frame0 = env.step()

    # Initial scan on band 0 (no retuning cost paid yet since it's first scan)
    obs1 = receiver.scan(band_id=0, time_step=0, ground_truth_frame=frame0)
    assert obs1.tuning_cost_paid == 0.0

    # Scan band 0 again -> no retuning cost
    obs2 = receiver.scan(band_id=0, time_step=1, ground_truth_frame=frame0)
    assert obs2.tuning_cost_paid == 0.0

    # Switch to band 1 -> retuning cost paid = 1.5
    obs3 = receiver.scan(band_id=1, time_step=2, ground_truth_frame=frame0)
    assert obs3.tuning_cost_paid == 1.5


def test_receiver_integration_with_environment():
    env = RFEnvironment(SimConfig(num_bands=5))
    emitter = ContinuousEmitter(
        EmitterConfig(
            emitter_id="TARGET_1",
            emitter_type="continuous",
            assigned_bands=[2],
            signal_power_dbm=-70.0,  # 30 dB SNR over -100 dBm noise floor
        )
    )
    env.add_emitter(emitter)

    receiver = ESReceiver(
        ReceiverConfig(snr_threshold_db=10.0, noise_std_db=0.0, p_false_alarm=0.0), seed=42
    )

    frame = env.step()

    # Scan active band 2 -> should detect signal
    obs_active = receiver.scan(band_id=2, time_step=0, ground_truth_frame=frame)
    assert obs_active.detected
    assert obs_active.band_id == 2
    assert obs_active.estimated_snr_db > 25.0

    # Scan quiet band 0 -> should not detect signal
    obs_quiet = receiver.scan(band_id=0, time_step=0, ground_truth_frame=frame)
    assert not obs_quiet.detected
    assert obs_quiet.band_id == 0
