"""
Unit tests for Page-Hinkley Change Detector and Change-Adaptive Scheduler.
"""

import pytest

from smart_ew.environment.schema import SimConfig, EmitterConfig
from smart_ew.environment.simulator import RFEnvironment
from smart_ew.emitters.continuous import ContinuousEmitter
from smart_ew.receiver.schema import ReceiverConfig
from smart_ew.receiver.model import ESReceiver
from smart_ew.knowledge.change_detector import PageHinkleyChangeDetector
from smart_ew.scheduling.change_adaptive import ChangeAdaptiveScheduler
from smart_ew.evaluation.runner import SimulationRunner


def test_page_hinkley_change_detector_drift():
    detector = PageHinkleyChangeDetector(threshold=2.0)

    # Low error phase
    for _ in range(20):
        is_change = detector.update(target=1, predicted_prob=0.9)
        assert not is_change

    # Sudden high error phase (environment shift)
    change_flagged = False
    for _ in range(15):
        if detector.update(target=0, predicted_prob=0.9):
            change_flagged = True
            break

    assert change_flagged
    assert detector.change_count == 1


def test_change_adaptive_scheduler_mid_simulation_shift():
    config = SimConfig(num_bands=5, total_time_steps=100)
    env = RFEnvironment(config)

    # Emitter 1: active on Band 0 from step 0 to 45
    e1 = ContinuousEmitter(
        EmitterConfig(
            emitter_id="E1",
            emitter_type="continuous",
            assigned_bands=[0],
            signal_power_dbm=-50.0,
            parameters={"start_time": 0, "end_time": 45},
        )
    )
    # Emitter 2: active on Band 4 from step 50 to 100
    e2 = ContinuousEmitter(
        EmitterConfig(
            emitter_id="E2",
            emitter_type="continuous",
            assigned_bands=[4],
            signal_power_dbm=-50.0,
            parameters={"start_time": 50, "end_time": 100},
        )
    )

    env.add_emitter(e1)
    env.add_emitter(e2)

    rx = ESReceiver(ReceiverConfig(snr_threshold_db=10.0), seed=42)
    scheduler = ChangeAdaptiveScheduler(num_bands=5, change_threshold=2.0, seed=42)
    runner = SimulationRunner()

    metrics = runner.run(env, rx, scheduler, total_steps=100)

    assert metrics.total_scans == 100
    # Scheduler should intercept both initial emitter E1 and post-shift emitter E2
    scanned_b4_post_shift = sum(
        1 for obs in metrics.observations[50:] if obs.band_id == 4 and obs.detected
    )
    assert scanned_b4_post_shift > 0
