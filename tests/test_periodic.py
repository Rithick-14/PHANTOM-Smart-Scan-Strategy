"""
Unit tests for Periodic Emitter Model and Periodicity Estimator.
"""

import pytest

from smart_ew.environment.schema import SimConfig, EmitterConfig
from smart_ew.environment.simulator import RFEnvironment
from smart_ew.emitters.periodic import PeriodicEmitter
from smart_ew.knowledge.periodicity import PeriodicityEstimator
from smart_ew.receiver.schema import ReceiverConfig
from smart_ew.receiver.model import ESReceiver
from smart_ew.scheduling.sequential import SequentialScheduler
from smart_ew.scheduling.uncertainty_aware import UncertaintyAwareScheduler
from smart_ew.evaluation.runner import SimulationRunner


def test_periodic_emitter_activation():
    config = EmitterConfig(
        emitter_id="P_RADAR_1",
        emitter_type="periodic",
        assigned_bands=[1],
        signal_power_dbm=-50.0,
        parameters={"period": 10, "duty_duration": 2, "phase_offset": 0},
    )
    emitter = PeriodicEmitter(config)

    # Active at steps 0 and 1
    assert emitter.is_active(0, 1)
    assert emitter.is_active(1, 1)
    # Quiet at steps 2 to 9
    assert not emitter.is_active(2, 1)
    assert not emitter.is_active(9, 1)
    # Active again at step 10
    assert emitter.is_active(10, 1)


def test_periodicity_estimator():
    estimator = PeriodicityEstimator()

    timestamps = [10, 20, 30, 40, 50]
    est_p = estimator.estimate_period(timestamps)
    assert est_p == 10

    # Predict at step 60 (exactly 1 cycle after 50) -> high probability
    prob_60 = estimator.predict_activity_window(timestamps, target_step=60)
    assert prob_60 > 0.9

    # Predict at step 65 (half cycle) -> low probability
    prob_65 = estimator.predict_activity_window(timestamps, target_step=65)
    assert prob_65 < 0.2


def test_periodic_emitter_simulation_benchmark():
    config = SimConfig(num_bands=5, total_time_steps=100)
    env = RFEnvironment(config)

    # Periodic emitter on Band 2 (Period = 8, Duration = 2)
    env.add_emitter(
        PeriodicEmitter(
            EmitterConfig(
                emitter_id="RADAR_B2",
                emitter_type="periodic",
                assigned_bands=[2],
                signal_power_dbm=-50.0,
                parameters={"period": 8, "duty_duration": 2},
            )
        )
    )

    rx = ESReceiver(ReceiverConfig(snr_threshold_db=10.0), seed=42)
    runner = SimulationRunner()

    seq_sched = SequentialScheduler(num_bands=5)
    unc_sched = UncertaintyAwareScheduler(num_bands=5, seed=42)

    metrics_seq = runner.run(env, rx, seq_sched, total_steps=100)
    metrics_unc = runner.run(env, rx, unc_sched, total_steps=100)

    assert metrics_seq.total_scans == 100
    assert metrics_unc.total_scans == 100
    assert metrics_unc.total_intercepts > 0
