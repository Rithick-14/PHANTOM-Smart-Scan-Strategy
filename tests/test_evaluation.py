"""
Unit tests for Evaluation Engine and SimulationRunner metrics calculation.
"""

import pytest

from smart_ew.environment.schema import SimConfig, EmitterConfig
from smart_ew.environment.simulator import RFEnvironment
from smart_ew.emitters.continuous import ContinuousEmitter
from smart_ew.emitters.intermittent import IntermittentEmitter
from smart_ew.receiver.schema import ReceiverConfig
from smart_ew.receiver.model import ESReceiver
from smart_ew.scheduling.sequential import SequentialScheduler
from smart_ew.scheduling.random_scan import RandomScheduler
from smart_ew.evaluation.runner import SimulationRunner


def test_simulation_runner_basic_execution():
    config = SimConfig(num_bands=5, total_time_steps=50)
    env = RFEnvironment(config)

    # Add continuous emitter on band 1
    emitter = ContinuousEmitter(
        EmitterConfig(
            emitter_id="C1",
            emitter_type="continuous",
            assigned_bands=[1],
            signal_power_dbm=-60.0,
        )
    )
    env.add_emitter(emitter)

    rx = ESReceiver(ReceiverConfig(snr_threshold_db=10.0, p_false_alarm=0.0), seed=42)
    scheduler = SequentialScheduler(num_bands=5)
    runner = SimulationRunner()

    metrics = runner.run(env, rx, scheduler, total_steps=50)

    assert metrics.total_steps == 50
    assert metrics.num_bands == 5
    assert metrics.total_scans == 50
    # In 50 steps, band 1 is scanned 10 times by sequential scheduler
    assert metrics.total_active_scans == 10
    assert metrics.total_intercepts == 10
    assert pytest.approx(metrics.probability_of_detection, abs=0.01) == 1.0
    assert metrics.spectrum_coverage_ratio == 1.0
    summary = metrics.summary_dict()
    assert summary["pd"] == 1.0
    assert summary["num_bands"] == 5


def test_scheduler_comparison_benchmark():
    config = SimConfig(num_bands=10, total_time_steps=100)
    env = RFEnvironment(config)

    # Intermittent emitter on band 3
    env.add_emitter(
        IntermittentEmitter(
            EmitterConfig(
                emitter_id="I3",
                emitter_type="intermittent",
                assigned_bands=[3],
                signal_power_dbm=-50.0,
                parameters={"p_on": 0.5, "p_off": 0.5, "initial_active": True},
            ),
            seed=42,
        )
    )

    rx = ESReceiver(ReceiverConfig(retuning_time_cost=1.0), seed=42)
    runner = SimulationRunner()

    seq_scheduler = SequentialScheduler(num_bands=10)
    rand_scheduler = RandomScheduler(num_bands=10, seed=123)

    metrics_seq = runner.run(env, rx, seq_scheduler, total_steps=100)
    metrics_rand = runner.run(env, rx, rand_scheduler, total_steps=100)

    assert metrics_seq.total_scans == 100
    assert metrics_rand.total_scans == 100
    # Sequential retuning cost for 100 steps: switches band 99 times -> cost = 99.0
    assert metrics_seq.total_retuning_cost == 99.0
