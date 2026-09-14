"""
Unit tests for LinUCB and Contextual Thompson Sampling Schedulers.
"""

import pytest
import numpy as np

from smart_ew.environment.schema import SimConfig, EmitterConfig
from smart_ew.environment.simulator import RFEnvironment
from smart_ew.emitters.intermittent import IntermittentEmitter
from smart_ew.receiver.schema import ReceiverConfig
from smart_ew.receiver.model import ESReceiver
from smart_ew.scheduling.sequential import SequentialScheduler
from smart_ew.scheduling.random_scan import RandomScheduler
from smart_ew.scheduling.ml_greedy import MLGreedyScheduler
from smart_ew.scheduling.contextual_bandit import (
    LinUCBScheduler,
    ContextualThompsonSamplingScheduler,
)
from smart_ew.evaluation.runner import SimulationRunner


def test_linucb_initialization_and_update():
    scheduler = LinUCBScheduler(num_bands=4, alpha=1.0, feature_dim=10, seed=42)

    assert len(scheduler.A) == 4
    assert scheduler.A[0].shape == (10, 10)
    assert scheduler.b_vec[0].shape == (10,)

    action = scheduler.select_next_band(time_step=0)
    assert 0 <= action < 4

    obs = scheduler.feature_extractor.extract_features(action, 0, 4)
    # Simulate update
    scheduler.update(
        type("Obs", (), {
            "band_id": action,
            "timestamp": 0,
            "detected": True,
            "is_false_alarm": False,
            "tuning_cost_paid": 0.0,
        })()
    )

    # A matrix for selected action should no longer be identity
    assert not np.array_equal(scheduler.A[action], np.eye(10))


def test_thompson_sampling_initialization():
    ts_scheduler = ContextualThompsonSamplingScheduler(num_bands=3, seed=123)
    action = ts_scheduler.select_next_band(time_step=0)
    assert 0 <= action < 3


def test_contextual_bandit_simulation_benchmark():
    config = SimConfig(num_bands=5, total_time_steps=80)
    env = RFEnvironment(config)

    # Add intermittent emitter on band 1
    env.add_emitter(
        IntermittentEmitter(
            EmitterConfig(
                emitter_id="INT_B1",
                emitter_type="intermittent",
                assigned_bands=[1],
                signal_power_dbm=-50.0,
                parameters={"p_on": 0.4, "p_off": 0.3, "initial_active": True},
            ),
            seed=42,
        )
    )

    rx = ESReceiver(ReceiverConfig(snr_threshold_db=10.0), seed=42)
    runner = SimulationRunner()

    seq_sched = SequentialScheduler(num_bands=5)
    rand_sched = RandomScheduler(num_bands=5, seed=42)
    linucb_sched = LinUCBScheduler(num_bands=5, alpha=0.5, seed=42)

    metrics_seq = runner.run(env, rx, seq_sched, total_steps=80)
    metrics_rand = runner.run(env, rx, rand_sched, total_steps=80)
    metrics_linucb = runner.run(env, rx, linucb_sched, total_steps=80)

    assert metrics_seq.total_scans == 80
    assert metrics_rand.total_scans == 80
    assert metrics_linucb.total_scans == 80

    # LinUCB should intercept a significant number of active transmissions
    assert metrics_linucb.total_intercepts > 0
