"""
Unit tests for Uncertainty-Aware Exploration Scheduler.
"""

import pytest

from smart_ew.environment.schema import SimConfig, EmitterConfig
from smart_ew.environment.simulator import RFEnvironment
from smart_ew.emitters.intermittent import IntermittentEmitter
from smart_ew.receiver.schema import ReceiverConfig
from smart_ew.receiver.model import ESReceiver
from smart_ew.scheduling.sequential import SequentialScheduler
from smart_ew.scheduling.ml_greedy import MLGreedyScheduler
from smart_ew.scheduling.uncertainty_aware import UncertaintyAwareScheduler
from smart_ew.evaluation.runner import SimulationRunner


def test_uncertainty_aware_scheduler_initialization():
    scheduler = UncertaintyAwareScheduler(
        num_bands=5, alpha_linucb=0.5, beta_uncertainty=0.5, gamma_recency=0.2, seed=42
    )

    assert scheduler.num_bands == 5
    action = scheduler.select_next_band(time_step=0)
    assert 0 <= action < 5


def test_uncertainty_aware_simulation_benchmark():
    config = SimConfig(num_bands=6, total_time_steps=90)
    env = RFEnvironment(config)

    # Intermittent emitter on band 2
    env.add_emitter(
        IntermittentEmitter(
            EmitterConfig(
                emitter_id="INT_B2",
                emitter_type="intermittent",
                assigned_bands=[2],
                signal_power_dbm=-50.0,
                parameters={"p_on": 0.3, "p_off": 0.4, "initial_active": True},
            ),
            seed=42,
        )
    )

    rx = ESReceiver(ReceiverConfig(snr_threshold_db=10.0), seed=42)
    runner = SimulationRunner()

    seq_sched = SequentialScheduler(num_bands=6)
    ml_greedy_sched = MLGreedyScheduler(num_bands=6, epsilon=0.05, seed=42)
    unc_sched = UncertaintyAwareScheduler(
        num_bands=6, alpha_linucb=0.5, beta_uncertainty=0.8, gamma_recency=0.3, seed=42
    )

    metrics_seq = runner.run(env, rx, seq_sched, total_steps=90)
    metrics_ml = runner.run(env, rx, ml_greedy_sched, total_steps=90)
    metrics_unc = runner.run(env, rx, unc_sched, total_steps=90)

    assert metrics_seq.total_scans == 90
    assert metrics_ml.total_scans == 90
    assert metrics_unc.total_scans == 90

    # Uncertainty-aware scheduler should achieve 100% spectrum coverage
    assert metrics_unc.spectrum_coverage_ratio == 1.0
    # Uncertainty-aware scheduler should intercept active transmissions
    assert metrics_unc.total_intercepts > 0
