"""
Unit tests for Machine Learning Activity Predictors and ML-Greedy Scheduler.
"""

import pytest
import numpy as np

from smart_ew.environment.schema import SimConfig, EmitterConfig
from smart_ew.environment.simulator import RFEnvironment
from smart_ew.emitters.continuous import ContinuousEmitter
from smart_ew.receiver.schema import ReceiverConfig
from smart_ew.receiver.model import ESReceiver
from smart_ew.models.predictor import (
    LogisticRegressionPredictor,
    RandomForestPredictor,
    GradientBoostingPredictor,
    MLActivityPredictor,
)
from smart_ew.scheduling.sequential import SequentialScheduler
from smart_ew.scheduling.ml_greedy import MLGreedyScheduler
from smart_ew.evaluation.runner import SimulationRunner


def test_predictors_fit_and_predict():
    X = np.array(
        [
            [0.1, 5, 0, 5, 1.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 10, 8, 2, 1.0, 1.0, 0.8, 0.8, 0.8, 0.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 1], dtype=int)

    for predictor_cls in [
        LogisticRegressionPredictor,
        RandomForestPredictor,
        GradientBoostingPredictor,
    ]:
        predictor = predictor_cls(seed=42)
        assert not predictor.is_fitted

        predictor.fit(X, y)
        assert predictor.is_fitted

        probas = predictor.predict_proba(X)
        assert probas.shape == (2,)
        assert 0.0 <= probas[0] <= 1.0
        assert 0.0 <= probas[1] <= 1.0


def test_ml_activity_predictor_buffering():
    ml_mgr = MLActivityPredictor(fit_interval=2, min_samples=4, seed=42)

    sample_x = np.ones(10, dtype=np.float32)
    ml_mgr.add_sample(sample_x, 0)
    ml_mgr.add_sample(sample_x, 1)
    ml_mgr.add_sample(sample_x, 0)

    # Below min_samples=4, should not be fitted yet
    assert not ml_mgr.predictor.is_fitted

    ml_mgr.add_sample(sample_x, 1)
    # Reached 4 samples (multiple of fit_interval=2) -> should fit now
    assert ml_mgr.predictor.is_fitted


def test_ml_greedy_scheduler_benchmark():
    config = SimConfig(num_bands=5, total_time_steps=60)
    env = RFEnvironment(config)

    # Continuous active emitter on band 2
    env.add_emitter(
        ContinuousEmitter(
            EmitterConfig(
                emitter_id="ACTIVE_B2",
                emitter_type="continuous",
                assigned_bands=[2],
                signal_power_dbm=-50.0,
            )
        )
    )

    rx = ESReceiver(ReceiverConfig(snr_threshold_db=10.0), seed=42)
    runner = SimulationRunner()

    seq_scheduler = SequentialScheduler(num_bands=5)
    ml_scheduler = MLGreedyScheduler(num_bands=5, epsilon=0.1, seed=42)

    metrics_seq = runner.run(env, rx, seq_scheduler, total_steps=60)
    metrics_ml = runner.run(env, rx, ml_scheduler, total_steps=60)

    assert metrics_seq.total_scans == 60
    assert metrics_ml.total_scans == 60
    # ML Greedy should learn that band 2 is active and scan it more frequently than sequential sweep
    scanned_b2_ml = sum(1 for obs in metrics_ml.observations if obs.band_id == 2)
    scanned_b2_seq = sum(1 for obs in metrics_seq.observations if obs.band_id == 2)

    assert scanned_b2_ml > scanned_b2_seq
