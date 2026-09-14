"""
Unit tests for Frequency-Agile Emitter and Transition Predictor.
"""

import pytest
import numpy as np

from smart_ew.environment.schema import SimConfig, EmitterConfig
from smart_ew.environment.simulator import RFEnvironment
from smart_ew.emitters.agile import FrequencyAgileEmitter
from smart_ew.knowledge.map import DynamicRFKnowledgeMap
from smart_ew.knowledge.transition_predictor import TransitionPredictor
from smart_ew.receiver.schema import ReceiverConfig
from smart_ew.receiver.model import ESReceiver
from smart_ew.scheduling.uncertainty_aware import UncertaintyAwareScheduler
from smart_ew.evaluation.runner import SimulationRunner


def test_frequency_agile_emitter_sequence():
    config = EmitterConfig(
        emitter_id="HOPPER_1",
        emitter_type="agile",
        assigned_bands=[0, 1, 2],
        signal_power_dbm=-50.0,
        parameters={"dwell_time": 2, "hop_sequence": [0, 2, 1]},
    )
    emitter = FrequencyAgileEmitter(config, seed=42)

    # Epoch 0 (steps 0, 1): active on Band 0
    assert emitter.is_active(0, 0)
    assert emitter.is_active(1, 0)
    assert not emitter.is_active(0, 2)

    # Epoch 1 (steps 2, 3): active on Band 2
    assert emitter.is_active(2, 2)
    assert emitter.is_active(3, 2)

    # Epoch 2 (steps 4, 5): active on Band 1
    assert emitter.is_active(4, 1)
    assert emitter.is_active(5, 1)


def test_transition_predictor():
    kmap = DynamicRFKnowledgeMap(num_bands=3)
    predictor = TransitionPredictor()

    # Simulate transition count updates: 0 -> 2
    for _ in range(5):
        kmap.bands[0].transition_counts[2] = kmap.bands[0].transition_counts.get(2, 0.0) + 1.0

    most_likely = predictor.get_most_likely_next_band(kmap, current_band=0)
    assert most_likely == 2


def test_agile_emitter_simulation_benchmark():
    config = SimConfig(num_bands=4, total_time_steps=80)
    env = RFEnvironment(config)

    # Agile hopper across bands [0, 2, 3]
    env.add_emitter(
        FrequencyAgileEmitter(
            EmitterConfig(
                emitter_id="AGILE_HOPPER",
                emitter_type="agile",
                assigned_bands=[0, 2, 3],
                signal_power_dbm=-50.0,
                parameters={"dwell_time": 4, "hop_sequence": [0, 2, 3]},
            ),
            seed=42,
        )
    )

    rx = ESReceiver(ReceiverConfig(snr_threshold_db=10.0), seed=42)
    runner = SimulationRunner()
    unc_sched = UncertaintyAwareScheduler(num_bands=4, seed=42)

    metrics = runner.run(env, rx, unc_sched, total_steps=80)

    assert metrics.total_scans == 80
    assert metrics.total_intercepts > 0
    assert metrics.spectrum_coverage_ratio == 1.0
