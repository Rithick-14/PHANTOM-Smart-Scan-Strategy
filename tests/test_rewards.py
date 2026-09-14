"""
Unit tests for Multi-Objective Reward Engine.
"""

import pytest

from smart_ew.rewards.schema import RewardConfig
from smart_ew.rewards.engine import MultiObjectiveRewardEngine
from smart_ew.receiver.schema import ReceiverObservation
from smart_ew.knowledge.map import DynamicRFKnowledgeMap


def test_reward_engine_hit_and_priority():
    config = RewardConfig(
        w_hit=2.0,
        w_threat_priority=1.5,
        w_false_alarm=-1.0,
        w_retuning_cost=-0.5,
        band_priorities={2: 2.0},  # Band 2 has priority 2.0
    )
    engine = MultiObjectiveRewardEngine(config)

    obs_hit = ReceiverObservation(
        band_id=2,
        timestamp=0,
        detected=True,
        measured_power_dbm=-60.0,
        estimated_snr_db=20.0,
        confidence=0.9,
        tuning_cost_paid=0.0,
        is_false_alarm=False,
    )

    comps = engine.compute_components(obs_hit)
    assert comps["hit_reward"] == 2.0
    assert comps["threat_priority_reward"] == 1.5 * 2.0  # 3.0
    assert comps["false_alarm_penalty"] == 0.0
    assert comps["retuning_penalty"] == 0.0
    assert comps["total_reward"] == 5.0


def test_reward_engine_false_alarm_and_retuning():
    config = RewardConfig(
        w_hit=2.0, w_false_alarm=-1.0, w_retuning_cost=-0.5
    )
    engine = MultiObjectiveRewardEngine(config)

    obs_fa = ReceiverObservation(
        band_id=0,
        timestamp=1,
        detected=True,
        measured_power_dbm=-98.0,
        estimated_snr_db=2.0,
        confidence=0.1,
        tuning_cost_paid=1.0,  # Retuned band
        is_false_alarm=True,
    )

    comps = engine.compute_components(obs_fa)
    assert comps["hit_reward"] == 0.0
    assert comps["false_alarm_penalty"] == -1.0
    assert comps["retuning_penalty"] == -0.5
    assert comps["total_reward"] == -1.5


def test_reward_engine_discovery_bonus():
    config = RewardConfig(w_hit=2.0, w_discovery_bonus=1.0)
    engine = MultiObjectiveRewardEngine(config)
    kmap = DynamicRFKnowledgeMap(num_bands=3)

    obs = ReceiverObservation(
        band_id=1,
        timestamp=0,
        detected=True,
        measured_power_dbm=-60.0,
        estimated_snr_db=20.0,
        confidence=0.9,
        tuning_cost_paid=0.0,
        is_false_alarm=False,
    )

    comps = engine.compute_components(obs, knowledge_map=kmap)
    assert comps["discovery_bonus"] > 0.0
    assert comps["total_reward"] > 2.0
