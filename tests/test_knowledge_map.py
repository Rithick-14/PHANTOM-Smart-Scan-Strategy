"""
Unit tests for Dynamic RF Knowledge Map.
"""

import pytest
import numpy as np

from smart_ew.knowledge.schema import BandKnowledge
from smart_ew.knowledge.map import DynamicRFKnowledgeMap
from smart_ew.receiver.schema import ReceiverObservation


def test_knowledge_map_initialization():
    kmap = DynamicRFKnowledgeMap(num_bands=5)
    assert len(kmap.bands) == 5

    bk0 = kmap.get_band_knowledge(0, current_step=0)
    assert bk0.beta_alpha == 1.0
    assert bk0.beta_beta == 1.0
    assert bk0.activity_probability == 0.5
    assert bk0.confidence == 0.0
    assert bk0.uncertainty > 0.0


def test_knowledge_map_hit_and_miss_updates():
    kmap = DynamicRFKnowledgeMap(num_bands=3)

    # Hit on band 1
    obs_hit = ReceiverObservation(1, 0, True, -60.0, 20.0, 0.9, 0.0, False)
    kmap.update(obs_hit)

    bk1 = kmap.get_band_knowledge(1, current_step=1)
    assert bk1.hits == 1
    assert bk1.beta_alpha == 2.0
    assert bk1.activity_probability == 2.0 / 3.0  # 2 / (2 + 1)
    assert bk1.confidence > 0.0

    # Miss on band 1
    obs_miss = ReceiverObservation(1, 1, False, -100.0, 0.0, 0.9, 0.0, False)
    kmap.update(obs_miss)

    bk1_after = kmap.get_band_knowledge(1, current_step=2)
    assert bk1_after.hits == 1
    assert bk1_after.misses == 1
    assert bk1_after.beta_alpha == 2.0
    assert bk1_after.beta_beta == 2.0
    assert bk1_after.activity_probability == 0.5  # 2 / (2 + 2)


def test_frequency_transition_tracking():
    kmap = DynamicRFKnowledgeMap(num_bands=4)

    # Sequence of hits: band 0 at step 0 -> band 2 at step 1
    obs0 = ReceiverObservation(0, 0, True, -50.0, 20.0, 0.9, 0.0, False)
    obs1 = ReceiverObservation(2, 1, True, -50.0, 20.0, 0.9, 0.0, False)

    kmap.update(obs0)
    kmap.update(obs1)

    prob_0_to_2 = kmap.get_transition_probability(from_band=0, to_band=2)
    prob_0_to_1 = kmap.get_transition_probability(from_band=0, to_band=1)

    assert prob_0_to_2 == 1.0
    assert prob_0_to_1 == 0.0


def test_knowledge_decay():
    kmap = DynamicRFKnowledgeMap(num_bands=2)
    obs = ReceiverObservation(0, 0, True, -50.0, 20.0, 0.9, 0.0, False)

    for _ in range(10):
        kmap.update(obs)

    bk_before = kmap.get_band_knowledge(0, current_step=10)
    alpha_before = bk_before.beta_alpha

    kmap.decay_stale_knowledge(decay_factor=0.8)

    bk_after = kmap.get_band_knowledge(0, current_step=10)
    assert bk_after.beta_alpha < alpha_before
    assert bk_after.beta_alpha >= 1.0
