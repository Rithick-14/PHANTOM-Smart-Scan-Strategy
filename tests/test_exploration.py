"""
Unit tests for Uncertainty Estimator and Exploration Policies (UCB, Epsilon-Greedy, Softmax).
"""

import pytest
import numpy as np

from smart_ew.knowledge.schema import BandKnowledge
from smart_ew.scheduling.exploration import UncertaintyEstimator, ExplorationPolicy


def test_uncertainty_estimator():
    estimator = UncertaintyEstimator()

    bk_unobserved = BandKnowledge(band_id=0, total_observations=0, beta_alpha=1.0, beta_beta=1.0)
    u_unobserved = estimator.compute_combined_uncertainty(bk_unobserved)

    bk_observed = BandKnowledge(band_id=0, total_observations=50, beta_alpha=40.0, beta_beta=10.0)
    u_observed = estimator.compute_combined_uncertainty(bk_observed)

    # Epistemic uncertainty should decrease significantly with observations
    assert u_unobserved > u_observed
    assert 0.0 <= u_observed <= 1.0
    assert 0.0 <= u_unobserved <= 1.0


def test_ensemble_disagreement():
    estimator = UncertaintyEstimator()

    # Low disagreement: models agree
    disagree_low = estimator.compute_ensemble_disagreement([0.80, 0.82, 0.81])
    # High disagreement: models disagree
    disagree_high = estimator.compute_ensemble_disagreement([0.90, 0.20, 0.50])

    assert disagree_high > disagree_low


def test_exploration_policies():
    probas = {0: 0.8, 1: 0.2, 2: 0.1}
    uncertainties = {0: 0.05, 1: 0.9, 2: 0.1}

    # UCB Policy: Band 1 has high uncertainty (0.9), so UCB score = 0.2 + 1.5*0.9 = 1.55 > Band 0 (0.8 + 1.5*0.05 = 0.875)
    ucb_policy = ExplorationPolicy(strategy="ucb", c_ucb=1.5, seed=42)
    selected_ucb = ucb_policy.select_action(probas, uncertainties)
    assert selected_ucb == 1

    # Epsilon-Greedy Policy (eps=0.0 -> purely greedy): selects Band 0
    greedy_policy = ExplorationPolicy(strategy="epsilon_greedy", epsilon=0.0, seed=42)
    selected_greedy = greedy_policy.select_action(probas, uncertainties)
    assert selected_greedy == 0

    # Softmax Policy
    softmax_policy = ExplorationPolicy(strategy="softmax", temperature=0.5, seed=42)
    selected_softmax = softmax_policy.select_action(probas, uncertainties)
    assert selected_softmax in [0, 1, 2]
