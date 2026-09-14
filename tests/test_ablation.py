"""
Unit tests for Ablation Experiments Suite.
"""

import pytest

# pyrefly: ignore [missing-import]
from experiments.run_ablation import run_ablation_suite, build_ablation_environment


def test_ablation_environment_creation():
    for sc in ["static", "intermittent", "periodic", "agile", "shift"]:
        env = build_ablation_environment(sc, num_bands=5, total_steps=20)
        assert env.config.num_bands == 5
        assert len(env.emitters) >= 1


def test_run_ablation_suite_execution():
    results = run_ablation_suite(num_bands=5, total_steps=30, seed=42)

    assert len(results) > 0
    variant_names = set(r["variant"] for r in results)
    assert "A: Base ML-Greedy" in variant_names
    assert "F: Full Proposed Framework" in variant_names

    for r in results:
        assert 0.0 <= r["pd"] <= 1.0
        assert 0.0 <= r["pfa"] <= 1.0
        assert r["total_retuning_cost"] >= 0.0
