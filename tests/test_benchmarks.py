"""
Unit tests for Final Benchmarking Suite.
"""

import pytest

# pyrefly: ignore [missing-import]
from experiments.run_benchmarks import build_scenario_environment, run_benchmark_suite


def test_build_all_ten_scenarios():
    for s_id in range(1, 11):
        env = build_scenario_environment(s_id, num_bands=6, total_steps=20)
        assert env.config.num_bands == 6
        assert len(env.emitters) >= 1


def test_run_benchmark_suite_execution():
    results = run_benchmark_suite(num_bands=5, total_steps=20, seeds=[42])

    assert len(results) == 50  # 10 scenarios * 1 seed * 5 schedulers
    scheduler_names = set(r["scheduler"] for r in results)
    assert "Sequential Sweep" in scheduler_names
    assert "Proposed Integrated Framework" in scheduler_names

    for r in results:
        assert 0.0 <= r["pd"] <= 1.0
        assert 0.0 <= r["pfa"] <= 1.0
        assert r["total_retuning_cost"] >= 0.0
