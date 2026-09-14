"""
STEP 10 DEMONSTRATION SCRIPT
Final Multi-Seed Benchmarking, Ablation Study & Novelty Positioning
"""

import json
import os
import pandas as pd
import numpy as np

from smart_ew.environment.schema import SimConfig, EmitterConfig
from smart_ew.environment.simulator import RFEnvironment
from smart_ew.emitters.continuous import ContinuousEmitter
from smart_ew.emitters.intermittent import IntermittentEmitter
from smart_ew.emitters.periodic import PeriodicEmitter
from smart_ew.emitters.agile import FrequencyAgileEmitter
from smart_ew.receiver.schema import ReceiverConfig
from smart_ew.receiver.model import ESReceiver
from smart_ew.scheduling.sequential import SequentialScheduler
from smart_ew.scheduling.random_scan import RandomScheduler
from smart_ew.scheduling.ml_greedy import MLGreedyScheduler
from smart_ew.scheduling.contextual_bandit import LinUCBScheduler
from smart_ew.scheduling.uncertainty_aware import UncertaintyAwareScheduler
from smart_ew.scheduling.change_adaptive import ChangeAdaptiveScheduler
from smart_ew.evaluation.runner import SimulationRunner


def run_step10_demo():
    print("=" * 90)
    print("STEP 10: FINAL BENCHMARKING, ABLATION STUDY & NOVELTY POSITIONING")
    print("=" * 90)

    # 1. Multi-Seed Benchmark Summary across 5 Scenarios
    print("\nPhase 1: Multi-Seed Benchmark Evaluation (Mean Metrics across 5 Scenarios & 3 Seeds)")
    
    runner = SimulationRunner()
    scenarios = [
        ("Scenario 1 (Static)", [ContinuousEmitter(EmitterConfig("C1", "continuous", [1], -50.0))]),
        ("Scenario 2 (Intermittent)", [IntermittentEmitter(EmitterConfig("I1", "intermittent", [2], -50.0, {"p_on": 0.3, "p_off": 0.4}), seed=42)]),
        ("Scenario 3 (Periodic)", [PeriodicEmitter(EmitterConfig("P1", "periodic", [3], -48.0, {"period": 10, "duty_duration": 2}))]),
        ("Scenario 4 (Agile)", [FrequencyAgileEmitter(EmitterConfig("A1", "agile", [0, 2, 5], -50.0, {"dwell_time": 3, "hop_sequence": [0, 2, 5]}), seed=42)]),
        ("Scenario 5 (Shift)", [ContinuousEmitter(EmitterConfig("E1", "continuous", [0], -50.0, {"start_time": 0, "end_time": 50})), ContinuousEmitter(EmitterConfig("E2", "continuous", [7], -50.0, {"start_time": 50, "end_time": 100}))]),
    ]

    bench_results = []
    seeds = [42, 123, 456]

    for sc_name, emitters in scenarios:
        for seed in seeds:
            config = SimConfig(num_bands=10, total_time_steps=100)
            env = RFEnvironment(config)
            for e in emitters:
                env.add_emitter(e)

            rx = ESReceiver(ReceiverConfig(), seed=seed)

            schedulers = {
                "A. Sequential Sweep": SequentialScheduler(num_bands=10),
                "B. Uniform Random": RandomScheduler(num_bands=10, seed=seed),
                "C. ML-Greedy Baseline": MLGreedyScheduler(num_bands=10, seed=seed),
                "D. Proposed Integrated Framework": ChangeAdaptiveScheduler(num_bands=10, seed=seed),
            }

            for s_name, scheduler in schedulers.items():
                metrics = runner.run(env, rx, scheduler, total_steps=100)
                bench_results.append({
                    "scenario": sc_name,
                    "seed": seed,
                    "scheduler": s_name,
                    "pd": metrics.probability_of_detection,
                    "pfa": metrics.probability_of_false_alarm,
                    "intercept_rate": metrics.intercept_rate,
                    "avg_intercept_time": metrics.average_intercept_time,
                    "spectrum_coverage": metrics.spectrum_coverage_ratio,
                    "total_retuning_cost": metrics.total_retuning_cost,
                    "cumulative_reward": metrics.cumulative_reward,
                })

    df_bench = pd.DataFrame(bench_results)
    summary_bench = df_bench.groupby("scheduler")[["intercept_rate", "avg_intercept_time", "spectrum_coverage", "total_retuning_cost", "cumulative_reward"]].mean().reset_index()

    print(f"\n{'Scheduler Architecture':<34} | {'Intercept Rate':<16} | {'Avg Delay (steps)':<18} | {'Coverage':<12} | {'Retune Cost':<12} | {'Cumulative Reward':<18}")
    print("-" * 120)

    for idx, row in summary_bench.iterrows():
        print(
            f"{row['scheduler']:<34} | "
            f"{row['intercept_rate']:<16.2%} | "
            f"{row['avg_intercept_time']:<18.2f} | "
            f"{row['spectrum_coverage']:<12.0%} | "
            f"{row['total_retuning_cost']:<12.1f} | "
            f"{row['cumulative_reward']:<18.1f}"
        )

    print("-" * 120)

    # 2. System Component Ablation Study
    print("\nPhase 2: Systematic Component Ablation Study (Component Contribution Analysis)")
    
    ablation_variants = {
        "Model A (Base ML-Greedy)": MLGreedyScheduler(num_bands=10, epsilon=0.05, seed=42),
        "Model B (Base + Uncertainty)": UncertaintyAwareScheduler(num_bands=10, beta_uncertainty=0.8, gamma_recency=0.0, seed=42),
        "Model C (Base + Recency)": UncertaintyAwareScheduler(num_bands=10, beta_uncertainty=0.0, gamma_recency=0.5, seed=42),
        "Model D (Base + LinUCB)": LinUCBScheduler(num_bands=10, alpha=0.5, seed=42),
        "Model E (Full Proposed Framework)": ChangeAdaptiveScheduler(num_bands=10, seed=42),
    }

    ablation_results = []
    for sc_name, emitters in scenarios:
        config = SimConfig(num_bands=10, total_time_steps=100)
        env = RFEnvironment(config)
        for e in emitters:
            env.add_emitter(e)

        rx = ESReceiver(ReceiverConfig(), seed=42)

        for v_name, scheduler in ablation_variants.items():
            metrics = runner.run(env, rx, scheduler, total_steps=100)
            ablation_results.append({
                "scenario": sc_name,
                "variant": v_name,
                "pd": metrics.probability_of_detection,
                "intercept_rate": metrics.intercept_rate,
                "avg_intercept_time": metrics.average_intercept_time,
                "cumulative_reward": metrics.cumulative_reward,
            })

    df_abl = pd.DataFrame(ablation_results)
    summary_abl = df_abl.groupby("variant")[["intercept_rate", "avg_intercept_time", "cumulative_reward"]].mean().reset_index()

    print(f"\n{'Ablation Model Variant':<36} | {'Intercept Rate':<16} | {'Avg Intercept Delay':<22} | {'Cumulative Reward':<18}")
    print("-" * 98)

    for idx, row in summary_abl.iterrows():
        print(
            f"{row['variant']:<36} | "
            f"{row['intercept_rate']:<16.2%} | "
            f"{row['avg_intercept_time']:<22.2f} | "
            f"{row['cumulative_reward']:<18.1f}"
        )

    print("-" * 98)

    # 3. SIH Technical Contribution & Novelty Statement
    print("\n" + "=" * 90)
    print("PROJECT TECHNICAL CONTRIBUTION STATEMENT FOR SIH JUDGES:")
    print("=" * 90)
    print("""
"An uncertainty-aware, change-adaptive, multi-objective RF scan scheduling framework
that combines activity prediction, temporal-frequency behavior modeling, dynamic knowledge
representation, exploration/exploitation, and environment-change adaptation for intelligent
receiver scheduling under limited instantaneous bandwidth and unreliable prior emitter intelligence."
    """)
    print("=" * 90)


if __name__ == "__main__":
    run_step10_demo()
