"""
Ablation Experiments Script.
Measures the individual marginal performance contribution of each proposed component:
A: Base ML-Greedy
B: Base + Uncertainty Exploration
C: Base + Temporal Periodicity Engine
D: Base + Environment Change Detection
E: Base + Frequency Transition Information
F: Full Proposed Integrated Framework
"""

import json
import os
import sys
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Ensure src is on Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from smart_ew.environment.schema import SimConfig, EmitterConfig
from smart_ew.environment.simulator import RFEnvironment
from smart_ew.emitters.continuous import ContinuousEmitter
from smart_ew.emitters.intermittent import IntermittentEmitter
from smart_ew.emitters.periodic import PeriodicEmitter
from smart_ew.emitters.agile import FrequencyAgileEmitter
from smart_ew.receiver.schema import ReceiverConfig
from smart_ew.receiver.model import ESReceiver
from smart_ew.scheduling.ml_greedy import MLGreedyScheduler
from smart_ew.scheduling.contextual_bandit import LinUCBScheduler
from smart_ew.scheduling.uncertainty_aware import UncertaintyAwareScheduler
from smart_ew.scheduling.change_adaptive import ChangeAdaptiveScheduler
from smart_ew.evaluation.runner import SimulationRunner


def build_ablation_environment(scenario_type: str, num_bands: int = 8, total_steps: int = 100) -> RFEnvironment:
    config = SimConfig(num_bands=num_bands, total_time_steps=total_steps)
    env = RFEnvironment(config)

    if scenario_type == "static":
        env.add_emitter(ContinuousEmitter(EmitterConfig("C1", "continuous", [1], -50.0)))
    elif scenario_type == "intermittent":
        env.add_emitter(
            IntermittentEmitter(
                EmitterConfig("I1", "intermittent", [2], -50.0, {"p_on": 0.3, "p_off": 0.4}), seed=42
            )
        )
    elif scenario_type == "periodic":
        env.add_emitter(
            PeriodicEmitter(EmitterConfig("P1", "periodic", [3], -50.0, {"period": 8, "duty_duration": 2}))
        )
    elif scenario_type == "agile":
        env.add_emitter(
            FrequencyAgileEmitter(
                EmitterConfig("A1", "agile", [0, 2, 5], -50.0, {"dwell_time": 2, "hop_sequence": [0, 2, 5]}),
                seed=42,
            )
        )
    elif scenario_type == "shift":
        env.add_emitter(
            ContinuousEmitter(EmitterConfig("E1", "continuous", [0], -50.0, {"start_time": 0, "end_time": 45}))
        )
        env.add_emitter(
            ContinuousEmitter(EmitterConfig("E2", "continuous", [6], -50.0, {"start_time": 50, "end_time": 100}))
        )

    return env


def run_ablation_suite(num_bands: int = 8, total_steps: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    runner = SimulationRunner()
    scenarios = ["static", "intermittent", "periodic", "agile", "shift"]
    results = []

    for sc in scenarios:
        env = build_ablation_environment(sc, num_bands, total_steps)
        rx = ESReceiver(ReceiverConfig(), seed=seed)

        variants = {
            "A: Base ML-Greedy": MLGreedyScheduler(num_bands, epsilon=0.05, seed=seed),
            "B: Base + Uncertainty": UncertaintyAwareScheduler(num_bands, beta_uncertainty=0.8, gamma_recency=0.0, seed=seed),
            "C: Base + Recency": UncertaintyAwareScheduler(num_bands, beta_uncertainty=0.0, gamma_recency=0.5, seed=seed),
            "D: Base + LinUCB": LinUCBScheduler(num_bands, alpha=0.5, seed=seed),
            "E: Base + Change Detection": ChangeAdaptiveScheduler(num_bands, change_threshold=3.0, seed=seed),
            "F: Full Proposed Framework": ChangeAdaptiveScheduler(num_bands, change_threshold=2.0, seed=seed),
        }

        for var_name, scheduler in variants.items():
            metrics = runner.run(env, rx, scheduler, total_steps=total_steps)
            rec = {
                "scenario": sc,
                "variant": var_name,
                "pd": round(metrics.probability_of_detection, 4),
                "pfa": round(metrics.probability_of_false_alarm, 4),
                "avg_intercept_time": round(metrics.average_intercept_time, 2),
                "spectrum_coverage": round(metrics.spectrum_coverage_ratio, 4),
                "total_retuning_cost": round(metrics.total_retuning_cost, 2),
                "total_intercepts": metrics.total_intercepts,
            }
            results.append(rec)

    return results


def main():
    print("Running Systematic Ablation Experiments Suite...")
    results = run_ablation_suite()

    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "ablation_results.json")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    print("\nAblation Summary Table (Mean metrics across scenarios):")
    summary_df = df.groupby("variant")[["pd", "pfa", "avg_intercept_time", "spectrum_coverage"]].mean().reset_index()
    print(summary_df.to_string(index=False))
    print(f"\nAblation dataset saved to: {json_path}")


if __name__ == "__main__":
    main()
