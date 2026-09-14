"""
Final Multi-Seed Benchmarking Script across Scenarios 1 to 10.
Compares Sequential Sweep, Uniform Random, ML-Greedy, LinUCB, and Proposed Framework.
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
from smart_ew.scheduling.sequential import SequentialScheduler
from smart_ew.scheduling.random_scan import RandomScheduler
from smart_ew.scheduling.ml_greedy import MLGreedyScheduler
from smart_ew.scheduling.contextual_bandit import LinUCBScheduler
from smart_ew.scheduling.change_adaptive import ChangeAdaptiveScheduler
from smart_ew.evaluation.runner import SimulationRunner


def build_scenario_environment(scenario_id: int, num_bands: int = 10, total_steps: int = 100) -> RFEnvironment:
    if scenario_id == 1:
        # Scenario 1: Static / Continuous Emitters
        config = SimConfig(num_bands=num_bands, total_time_steps=total_steps)
        env = RFEnvironment(config)
        env.add_emitter(ContinuousEmitter(EmitterConfig("S1_C1", "continuous", [1], -50.0)))
        env.add_emitter(ContinuousEmitter(EmitterConfig("S1_C2", "continuous", [num_bands - 2], -55.0)))
        return env

    elif scenario_id == 2:
        # Scenario 2: Intermittent Poisson Emitters
        config = SimConfig(num_bands=num_bands, total_time_steps=total_steps)
        env = RFEnvironment(config)
        env.add_emitter(IntermittentEmitter(EmitterConfig("S2_I1", "intermittent", [2], -50.0, {"p_on": 0.3, "p_off": 0.4}), seed=42))
        env.add_emitter(IntermittentEmitter(EmitterConfig("S2_I2", "intermittent", [num_bands - 3], -55.0, {"p_on": 0.2, "p_off": 0.5}), seed=123))
        return env

    elif scenario_id == 3:
        # Scenario 3: Periodic Scanning Emitter
        config = SimConfig(num_bands=num_bands, total_time_steps=total_steps)
        env = RFEnvironment(config)
        env.add_emitter(PeriodicEmitter(EmitterConfig("S3_P1", "periodic", [3], -48.0, {"period": 10, "duty_duration": 2})))
        return env

    elif scenario_id == 4:
        # Scenario 4: Frequency-Agile Hopping Emitter
        config = SimConfig(num_bands=num_bands, total_time_steps=total_steps)
        env = RFEnvironment(config)
        env.add_emitter(FrequencyAgileEmitter(EmitterConfig("S4_A1", "agile", [0, 3, num_bands - 1], -50.0, {"dwell_time": 3, "hop_sequence": [0, 3, num_bands - 1]}), seed=42))
        return env

    elif scenario_id == 5:
        # Scenario 5: Sudden Emitter Appearance
        config = SimConfig(num_bands=num_bands, total_time_steps=total_steps)
        env = RFEnvironment(config)
        env.add_emitter(ContinuousEmitter(EmitterConfig("S5_C1", "continuous", [1], -50.0)))
        env.add_emitter(ContinuousEmitter(EmitterConfig("S5_LATE", "continuous", [num_bands - 1], -50.0, {"start_time": total_steps // 2})))
        return env

    elif scenario_id == 6:
        # Scenario 6: Sudden Environment Shift
        config = SimConfig(num_bands=num_bands, total_time_steps=total_steps)
        env = RFEnvironment(config)
        env.add_emitter(ContinuousEmitter(EmitterConfig("S6_E1", "continuous", [0], -50.0, {"start_time": 0, "end_time": total_steps // 2})))
        env.add_emitter(ContinuousEmitter(EmitterConfig("S6_E2", "continuous", [num_bands - 1], -50.0, {"start_time": total_steps // 2, "end_time": total_steps})))
        return env

    elif scenario_id == 7:
        # Scenario 7: High Noise / Low SNR Regime
        config = SimConfig(num_bands=num_bands, total_time_steps=total_steps, noise_floor_dbm=-90.0)
        env = RFEnvironment(config)
        env.add_emitter(ContinuousEmitter(EmitterConfig("S7_LOW_SNR", "continuous", [2], -85.0)))
        return env

    elif scenario_id == 8:
        # Scenario 8: Dense Spectrum (Many emitters)
        config = SimConfig(num_bands=num_bands, total_time_steps=total_steps)
        env = RFEnvironment(config)
        for b in range(0, num_bands, 2):
            env.add_emitter(ContinuousEmitter(EmitterConfig(f"S8_D_{b}", "continuous", [b], -55.0)))
        return env

    elif scenario_id == 9:
        # Scenario 9: Ultra-Sparse Emitter Activity
        config = SimConfig(num_bands=num_bands, total_time_steps=total_steps)
        env = RFEnvironment(config)
        env.add_emitter(IntermittentEmitter(EmitterConfig("S9_SPARSE", "intermittent", [4], -50.0, {"p_on": 0.05, "p_off": 0.8}), seed=99))
        return env

    elif scenario_id == 10:
        # Scenario 10: Completely Unknown Environment
        config = SimConfig(num_bands=num_bands, total_time_steps=total_steps)
        env = RFEnvironment(config)
        env.add_emitter(FrequencyAgileEmitter(EmitterConfig("S10_UNK_AGILE", "agile", [1, 4, 7], -50.0, {"dwell_time": 2, "hop_sequence": [1, 4, 7]}), seed=77))
        env.add_emitter(PeriodicEmitter(EmitterConfig("S10_UNK_RADAR", "periodic", [2], -50.0, {"period": 7, "duty_duration": 1})))
        return env

    else:
        config = SimConfig(num_bands=num_bands, total_time_steps=total_steps)
        return RFEnvironment(config)


def run_benchmark_suite(num_bands: int = 10, total_steps: int = 100, seeds: List[int] = [42, 123, 456]) -> List[Dict[str, Any]]:
    runner = SimulationRunner()
    results = []

    for s_id in range(1, 11):
        for seed in seeds:
            env = build_scenario_environment(s_id, num_bands, total_steps)
            rx = ESReceiver(ReceiverConfig(), seed=seed)

            schedulers = {
                "Sequential Sweep": SequentialScheduler(num_bands),
                "Uniform Random": RandomScheduler(num_bands, seed=seed),
                "ML-Greedy Baseline": MLGreedyScheduler(num_bands, seed=seed),
                "LinUCB Bandit": LinUCBScheduler(num_bands, alpha=0.5, seed=seed),
                "Proposed Integrated Framework": ChangeAdaptiveScheduler(num_bands, seed=seed),
            }

            for sched_name, scheduler in schedulers.items():
                metrics = runner.run(env, rx, scheduler, total_steps=total_steps)
                rec = {
                    "scenario_id": s_id,
                    "seed": seed,
                    "scheduler": sched_name,
                    "pd": float(metrics.probability_of_detection),
                    "pfa": float(metrics.probability_of_false_alarm),
                    "avg_intercept_time": float(metrics.average_intercept_time),
                    "spectrum_coverage": float(metrics.spectrum_coverage_ratio),
                    "total_retuning_cost": float(metrics.total_retuning_cost),
                    "total_intercepts": int(metrics.total_intercepts),
                }
                results.append(rec)

    return results


def main():
    print("Running Final Multi-Seed Benchmarking Suite (Scenarios 1-10)...")
    results = run_benchmark_suite()

    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "benchmark_results.json")
    report_path = os.path.join(output_dir, "benchmark_report.md")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    summary = df.groupby(["scheduler"])[["pd", "pfa", "avg_intercept_time", "spectrum_coverage", "total_retuning_cost"]].mean().reset_index()

    report_content = f"# Final Benchmarking Report - Smart Scan EW Scheduler\n\n"
    report_content += "## Summary Benchmark Table (Averaged across 10 Scenarios & Seeds)\n\n"
    report_content += summary.to_markdown(index=False)
    report_content += "\n\n## Per-Scenario Performance Breakdown\n\n"

    for s_id in range(1, 11):
        s_df = df[df["scenario_id"] == s_id].groupby("scheduler")[["pd", "pfa", "avg_intercept_time", "spectrum_coverage"]].mean().reset_index()
        report_content += f"### Scenario {s_id}\n"
        report_content += s_df.to_markdown(index=False)
        report_content += "\n\n"

    with open(report_path, "w") as f:
        f.write(report_content)

    print("\nBenchmark Summary Table:")
    print(summary.to_string(index=False))
    print(f"\nRaw benchmark data saved to: {json_path}")
    print(f"Benchmark report saved to: {report_path}")


if __name__ == "__main__":
    main()
