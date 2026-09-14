"""
STEP 4 DEMONSTRATION SCRIPT
Full Evaluation Metrics Harness - Benchmarking Baseline Strategies
"""

import os
import numpy as np

from smart_ew.environment.schema import SimConfig, EmitterConfig
from smart_ew.environment.simulator import RFEnvironment
from smart_ew.emitters.continuous import ContinuousEmitter
from smart_ew.emitters.intermittent import IntermittentEmitter
from smart_ew.receiver.schema import ReceiverConfig
from smart_ew.receiver.model import ESReceiver
from smart_ew.scheduling.sequential import SequentialScheduler
from smart_ew.scheduling.random_scan import RandomScheduler
from smart_ew.evaluation.runner import SimulationRunner


def run_step4_demo():
    print("=" * 80)
    print("STEP 4: EVALUATION METRICS HARNESS - FULL PERFORMANCE BENCHMARK")
    print("=" * 80)

    # 1. Environment Setup (10 Bands, 100 Steps, 3 Emitters)
    sim_config = SimConfig(num_bands=10, total_time_steps=100, seed=42)
    env = RFEnvironment(sim_config)

    # E1: Band 1 (Continuous t=10..80)
    env.add_emitter(
        ContinuousEmitter(
            EmitterConfig(
                emitter_id="Continuous_E1",
                emitter_type="continuous",
                assigned_bands=[1],
                signal_power_dbm=-75.0,
                parameters={"start_time": 10, "end_time": 80},
            )
        )
    )
    # E2: Band 4 (Intermittent Markov switching)
    env.add_emitter(
        IntermittentEmitter(
            EmitterConfig(
                emitter_id="Intermittent_E2",
                emitter_type="intermittent",
                assigned_bands=[4],
                signal_power_dbm=-80.0,
                parameters={"p_on": 0.3, "p_off": 0.3},
            ),
            seed=42,
        )
    )
    # E3: Band 7 (Continuous t=30..95)
    env.add_emitter(
        ContinuousEmitter(
            EmitterConfig(
                emitter_id="Continuous_E3",
                emitter_type="continuous",
                assigned_bands=[7],
                signal_power_dbm=-70.0,
                parameters={"start_time": 30, "end_time": 95},
            )
        )
    )

    rx_config = ReceiverConfig(
        instantaneous_bw_mhz=50.0,
        retuning_time_cost=1.0,
        snr_threshold_db=10.0,
        p_false_alarm=0.01,
    )
    rx = ESReceiver(config=rx_config, seed=42)
    runner = SimulationRunner()

    # 2. Evaluate Sequential Scheduler
    seq_scheduler = SequentialScheduler(num_bands=10)
    seq_metrics = runner.run(env, rx, seq_scheduler, total_steps=100)

    # 3. Evaluate Random Scheduler
    rand_scheduler = RandomScheduler(num_bands=10, seed=123)
    rand_metrics = runner.run(env, rx, rand_scheduler, total_steps=100)

    # 4. Print Comprehensive 10-Metric Performance Table
    print("\n" + "=" * 80)
    print(f"{'METRIC NAME':<32} | {'SEQUENTIAL SWEEP':<20} | {'RANDOM SWEEP':<20}")
    print("=" * 80)

    metrics_list = [
        ("1. Probability of Detection (Pd)", f"{seq_metrics.probability_of_detection:.2%}", f"{rand_metrics.probability_of_detection:.2%}"),
        ("2. Probability of False Alarm (Pfa)", f"{seq_metrics.probability_of_false_alarm:.2%}", f"{rand_metrics.probability_of_false_alarm:.2%}"),
        ("3. Intercept Rate", f"{seq_metrics.intercept_rate:.2%}", f"{rand_metrics.intercept_rate:.2%}"),
        ("4. Avg Intercept Time (steps)", f"{seq_metrics.average_intercept_time:.2f}", f"{rand_metrics.average_intercept_time:.2f}"),
        ("5. Intercept Delay Std (steps)", f"{seq_metrics.intercept_time_error_std:.2f}", f"{rand_metrics.intercept_time_error_std:.2f}"),
        ("6. Spectrum Coverage Ratio", f"{seq_metrics.spectrum_coverage_ratio:.2%}", f"{rand_metrics.spectrum_coverage_ratio:.2%}"),
        ("7. Total Retuning Cost (units)", f"{seq_metrics.total_retuning_cost:.1f}", f"{rand_metrics.total_retuning_cost:.1f}"),
        ("8. Cumulative Reward", f"{seq_metrics.cumulative_reward:.1f}", f"{rand_metrics.cumulative_reward:.1f}"),
        ("9. Total Intercepts Achieved", f"{seq_metrics.total_intercepts}", f"{rand_metrics.total_intercepts}"),
        ("10. Total False Alarms", f"{seq_metrics.total_false_alarms}", f"{rand_metrics.total_false_alarms}"),
    ]

    for name, val_seq, val_rand in metrics_list:
        print(f"{name:<32} | {val_seq:<20} | {val_rand:<20}")

    print("=" * 80)
    print("\nSummary Dictionary Output:")
    print("Sequential Summary:", seq_metrics.summary_dict())
    print("Random Summary:    ", rand_metrics.summary_dict())
    print("=" * 80)


if __name__ == "__main__":
    run_step4_demo()
