"""
STEP 3 DEMONSTRATION SCRIPT
Baseline Scan Strategies - Sequential vs Uniform Random Sweep Evaluation
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


def evaluate_baseline_scheduler(scheduler, env, rx_config, steps=100, seed=42):
    env.reset()
    receiver = ESReceiver(config=rx_config, seed=seed)
    scheduler.reset()

    total_real_opportunities = 0
    total_detections = 0
    total_false_alarms = 0
    total_tuning_cost = 0.0

    for t in range(steps):
        # 1. Step environment to get ground truth
        gt_frame = env.step()
        
        # Count total active transmission opportunities across all bands
        for b_id, active in gt_frame.active_bands.items():
            if active:
                total_real_opportunities += 1

        # 2. Scheduler selects next band
        target_band = scheduler.select_next_band(t)

        # 3. Receiver scans target band
        obs = receiver.scan(band_id=target_band, time_step=t, ground_truth_frame=gt_frame)

        # 4. Update scheduler (open-loop baseline ignores, but API contract calls update)
        scheduler.update(obs)

        # 5. Track statistics
        total_tuning_cost += obs.tuning_cost_paid
        if obs.detected:
            if obs.is_false_alarm:
                total_false_alarms += 1
            else:
                total_detections += 1

    intercept_rate = (total_detections / total_real_opportunities) if total_real_opportunities > 0 else 0.0

    return {
        "scheduler_name": scheduler.name,
        "steps": steps,
        "real_opportunities": total_real_opportunities,
        "detections": total_detections,
        "false_alarms": total_false_alarms,
        "intercept_rate": intercept_rate,
        "tuning_cost": total_tuning_cost,
    }


def run_step3_demo():
    print("=" * 75)
    print("STEP 3: BASELINE SCAN STRATEGIES - SEQUENTIAL VS RANDOM SWEEP DEMO")
    print("=" * 75)

    # 1. Environment Setup (10 Bands, 100 Steps, 3 Emitters)
    sim_config = SimConfig(num_bands=10, total_time_steps=100, seed=42)
    env = RFEnvironment(sim_config)

    e1 = ContinuousEmitter(
        EmitterConfig(
            emitter_id="Continuous_E1",
            emitter_type="continuous",
            assigned_bands=[1],
            signal_power_dbm=-75.0,
            parameters={"start_time": 10, "end_time": 80},
        )
    )
    e2 = IntermittentEmitter(
        EmitterConfig(
            emitter_id="Intermittent_E2",
            emitter_type="intermittent",
            assigned_bands=[4],
            signal_power_dbm=-80.0,
            parameters={"p_on": 0.3, "p_off": 0.3},
        ),
        seed=42,
    )
    e3 = ContinuousEmitter(
        EmitterConfig(
            emitter_id="Continuous_E3",
            emitter_type="continuous",
            assigned_bands=[7],
            signal_power_dbm=-70.0,
            parameters={"start_time": 30, "end_time": 95},
        )
    )

    env.add_emitter(e1)
    env.add_emitter(e2)
    env.add_emitter(e3)

    rx_config = ReceiverConfig(
        instantaneous_bw_mhz=50.0,
        retuning_time_cost=1.0,
        snr_threshold_db=10.0,
        p_false_alarm=0.01,
    )

    # 2. Evaluate Sequential Scheduler
    seq_scheduler = SequentialScheduler(num_bands=10, start_band=0)
    seq_results = evaluate_baseline_scheduler(seq_scheduler, env, rx_config, steps=100, seed=42)

    # 3. Evaluate Random Scheduler
    rnd_scheduler = RandomScheduler(num_bands=10, seed=42)
    rnd_results = evaluate_baseline_scheduler(rnd_scheduler, env, rx_config, steps=100, seed=42)

    # 4. Display Comparative Baseline Analysis
    print(f"\nSimulation Configuration:")
    print(f" - Spectrum: 10 Bands | Duration: 100 Time Steps")
    print(f" - Active Emitters: 3 Emitters (Bands 1, 4, 7)")
    print(f" - Total Real Transmission Opportunities: {seq_results['real_opportunities']}")
    print()

    print(f"{'Strategy Name':<22} | {'Detections':<11} | {'False Alarms':<12} | {'Intercept Rate':<16} | {'Retune Cost':<12}")
    print("-" * 85)

    for res in [seq_results, rnd_results]:
        print(
            f"{res['scheduler_name']:<22} | "
            f"{res['detections']:<11} | "
            f"{res['false_alarms']:<12} | "
            f"{res['intercept_rate']:<16.2%} | "
            f"{res['tuning_cost']:<12.1f}"
        )

    print("-" * 85)
    print("\nKey Analytical Takeaways:")
    print(" 1. Sequential Scanning achieves ~10% probability of interception in a 10-band spectrum because it sweeps every band blindly in order.")
    print(" 2. Fixed/Sequential scan spends 70% of its sensing time on completely empty, silent bands (Bands 0, 2, 3, 5, 6, 8, 9).")
    print(" 3. Retuning overhead is paid continuously because the receiver changes bands at every time step.")
    print(" 4. This establishes our mandatory baseline. Intelligent AI scheduling MUST significantly exceed ~10-18% baseline intercept rates.")
    print("=" * 75)


if __name__ == "__main__":
    run_step3_demo()
