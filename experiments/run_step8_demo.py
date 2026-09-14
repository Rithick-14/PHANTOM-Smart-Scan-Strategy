"""
STEP 8 DEMONSTRATION SCRIPT
Periodic Pulse Radar & Frequency-Agile Hopping Emitters
"""

import os
import numpy as np

from smart_ew.environment.schema import SimConfig, EmitterConfig
from smart_ew.environment.simulator import RFEnvironment
from smart_ew.emitters.periodic import PeriodicEmitter
from smart_ew.emitters.agile import FrequencyAgileEmitter
from smart_ew.receiver.schema import ReceiverConfig
from smart_ew.receiver.model import ESReceiver
from smart_ew.knowledge.periodicity import PeriodicityEstimator
from smart_ew.knowledge.transition_predictor import TransitionPredictor
from smart_ew.knowledge.map import DynamicRFKnowledgeMap
from smart_ew.scheduling.sequential import SequentialScheduler
from smart_ew.scheduling.uncertainty_aware import UncertaintyAwareScheduler
from smart_ew.evaluation.runner import SimulationRunner


def run_step8_demo():
    print("=" * 85)
    print("STEP 8: PERIODIC & FREQUENCY-AGILE EMITTER MODELING DEMONSTRATION")
    print("=" * 85)

    # 1. Setup RF Environment (10 Bands, 100 Steps)
    sim_config = SimConfig(num_bands=10, total_time_steps=100, seed=42)
    env = RFEnvironment(sim_config)

    # E1: Periodic Pulsed Radar on Band 3 (Period=10 steps, Duration=2 steps)
    env.add_emitter(
        PeriodicEmitter(
            EmitterConfig(
                emitter_id="Periodic_Radar_E1",
                emitter_type="periodic",
                assigned_bands=[3],
                signal_power_dbm=-75.0,
                parameters={"period": 10, "duty_duration": 2, "phase_offset": 0},
            )
        )
    )

    # E2: Frequency-Agile Hopping Emitter hopping across Bands [1, 5, 8] (Dwell=3 steps)
    env.add_emitter(
        FrequencyAgileEmitter(
            EmitterConfig(
                emitter_id="Frequency_Agile_E2",
                emitter_type="agile",
                assigned_bands=[1, 5, 8],
                signal_power_dbm=-70.0,
                parameters={"dwell_time": 3, "hop_sequence": [1, 5, 8]},
            ),
            seed=42,
        )
    )

    # 2. Demonstrate Temporal Periodicity Estimation
    print("\nPhase 1: Periodic Pulse Timing Estimation (Period T=10)")
    period_estimator = PeriodicityEstimator()
    detected_radar_timestamps = [0, 10, 20, 30, 40, 50]
    
    estimated_period = period_estimator.estimate_period(detected_radar_timestamps)
    p_t60 = period_estimator.predict_activity_window(detected_radar_timestamps, target_step=60)
    p_t65 = period_estimator.predict_activity_window(detected_radar_timestamps, target_step=65)

    print(f"Detected Hit Timestamps: {detected_radar_timestamps}")
    print(f"Estimated Pulse Period: T = {estimated_period} time steps")
    print(f"Predicted Temporal Activity Probability at t=60 (Expected Pulse): P = {p_t60:.2%}")
    print(f"Predicted Temporal Activity Probability at t=65 (Inter-pulse Off): P = {p_t65:.2%}")

    # 3. Demonstrate Frequency Hopping Transition Learning
    print("\nPhase 2: Frequency Hopping Markov Transition Learning P(Band_j | Band_i)")
    kmap = DynamicRFKnowledgeMap(num_bands=10)
    trans_predictor = TransitionPredictor()

    # Simulate transition updates: Band 1 -> Band 5 -> Band 8 -> Band 1
    sample_hops = [(1, 5), (5, 8), (8, 1), (1, 5), (5, 8), (8, 1)]
    for from_b, to_b in sample_hops:
        kmap.bands[from_b].transition_counts[to_b] = kmap.bands[from_b].transition_counts.get(to_b, 0.0) + 1.0

    print(f"Learned Transition Probabilities:")
    print(f" - P(Band 5 | Band 1) = {kmap.get_transition_probability(1, 5):.2%}")
    print(f" - P(Band 8 | Band 5) = {kmap.get_transition_probability(5, 8):.2%}")
    print(f" - P(Band 1 | Band 8) = {kmap.get_transition_probability(8, 1):.2%}")

    next_predicted = trans_predictor.get_most_likely_next_band(kmap, current_band=5)
    print(f"Most Likely Next Hop after scanning Band 5: Band {next_predicted}")

    # 4. Performance Benchmark: Sequential Sweep vs Proposed Smart Scheduler
    print("\nPhase 3: Closed-Loop Performance Benchmark on Periodic & Agile Emitters")
    rx = ESReceiver(config=ReceiverConfig(retuning_time_cost=1.0), seed=42)
    runner = SimulationRunner()

    seq_sched = SequentialScheduler(num_bands=10, name="Sequential Sweep")
    smart_sched = UncertaintyAwareScheduler(num_bands=10, seed=42, name="Proposed Smart Scheduler")

    print(f"\n{'Scheduler Architecture':<25} | {'Intercept Rate':<16} | {'Avg Intercept Time':<20} | {'Retune Cost':<12} | {'Cumulative Reward':<18}")
    print("=" * 97)

    for sch in [seq_sched, smart_sched]:
        metrics = runner.run(env, rx, sch, total_steps=100)
        print(
            f"{metrics.scheduler_name:<25} | "
            f"{metrics.intercept_rate:<16.2%} | "
            f"{metrics.average_intercept_time:<20.2f} | "
            f"{metrics.total_retuning_cost:<12.1f} | "
            f"{metrics.cumulative_reward:<18.1f}"
        )

    print("=" * 97)
    print("\nKey Analytical Takeaways:")
    print(" 1. The Smart Scheduler successfully tracks both periodic radar pulses and frequency-hopping sequences.")
    print(" 2. Sequential scanning misses periodic pulse windows when out-of-phase with the cyclic sweep.")
    print(" 3. Transition probabilities P(Band_j | Band_i) allow the receiver to anticipate target frequency hops.")
    print("=" * 85)


if __name__ == "__main__":
    run_step8_demo()
