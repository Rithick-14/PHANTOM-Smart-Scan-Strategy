"""
STEP 1 DEMONSTRATION SCRIPT
RF Environment Simulator - Ground Truth Spectrum Activity Generation
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from smart_ew.environment.schema import SimConfig, EmitterConfig
from smart_ew.environment.simulator import RFEnvironment
from smart_ew.emitters.continuous import ContinuousEmitter
from smart_ew.emitters.intermittent import IntermittentEmitter


def run_step1_demo():
    print("=" * 70)
    print("STEP 1: RF ENVIRONMENT SIMULATOR - GROUND TRUTH GENERATION")
    print("=" * 70)

    # 1. Configuration: 10 frequency bands, 100 time slots
    config = SimConfig(
        num_bands=10,
        f_start_mhz=1000.0,
        f_step_mhz=50.0,
        bandwidth_mhz=50.0,
        total_time_steps=100,
        noise_floor_dbm=-100.0,
        seed=42
    )

    env = RFEnvironment(config)

    # 2. Add 3 Emitters
    # Emitter 1: Continuous active from t=10 to t=80 on Band 1 (1050 MHz)
    e1 = ContinuousEmitter(
        EmitterConfig(
            emitter_id="Emitter_E1_Continuous",
            emitter_type="continuous",
            assigned_bands=[1],
            signal_power_dbm=-35.0,
            parameters={"start_time": 10, "end_time": 80}
        )
    )

    # Emitter 2: Intermittent Markov switching on Band 4 (1200 MHz)
    e2 = IntermittentEmitter(
        EmitterConfig(
            emitter_id="Emitter_E2_Intermittent",
            emitter_type="intermittent",
            assigned_bands=[4],
            signal_power_dbm=-45.0,
            parameters={"p_on": 0.25, "p_off": 0.35, "initial_active": False}
        ),
        seed=42
    )

    # Emitter 3: Continuous active from t=30 to t=95 on Band 7 (1350 MHz)
    e3 = ContinuousEmitter(
        EmitterConfig(
            emitter_id="Emitter_E3_Continuous",
            emitter_type="continuous",
            assigned_bands=[7],
            signal_power_dbm=-40.0,
            parameters={"start_time": 30, "end_time": 95}
        )
    )

    env.add_emitter(e1)
    env.add_emitter(e2)
    env.add_emitter(e3)

    print(f"\nInitialized RF Environment with {config.num_bands} bands across {config.f_start_mhz} - {config.f_start_mhz + config.num_bands * config.f_step_mhz} MHz.")
    print(f"Registered {len(env.emitters)} emitters:")
    for e in env.emitters:
        print(f" - ID: {e.emitter_id:<25} | Type: {e.emitter_type:<15} | Bands: {e.assigned_bands} | Power: {e.signal_power_dbm} dBm")

    # 3. Generate Ground Truth Activity Matrix
    activity_matrix = env.get_ground_truth_matrix(total_steps=100)
    power_matrix = env.get_power_matrix(total_steps=100)

    print("\n--- GROUND TRUTH SUMMARY ---")
    print(f"Activity Matrix Shape: {activity_matrix.shape} (Bands x Time Steps)")
    total_active_cells = np.sum(activity_matrix)
    total_possible_cells = activity_matrix.size
    print(f"Overall Spectrum Occupancy Rate: {total_active_cells / total_possible_cells:.2%}")

    print("\nBand-by-Band Activity Statistics:")
    for b_id in range(config.num_bands):
        band = env.bands[b_id]
        active_count = np.sum(activity_matrix[b_id, :])
        occupancy = active_count / config.total_time_steps
        print(f" Band {b_id} ({band.f_center_mhz:.0f} MHz): Active {active_count:3d}/{config.total_time_steps} steps ({occupancy:6.1%})")

    # 4. Print ASCII Spectrum Heatmap
    print("\n--- GROUND TRUTH TIME-FREQUENCY SPECTRUM MAP (ASCII) ---")
    print("Time -> 0         10        20        30        40        50        60        70        80        90        99")
    print("        |---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|")
    for b_id in range(config.num_bands - 1, -1, -1):
        band = env.bands[b_id]
        row_str = "".join(["#" if activity_matrix[b_id, t] == 1 else "." for t in range(100)])
        print(f"Band {b_id} ({band.f_center_mhz:.0f} MHz) |{row_str}|")

    # 5. Save Graphic Heatmap Plot
    os.makedirs("experiments/results", exist_ok=True)
    plot_path = "experiments/results/step1_spectrum_activity.png"
    
    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.imshow(activity_matrix, aspect='auto', origin='lower', cmap='YlGnBu', interpolation='nearest')
    ax.set_title("Step 1: Ground Truth RF Spectrum Activity Matrix (Bands vs Time)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time Step (t)", fontsize=12)
    ax.set_ylabel("Frequency Band ID", fontsize=12)
    ax.set_yticks(range(config.num_bands))
    ax.set_yticklabels([f"Band {b} ({env.bands[b].f_center_mhz:.0f} MHz)" for b in range(config.num_bands)])
    plt.colorbar(cax, label="Activity (1=Active, 0=Quiet)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\nGraphic heatmap plot saved to: {plot_path}")
    print("=" * 70)
    return activity_matrix

if __name__ == "__main__":
    run_step1_demo()
