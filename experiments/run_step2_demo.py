"""
STEP 2 DEMONSTRATION SCRIPT
ES Receiver Simulator - Action-Driven Band Scanning and Observation Modeling
"""

import os
import numpy as np

from smart_ew.environment.schema import SimConfig, EmitterConfig
from smart_ew.environment.simulator import RFEnvironment
from smart_ew.emitters.continuous import ContinuousEmitter
from smart_ew.emitters.intermittent import IntermittentEmitter
from smart_ew.receiver.schema import ReceiverConfig
from smart_ew.receiver.model import ESReceiver


def run_step2_demo():
    print("=" * 75)
    print("STEP 2: RECEIVER SIMULATOR - OBSERVATION & RETUNING COST DEMO")
    print("=" * 75)

    # 1. Setup RF Environment (10 Bands, Ground Truth)
    sim_config = SimConfig(num_bands=10, f_start_mhz=1000.0, f_step_mhz=50.0, seed=42)
    env = RFEnvironment(sim_config)

    # Add Emitters:
    # E1: High SNR signal on Band 2 (1100 MHz)
    # E2: Low SNR / Weak signal on Band 6 (1300 MHz)
    env.add_emitter(
        ContinuousEmitter(
            EmitterConfig(
                emitter_id="STRONG_RADAR_E1",
                emitter_type="continuous",
                assigned_bands=[2],
                signal_power_dbm=-75.0,  # +25 dB SNR over -100 dBm noise floor
            )
        )
    )
    env.add_emitter(
        IntermittentEmitter(
            EmitterConfig(
                emitter_id="WEAK_COMM_E2",
                emitter_type="intermittent",
                assigned_bands=[6],
                signal_power_dbm=-92.0,  # +8 dB SNR over -100 dBm noise floor (near threshold)
                parameters={"p_on": 0.4, "p_off": 0.4},
            ),
            seed=42,
        )
    )

    # 2. Setup ES Receiver Model
    rx_config = ReceiverConfig(
        instantaneous_bw_mhz=50.0,
        retuning_time_cost=1.5,
        noise_floor_dbm=-100.0,
        snr_threshold_db=10.0,
        p_false_alarm=0.02,
        probabilistic_detection=True,
        noise_std_db=1.0,
    )
    receiver = ESReceiver(config=rx_config, seed=42)

    print(f"Receiver Configuration:")
    print(f" - Instantaneous BW: {rx_config.instantaneous_bw_mhz} MHz")
    print(f" - Retuning Time Cost: {rx_config.retuning_time_cost} time units")
    print(f" - Noise Floor: {rx_config.noise_floor_dbm} dBm")
    print(f" - SNR Detection Threshold: {rx_config.snr_threshold_db} dB")
    print(f" - False Alarm Probability (Pfa): {rx_config.p_false_alarm:.2%}")
    print()

    # 3. Simulate Receiver Scanning Actions across 15 time steps
    # We will simulate scanning different bands to demonstrate retuning costs,
    # hits, misses, and false alarms.
    scan_plan = [2, 2, 2, 6, 6, 6, 0, 0, 2, 6, 9, 9, 2, 6, 2]

    print(f"{'Step':<5} | {'Target Band':<11} | {'Ground Truth':<14} | {'Detected?':<10} | {'Meas Power':<12} | {'SNR (dB)':<9} | {'Retune Cost':<11} | {'Status':<15}")
    print("-" * 105)

    total_tuning_cost = 0.0
    detections = 0
    false_alarms = 0

    for t in range(len(scan_plan)):
        target_band = scan_plan[t]
        frame = env.step()
        
        # Execute scan action - MUST ONLY SEE CURRENT BAND THROUGH OBSERVATION
        obs = receiver.scan(band_id=target_band, time_step=t, ground_truth_frame=frame, observation_duration=1.0)
        
        ground_truth_active = frame.active_bands[target_band]
        total_tuning_cost += obs.tuning_cost_paid

        if obs.detected:
            if obs.is_false_alarm:
                false_alarms += 1
                status = "FALSE ALARM"
            else:
                detections += 1
                status = "HIT (DETECTION)"
        else:
            status = "MISS" if ground_truth_active else "QUIET (CORRECT)"

        gt_str = "ACTIVE" if ground_truth_active else "QUIET"
        det_str = "YES" if obs.detected else "NO"
        
        print(f"{t:<5} | Band {target_band:<6} | {gt_str:<14} | {det_str:<10} | {obs.measured_power_dbm:<12.1f} | {obs.estimated_snr_db:<9.1f} | {obs.tuning_cost_paid:<11.1f} | {status:<15}")

    print("-" * 105)
    print("\n--- RECEIVER PERFORMANCE SUMMARY ---")
    print(f"Total Time Steps Simulated: {len(scan_plan)}")
    print(f"Total Successful Detections: {detections}")
    print(f"Total False Alarms Incurred: {false_alarms}")
    print(f"Total Retuning Overhead Paid: {total_tuning_cost:.1f} time units")
    print("=" * 75)

if __name__ == "__main__":
    run_step2_demo()
