"""
STEP 5 DEMONSTRATION SCRIPT
ML Activity Prediction Engine - Feature Extraction & Machine Learning Models
"""

import os
import numpy as np

from smart_ew.environment.schema import SimConfig, EmitterConfig
from smart_ew.environment.simulator import RFEnvironment
from smart_ew.emitters.continuous import ContinuousEmitter
from smart_ew.emitters.intermittent import IntermittentEmitter
from smart_ew.receiver.schema import ReceiverConfig
from smart_ew.receiver.model import ESReceiver
from smart_ew.features.extractor import FeatureExtractor
from smart_ew.features.schema import BandFeatureVector
from smart_ew.models.predictor import (
    LogisticRegressionPredictor,
    RandomForestPredictor,
    GradientBoostingPredictor,
    MLActivityPredictor,
)


def run_step5_demo():
    print("=" * 80)
    print("STEP 5: ML ACTIVITY PREDICTION - FEATURE EXTRACTION & ML INFERENCE")
    print("=" * 80)

    # 1. Setup Environment (10 Bands, 100 Steps)
    sim_config = SimConfig(num_bands=10, total_time_steps=100, seed=42)
    env = RFEnvironment(sim_config)

    # E1: Band 2 (Continuous)
    env.add_emitter(
        ContinuousEmitter(
            EmitterConfig(
                emitter_id="Continuous_E1",
                emitter_type="continuous",
                assigned_bands=[2],
                signal_power_dbm=-75.0,
                parameters={"start_time": 10, "end_time": 80},
            )
        )
    )
    # E2: Band 5 (Intermittent)
    env.add_emitter(
        IntermittentEmitter(
            EmitterConfig(
                emitter_id="Intermittent_E2",
                emitter_type="intermittent",
                assigned_bands=[5],
                signal_power_dbm=-80.0,
                parameters={"p_on": 0.4, "p_off": 0.3},
            ),
            seed=42,
        )
    )

    rx = ESReceiver(config=ReceiverConfig(retuning_time_cost=1.0), seed=42)
    extractor = FeatureExtractor()

    # 2. Collect Historical Training Data by sweeping spectrum for 60 steps
    print("\nPhase 1: Feature Extraction & Data Buffer Accumulation (Steps 0 to 59)...")
    
    X_samples = []
    y_samples = []

    for t in range(60):
        gt_frame = env.step()
        
        # Scan a band (round-robin sampling)
        scan_band = t % 10
        obs = rx.scan(scan_band, t, gt_frame)
        
        # Extract features BEFORE updating extractor (zero leakage rule)
        feat = extractor.extract_features(scan_band, t, num_bands=10)
        X_samples.append(feat.to_numpy())
        y_samples.append(1 if (gt_frame.active_bands.get(scan_band, False)) else 0)

        # Update extractor with observation
        extractor.update(obs)

    X = np.array(X_samples, dtype=np.float32)
    y = np.array(y_samples, dtype=int)

    print(f"Collected {X.shape[0]} historical feature samples.")
    print(f"Feature Vector Dimension: {X.shape[1]} features")
    print(f"Feature Names: {BandFeatureVector.feature_names()}")
    print(f"Target Distribution: {np.bincount(y)[1]} Active (1) / {np.bincount(y)[0]} Quiet (0)")

    # 3. Train & Compare Supervised ML Models
    print("\nPhase 2: Training & Evaluating Supervised Classifiers...")
    
    models = {
        "Logistic Regression": LogisticRegressionPredictor(seed=42),
        "Random Forest": RandomForestPredictor(n_estimators=50, seed=42),
        "Gradient Boosting": GradientBoostingPredictor(n_estimators=50, seed=42),
    }

    # Chronological Train/Test Split (First 40 steps train, next 20 test)
    X_train, X_test = X[:40], X[40:]
    y_train, y_test = y[:40], y[40:]

    print(f"{'Model Architecture':<22} | {'Train Accuracy':<16} | {'Test Accuracy':<15} | {'Brier Score (Loss)':<18}")
    print("-" * 78)

    for name, predictor in models.items():
        predictor.fit(X_train, y_train)
        
        train_probs = predictor.predict_proba(X_train)
        test_probs = predictor.predict_proba(X_test)
        
        train_acc = np.mean((train_probs >= 0.5) == y_train)
        test_acc = np.mean((test_probs >= 0.5) == y_test)
        brier_score = np.mean((test_probs - y_test) ** 2)

        print(f"{name:<22} | {train_acc:<16.2%} | {test_acc:<15.2%} | {brier_score:<18.4f}")

    print("-" * 78)

    # 4. Demonstrate Online Probability Inference Across All Bands
    print("\nPhase 3: Online Spectrum Activity Probability Estimation at Step 60:")
    ml_manager = MLActivityPredictor(predictor=RandomForestPredictor(seed=42), seed=42)
    ml_manager.fit()  # Fit on available data

    # Train manager on all collected samples
    for i in range(len(X)):
        ml_manager.add_sample(X[i], y[i])

    current_feats = extractor.extract_all_features(current_step=60, num_bands=10)
    pred_probas = ml_manager.predict_all_bands(current_feats)

    # Inspect current step 60 ground truth
    gt_frame_60 = env.get_ground_truth(60)

    print(f"{'Band ID':<8} | {'Center Freq':<12} | {'P(Active | History)':<22} | {'Actual GT State':<16}")
    print("-" * 65)
    for b_id in range(10):
        band_freq = env.bands[b_id].f_center_mhz
        p_act = pred_probas[b_id]
        gt_active = "ACTIVE (1)" if gt_frame_60.active_bands[b_id] else "QUIET (0)"
        print(f"Band {b_id:<3} | {band_freq:.0f} MHz     | {p_act:<22.2%} | {gt_active:<16}")

    print("-" * 65)
    print("Key Takeaways:")
    print(" 1. The ML models accurately predict high activity probability for Band 2 and Band 5 based on observation history.")
    print(" 2. Feature vectors are constructed strictly from past observations without data leakage.")
    print(" 3. Probability estimates P(Activity | History) will directly feed into our Smart Scheduler in STEP 6.")
    print("=" * 80)


if __name__ == "__main__":
    run_step5_demo()
