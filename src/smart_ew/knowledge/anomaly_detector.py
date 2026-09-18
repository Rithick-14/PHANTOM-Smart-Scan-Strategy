"""
RF Spectrum Anomaly Detector.
Identifies unexpected emitter activations, statistical environment shifts, power spikes,
and agile behavior anomalies from receiver observations and knowledge map state.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np

from smart_ew.receiver.schema import ReceiverObservation
from smart_ew.knowledge.change_detector import PageHinkleyChangeDetector


@dataclass
class AnomalyRecord:
    """Represents a single evaluated RF observation anomaly status."""
    timestamp: int
    band_id: int
    is_anomaly: bool
    anomaly_category: str  # "NORMAL", "ENVIRONMENT_DRIFT", "UNEXPECTED_BURST", "AGILE_PATTERN_SHIFT", "POWER_SPIKE"
    surprise_score: float  # Residual / surprise metric (0.0 to 1.0)
    measured_power_dbm: float
    description: str
    action_triggered: str  # E.g., "Exploration Boost", "Knowledge Reset", "Logged"


class RFAnomalyDetector:
    """
    Evaluates streaming receiver observations to detect anomalies in RF spectrum behavior.
    Combines Page-Hinkley drift detection with surprise residual scoring and band history.
    """

    def __init__(
        self,
        num_bands: int,
        ph_threshold: float = 3.5,
        surprise_threshold: float = 0.65,
        quiet_burst_threshold: int = 12,
    ):
        self.num_bands = num_bands
        self.ph_threshold = ph_threshold
        self.surprise_threshold = surprise_threshold
        self.quiet_burst_threshold = quiet_burst_threshold

        self.change_detector = PageHinkleyChangeDetector(threshold=ph_threshold)
        self.last_seen_step: Dict[int, int] = {b: -1 for b in range(num_bands)}
        self.band_hit_counts: Dict[int, int] = {b: 0 for b in range(num_bands)}
        self.anomaly_records: List[AnomalyRecord] = []

    def evaluate_observation(
        self,
        obs: ReceiverObservation,
        predicted_prob: float = 0.5,
    ) -> AnomalyRecord:
        """
        Evaluates a single ReceiverObservation against historical expectations.
        Returns an AnomalyRecord detailing whether an anomaly occurred.
        """
        b_id = obs.band_id
        t = obs.timestamp
        is_hit = (obs.detected and not obs.is_false_alarm)
        target = 1 if is_hit else 0

        # Calculate residual surprise score
        surprise_score = float(abs(target - predicted_prob))

        # Check Page-Hinkley environment drift
        is_drift = self.change_detector.update(target, predicted_prob)

        is_anomaly = False
        category = "NORMAL"
        description = "Nominal observation"
        action = "Logged"

        if is_drift:
            is_anomaly = True
            category = "ENVIRONMENT_DRIFT"
            description = f"Page-Hinkley drift detected at step {t} on Band {b_id} (Residual: {surprise_score:.2f})"
            action = "Triggered Exploration Boost & Decayed Knowledge"

        elif is_hit and predicted_prob < 0.25:
            # Signal appeared on a band expected to be quiet
            time_since_last = t - self.last_seen_step[b_id] if self.last_seen_step[b_id] != -1 else t
            if time_since_last >= self.quiet_burst_threshold or self.band_hit_counts[b_id] <= 1:
                is_anomaly = True
                category = "UNEXPECTED_BURST"
                description = f"Unexpected transmission on quiet Band {b_id} (Prior P={predicted_prob:.2f}, Silence={time_since_last} steps)"
                action = "Updated Prior & Raised Priority"

        elif is_hit and obs.estimated_snr_db > 22.0:
            is_anomaly = True
            category = "POWER_SPIKE"
            description = f"Abnormally high signal SNR ({obs.estimated_snr_db:.1f} dB) detected on Band {b_id}"
            action = "Flagged High-Power Threat"

        elif is_hit and self.band_hit_counts[b_id] == 0 and t > 15:
            is_anomaly = True
            category = "AGILE_PATTERN_SHIFT"
            description = f"First-time signal burst on previously inactive Band {b_id} at step {t}"
            action = "Registered New Emitter Pattern"

        # Update tracking state
        if is_hit:
            self.last_seen_step[b_id] = t
            self.band_hit_counts[b_id] += 1

        record = AnomalyRecord(
            timestamp=t,
            band_id=b_id,
            is_anomaly=is_anomaly,
            anomaly_category=category,
            surprise_score=surprise_score,
            measured_power_dbm=obs.measured_power_dbm,
            description=description,
            action_triggered=action,
        )
        self.anomaly_records.append(record)
        return record

    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Returns aggregate metrics on detected anomalies."""
        total_evals = len(self.anomaly_records)
        anomalies = [r for r in self.anomaly_records if r.is_anomaly]
        total_anomalies = len(anomalies)

        categories = {}
        for r in anomalies:
            categories[r.anomaly_category] = categories.get(r.anomaly_category, 0) + 1

        return {
            "total_evaluations": total_evals,
            "total_anomalies": total_anomalies,
            "anomaly_rate": float(total_anomalies / total_evals) if total_evals > 0 else 0.0,
            "category_breakdown": categories,
        }

    def reset(self) -> None:
        """Resets detector state."""
        self.change_detector.reset()
        self.last_seen_step = {b: -1 for b in range(self.num_bands)}
        self.band_hit_counts = {b: 0 for b in range(self.num_bands)}
        self.anomaly_records.clear()
