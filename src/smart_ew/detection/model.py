"""
Signal Detection Model implementing threshold and SNR-dependent probabilistic detection.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DetectionResult:
    """Outcome of evaluating a receiver signal detection attempt."""
    detected: bool
    measured_power_dbm: float
    estimated_snr_db: float
    confidence: float
    is_false_alarm: bool


class DetectionModel:
    """
    Evaluates signal detection probability and noise characteristics
    based on signal-to-noise ratio (SNR) and detection thresholds.
    """

    def __init__(
        self,
        snr_threshold_db: float = 10.0,
        p_false_alarm: float = 0.01,
        noise_std_db: float = 1.0,
        probabilistic: bool = True,
        seed: Optional[int] = None,
    ):
        self.snr_threshold_db = snr_threshold_db
        self.p_false_alarm = p_false_alarm
        self.noise_std_db = noise_std_db
        self.probabilistic = probabilistic
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def compute_pd(self, snr_db: float) -> float:
        """
        Computes the theoretical probability of detection (Pd) as a function of SNR.
        Uses a sigmoid ROC approximation centered around snr_threshold_db.
        """
        if not self.probabilistic:
            return 1.0 if snr_db >= self.snr_threshold_db else 0.0
        # Sigmoid curve: Pd reaches 0.5 at threshold, >0.95 at threshold + 6dB
        k = 0.5
        return float(1.0 / (1.0 + np.exp(-k * (snr_db - self.snr_threshold_db))))

    def evaluate(
        self, true_power_dbm: float, noise_floor_dbm: float, is_truly_active: bool
    ) -> DetectionResult:
        """
        Evaluates detection on a given frequency band given true ground truth state.
        """
        noise = self.rng.normal(0.0, self.noise_std_db)

        if is_truly_active:
            measured_power = true_power_dbm + noise
            snr_db = measured_power - noise_floor_dbm
            p_d = self.compute_pd(snr_db)

            if self.probabilistic:
                detected = bool(self.rng.random() < p_d)
            else:
                detected = bool(snr_db >= self.snr_threshold_db)

            return DetectionResult(
                detected=detected,
                measured_power_dbm=float(measured_power),
                estimated_snr_db=float(max(0.0, snr_db)),
                confidence=float(p_d if detected else 1.0 - p_d),
                is_false_alarm=False,
            )
        else:
            measured_power = noise_floor_dbm + abs(noise)
            snr_db = max(0.0, measured_power - noise_floor_dbm)
            false_alarm = bool(self.rng.random() < self.p_false_alarm)

            return DetectionResult(
                detected=false_alarm,
                measured_power_dbm=float(measured_power),
                estimated_snr_db=float(snr_db),
                confidence=float(self.p_false_alarm if false_alarm else 1.0 - self.p_false_alarm),
                is_false_alarm=false_alarm,
            )

    def reset(self) -> None:
        """Resets random number generator state."""
        self.rng = np.random.default_rng(self.seed)
