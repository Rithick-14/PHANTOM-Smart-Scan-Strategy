"""
Feature Extractor generating online contextual features from receiver observation history.
"""

from typing import Dict, List
import numpy as np

from smart_ew.receiver.schema import ReceiverObservation
from smart_ew.features.schema import BandFeatureVector


class FeatureExtractor:
    """
    Maintains per-band receiver observation history and constructs
    contextual feature vectors without data leakage.
    """

    def __init__(self):
        # Maps band_id -> List of ReceiverObservation
        self._history: Dict[int, List[ReceiverObservation]] = {}
        # Maps band_id -> step of last hit
        self._last_hit_step: Dict[int, int] = {}
        # Maps band_id -> step of last scan
        self._last_scan_step: Dict[int, int] = {}

    def update(self, observation: ReceiverObservation) -> None:
        """Appends a new receiver observation to history."""
        band_id = observation.band_id
        if band_id not in self._history:
            self._history[band_id] = []

        self._history[band_id].append(observation)
        self._last_scan_step[band_id] = observation.timestamp

        if observation.detected:
            self._last_hit_step[band_id] = observation.timestamp

    def extract_features(
        self, band_id: int, current_step: int, num_bands: int
    ) -> BandFeatureVector:
        """
        Computes feature vector for band_id at current_step based solely on observation history.
        """
        freq_norm = float(band_id / max(1, num_bands - 1))
        obs_list = self._history.get(band_id, [])
        total_scans = len(obs_list)

        if total_scans == 0:
            return BandFeatureVector(
                band_id=band_id,
                frequency_norm=freq_norm,
                total_scans=0,
                hit_count=0,
                miss_count=0,
                time_since_last_scan=float(current_step),
                time_since_last_hit=float(current_step + 100),
                recent_hit_rate_k5=0.0,
                recent_hit_rate_k20=0.0,
                overall_hit_rate=0.0,
                activity_trend=0.0,
            )

        hits = [1 if o.detected else 0 for o in obs_list]
        hit_count = sum(hits)
        miss_count = total_scans - hit_count

        last_scan = self._last_scan_step.get(band_id, 0)
        time_since_scan = float(current_step - last_scan)

        if band_id in self._last_hit_step:
            time_since_hit = float(current_step - self._last_hit_step[band_id])
        else:
            time_since_hit = float(current_step + 100)

        # Sliding window rates
        k5_hits = hits[-5:]
        k20_hits = hits[-20:]

        rate_k5 = float(np.mean(k5_hits)) if k5_hits else 0.0
        rate_k20 = float(np.mean(k20_hits)) if k20_hits else 0.0
        overall_rate = float(hit_count / total_scans)
        trend = float(rate_k5 - rate_k20)

        return BandFeatureVector(
            band_id=band_id,
            frequency_norm=freq_norm,
            total_scans=total_scans,
            hit_count=hit_count,
            miss_count=miss_count,
            time_since_last_scan=time_since_scan,
            time_since_last_hit=time_since_hit,
            recent_hit_rate_k5=rate_k5,
            recent_hit_rate_k20=rate_k20,
            overall_hit_rate=overall_rate,
            activity_trend=trend,
        )

    def extract_all_features(
        self, current_step: int, num_bands: int
    ) -> Dict[int, BandFeatureVector]:
        """Extracts feature vectors for all frequency bands (0 to num_bands - 1)."""
        return {
            b: self.extract_features(b, current_step, num_bands) for b in range(num_bands)
        }

    def reset(self) -> None:
        """Resets all history."""
        self._history.clear()
        self._last_hit_step.clear()
        self._last_scan_step.clear()
