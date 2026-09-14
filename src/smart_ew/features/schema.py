"""
Data structure for per-band feature vectors used by ML predictors and schedulers.
"""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class BandFeatureVector:
    """Numerical feature representation for a single frequency band at a specific time step."""
    band_id: int
    frequency_norm: float
    total_scans: int
    hit_count: int
    miss_count: int
    time_since_last_scan: float
    time_since_last_hit: float
    recent_hit_rate_k5: float
    recent_hit_rate_k20: float
    overall_hit_rate: float
    activity_trend: float

    def to_numpy(self) -> np.ndarray:
        """Converts feature vector to 1D float32 numpy array for ML model consumption."""
        return np.array(
            [
                self.frequency_norm,
                float(self.total_scans),
                float(self.hit_count),
                float(self.miss_count),
                self.time_since_last_scan,
                self.time_since_last_hit,
                self.recent_hit_rate_k5,
                self.recent_hit_rate_k20,
                self.overall_hit_rate,
                self.activity_trend,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def feature_names() -> List[str]:
        """Returns ordered list of feature names matching to_numpy()."""
        return [
            "frequency_norm",
            "total_scans",
            "hit_count",
            "miss_count",
            "time_since_last_scan",
            "time_since_last_hit",
            "recent_hit_rate_k5",
            "recent_hit_rate_k20",
            "overall_hit_rate",
            "activity_trend",
        ]
