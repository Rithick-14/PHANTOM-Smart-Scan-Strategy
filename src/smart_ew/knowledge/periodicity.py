"""
Periodicity Estimator for discovering recurring transmission cycles from sparse intercept timestamps.
"""

from typing import List, Optional
from collections import Counter
import math


class PeriodicityEstimator:
    """
    Estimates cycle periods of periodic emitters from sparse intercept timestamps.
    """

    def estimate_period(self, intercept_timestamps: List[int]) -> Optional[int]:
        """
        Estimates transmission period P from a list of intercept timestamps.
        """
        if len(intercept_timestamps) < 3:
            return None

        # Compute adjacent timestamp differences
        diffs = [
            intercept_timestamps[i + 1] - intercept_timestamps[i]
            for i in range(len(intercept_timestamps) - 1)
        ]

        positive_diffs = [d for d in diffs if d > 0]
        if not positive_diffs:
            return None

        # Find mode of differences
        counts = Counter(positive_diffs)
        most_common_diff, count = counts.most_common(1)[0]

        if count >= 2 or len(positive_diffs) <= 4:
            return int(most_common_diff)

        # Fallback: compute Greatest Common Divisor (GCD) of differences
        gcd_val = positive_diffs[0]
        for d in positive_diffs[1:]:
            gcd_val = math.gcd(gcd_val, d)

        if gcd_val >= 2:
            return int(gcd_val)

        return int(most_common_diff)

    def predict_activity_window(
        self, intercept_timestamps: List[int], target_step: int
    ) -> float:
        """
        Predicts activity probability at target_step based on estimated periodicity.
        """
        est_period = self.estimate_period(intercept_timestamps)
        if est_period is None or len(intercept_timestamps) == 0:
            return 0.5

        last_hit = intercept_timestamps[-1]
        delta = target_step - last_hit

        if delta < 0:
            return 0.5

        phase = delta % est_period
        if phase == 0:
            return 0.95
        elif phase == 1:
            return 0.70
        else:
            return 0.10
