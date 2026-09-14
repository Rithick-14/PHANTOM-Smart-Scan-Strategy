"""
Environment Change Detector using Page-Hinkley test for monitoring prediction residuals.
"""

from typing import Optional


class PageHinkleyChangeDetector:
    """
    Page-Hinkley statistical change-point detector.
    Monitors prediction residual errors e_t = |target - predicted_prob|
    and detects non-stationary drift in the RF spectrum environment.
    """

    def __init__(
        self,
        delta: float = 0.05,
        threshold: float = 4.0,
        alpha_decay: float = 0.99,
    ):
        self.delta = delta
        self.threshold = threshold
        self.alpha_decay = alpha_decay

        self.mean_error: float = 0.0
        self.cum_sum: float = 0.0
        self.min_cum_sum: float = 0.0
        self.sample_count: int = 0
        self.change_count: int = 0

    def update(self, target: int, predicted_prob: float) -> bool:
        """
        Updates detector with new observed target (1 or 0) and predicted probability.
        Returns True if a statistical environment change is detected.
        """
        error = abs(float(target) - float(predicted_prob))
        self.sample_count += 1

        # Exponentially weighted mean error update
        if self.sample_count == 1:
            self.mean_error = error
        else:
            self.mean_error = self.alpha_decay * self.mean_error + (1.0 - self.alpha_decay) * error

        # Page-Hinkley cumulative sum update
        self.cum_sum += error - self.mean_error - self.delta
        if self.cum_sum < self.min_cum_sum:
            self.min_cum_sum = self.cum_sum

        ph_statistic = self.cum_sum - self.min_cum_sum

        if ph_statistic > self.threshold:
            self.change_count += 1
            self.reset_sums()
            return True

        return False

    def reset_sums(self) -> None:
        """Resets cumulative sum counters while keeping change count statistics."""
        self.cum_sum = 0.0
        self.min_cum_sum = 0.0

    def reset(self) -> None:
        """Fully resets detector state."""
        self.mean_error = 0.0
        self.cum_sum = 0.0
        self.min_cum_sum = 0.0
        self.sample_count = 0
        self.change_count = 0
