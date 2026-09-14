"""
Dynamic RF Knowledge Map maintaining Bayesian belief, confidence, uncertainty, and transition statistics.
"""

from typing import Dict, Optional
import numpy as np

from smart_ew.receiver.schema import ReceiverObservation
from smart_ew.knowledge.schema import BandKnowledge


class DynamicRFKnowledgeMap:
    """
    Maintains dynamic per-band knowledge state (activity probability, confidence,
    uncertainty, recency, and frequency transition behavior).
    """

    def __init__(
        self,
        num_bands: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        self.num_bands = num_bands
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.last_active_band: Optional[int] = None
        self.bands: Dict[int, BandKnowledge] = self._init_bands()

    def _init_bands(self) -> Dict[int, BandKnowledge]:
        return {
            b: BandKnowledge(
                band_id=b,
                beta_alpha=self.prior_alpha,
                beta_beta=self.prior_beta,
                uncertainty=self._compute_variance(self.prior_alpha, self.prior_beta),
            )
            for b in range(self.num_bands)
        }

    @staticmethod
    def _compute_variance(alpha: float, beta: float) -> float:
        total = alpha + beta
        var = (alpha * beta) / ((total**2) * (total + 1.0))
        return float(np.sqrt(var))

    def update(
        self, observation: ReceiverObservation, predicted_prob: Optional[float] = None
    ) -> None:
        """Updates knowledge state for the observed band."""
        b_id = observation.band_id
        if b_id not in self.bands:
            return

        bk = self.bands[b_id]
        bk.total_observations += 1
        bk.last_scan_time = observation.timestamp

        is_hit = observation.detected and not observation.is_false_alarm

        if is_hit:
            bk.hits += 1
            bk.last_hit_time = observation.timestamp
            bk.beta_alpha += 1.0

            # Inter-band transition statistics
            if self.last_active_band is not None and self.last_active_band != b_id:
                prev_bk = self.bands[self.last_active_band]
                prev_bk.transition_counts[b_id] = (
                    prev_bk.transition_counts.get(b_id, 0.0) + 1.0
                )

            self.last_active_band = b_id
        else:
            bk.misses += 1
            bk.beta_beta += 1.0

        # Bayesian belief mean & uncertainty
        bayesian_prob = bk.beta_alpha / (bk.beta_alpha + bk.beta_beta)
        if predicted_prob is not None:
            bk.activity_probability = float(0.5 * bayesian_prob + 0.5 * predicted_prob)
        else:
            bk.activity_probability = float(bayesian_prob)

        bk.uncertainty = self._compute_variance(bk.beta_alpha, bk.beta_beta)
        # Confidence increases asymptotically with total observations
        bk.confidence = float(1.0 - (1.0 / np.sqrt(bk.total_observations + 1.0)))

    def get_band_knowledge(self, band_id: int, current_step: int) -> BandKnowledge:
        """Retrieves knowledge state for band_id updated with recency deltas."""
        if band_id not in self.bands:
            return BandKnowledge(band_id=band_id)

        bk = self.bands[band_id]

        if bk.last_scan_time >= 0:
            bk.time_since_last_scan = float(current_step - bk.last_scan_time)
        else:
            bk.time_since_last_scan = float(current_step)

        if bk.last_hit_time >= 0:
            bk.time_since_last_hit = float(current_step - bk.last_hit_time)
        else:
            bk.time_since_last_hit = float(current_step + 100)

        return bk

    def get_all_knowledge(self, current_step: int) -> Dict[int, BandKnowledge]:
        """Retrieves knowledge state for all bands."""
        return {b: self.get_band_knowledge(b, current_step) for b in range(self.num_bands)}

    def get_transition_probability(self, from_band: int, to_band: int) -> float:
        """Returns empirical transition probability P(to_band | from_band)."""
        if from_band not in self.bands:
            return 0.0
        counts = self.bands[from_band].transition_counts
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return float(counts.get(to_band, 0.0) / total)

    def decay_stale_knowledge(self, decay_factor: float = 0.95) -> None:
        """Applies memory decay to Bayesian parameters to handle environmental non-stationarity."""
        for bk in self.bands.values():
            bk.beta_alpha = max(self.prior_alpha, bk.beta_alpha * decay_factor)
            bk.beta_beta = max(self.prior_beta, bk.beta_beta * decay_factor)
            bk.uncertainty = self._compute_variance(bk.beta_alpha, bk.beta_beta)

    def reset(self) -> None:
        """Resets knowledge map state."""
        self.last_active_band = None
        self.bands = self._init_bands()
