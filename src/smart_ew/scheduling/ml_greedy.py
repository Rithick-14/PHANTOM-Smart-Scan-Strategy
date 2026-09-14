"""
ML-Greedy baseline scan scheduler.
"""

from typing import Optional
import numpy as np

from smart_ew.scheduling.base import BaseScheduler
from smart_ew.features.extractor import FeatureExtractor
from smart_ew.models.predictor import MLActivityPredictor
from smart_ew.receiver.schema import ReceiverObservation


class MLGreedyScheduler(BaseScheduler):
    """
    ML-Greedy baseline scheduler.
    Uses MLActivityPredictor to estimate activity probability P(active | X_{b,t})
    and selects the band with highest predicted probability (with small epsilon exploration).
    """

    def __init__(
        self,
        num_bands: int,
        predictor: Optional[MLActivityPredictor] = None,
        epsilon: float = 0.1,
        seed: Optional[int] = None,
        name: str = "ML-Greedy Baseline",
    ):
        super().__init__(num_bands, name)
        self.seed = seed
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.feature_extractor = FeatureExtractor()
        self.predictor = predictor or MLActivityPredictor(seed=seed)

    def select_next_band(self, time_step: int) -> int:
        features = self.feature_extractor.extract_all_features(time_step, self.num_bands)
        probas = self.predictor.predict_all_bands(features)

        # Epsilon-greedy selection
        if self.rng.random() < self.epsilon or not probas:
            return int(self.rng.integers(0, self.num_bands))

        # Select band with maximum predicted probability
        best_band = max(probas.keys(), key=lambda b: (probas[b], self.rng.random()))
        return best_band

    def update(self, observation: ReceiverObservation) -> None:
        # Extract features for observed band before updating history
        feat = self.feature_extractor.extract_features(
            observation.band_id, observation.timestamp, self.num_bands
        )
        self.feature_extractor.update(observation)

        target = 1 if (observation.detected and not observation.is_false_alarm) else 0
        self.predictor.add_sample(feat.to_numpy(), target)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.feature_extractor.reset()
        self.predictor.reset()
