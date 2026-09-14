"""
Contextual Multi-Armed Bandit Schedulers (LinUCB and Contextual Thompson Sampling).
"""

from typing import Optional, Dict
import numpy as np

from smart_ew.scheduling.base import BaseScheduler
from smart_ew.features.extractor import FeatureExtractor
from smart_ew.receiver.schema import ReceiverObservation


class LinUCBScheduler(BaseScheduler):
    """
    LinUCB (Disjoint Linear Upper Confidence Bound) Contextual Bandit Scheduler.
    Models reward for each frequency band as a linear function of contextual feature vector X_{b,t}
    with explicit upper-confidence bound exploration.
    """

    def __init__(
        self,
        num_bands: int,
        alpha: float = 1.0,
        feature_dim: int = 10,
        seed: Optional[int] = None,
        name: str = "LinUCB Contextual Bandit",
    ):
        super().__init__(num_bands, name)
        self.alpha = alpha
        self.feature_dim = feature_dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.feature_extractor = FeatureExtractor()

        # LinUCB parameters per band:
        # A_b: (d, d) matrix, initialized to Identity
        # b_b: (d,) vector, initialized to Zeros
        self.A: Dict[int, np.ndarray] = {
            b: np.eye(feature_dim, dtype=np.float32) for b in range(num_bands)
        }
        self.b_vec: Dict[int, np.ndarray] = {
            b: np.zeros(feature_dim, dtype=np.float32) for b in range(num_bands)
        }

        # Store feature vector extracted at action decision step
        self._last_context: Dict[int, np.ndarray] = {}

    def select_next_band(self, time_step: int) -> int:
        features = self.feature_extractor.extract_all_features(time_step, self.num_bands)
        p_ucb: Dict[int, float] = {}

        for b in range(self.num_bands):
            x = features[b].to_numpy()
            self._last_context[b] = x

            A_inv = np.linalg.inv(self.A[b])
            theta_hat = A_inv @ self.b_vec[b]

            # Linear expected payoff + UCB exploration bound
            expected_payoff = float(theta_hat @ x)
            ucb_bound = float(self.alpha * np.sqrt(x.T @ A_inv @ x))
            p_ucb[b] = expected_payoff + ucb_bound

        # Select band with maximum UCB score (break ties randomly)
        max_score = max(p_ucb.values())
        best_bands = [b for b, score in p_ucb.items() if np.isclose(score, max_score)]
        return int(self.rng.choice(best_bands))

    def update(self, observation: ReceiverObservation) -> None:
        b_id = observation.band_id
        x = self._last_context.get(
            b_id,
            self.feature_extractor.extract_features(
                b_id, observation.timestamp, self.num_bands
            ).to_numpy(),
        )

        self.feature_extractor.update(observation)

        # Compute scalar reward:
        # +1.0 for true positive detection, -0.5 for false alarm, 0.0 for quiet miss, minus retuning penalty
        if observation.detected and not observation.is_false_alarm:
            reward = 1.0
        elif observation.is_false_alarm:
            reward = -0.5
        else:
            reward = 0.0

        reward -= 0.1 * observation.tuning_cost_paid

        # Ridge regression update
        self.A[b_id] += np.outer(x, x)
        self.b_vec[b_id] += reward * x

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.feature_extractor.reset()
        self._last_context.clear()
        self.A = {b: np.eye(self.feature_dim, dtype=np.float32) for b in range(self.num_bands)}
        self.b_vec = {b: np.zeros(self.feature_dim, dtype=np.float32) for b in range(self.num_bands)}


class ContextualThompsonSamplingScheduler(BaseScheduler):
    """
    Contextual Thompson Sampling Scheduler.
    Samples weight vector theta_b ~ N(mu_b, v^2 * A_b^-1) at each step to select band.
    """

    def __init__(
        self,
        num_bands: int,
        v_sq: float = 0.25,
        feature_dim: int = 10,
        seed: Optional[int] = None,
        name: str = "Contextual Thompson Sampling",
    ):
        super().__init__(num_bands, name)
        self.v_sq = v_sq
        self.feature_dim = feature_dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.feature_extractor = FeatureExtractor()
        self.B: Dict[int, np.ndarray] = {
            b: np.eye(feature_dim, dtype=np.float32) for b in range(num_bands)
        }
        self.f_vec: Dict[int, np.ndarray] = {
            b: np.zeros(feature_dim, dtype=np.float32) for b in range(num_bands)
        }
        self._last_context: Dict[int, np.ndarray] = {}

    def select_next_band(self, time_step: int) -> int:
        features = self.feature_extractor.extract_all_features(time_step, self.num_bands)
        sampled_payoffs: Dict[int, float] = {}

        for b in range(self.num_bands):
            x = features[b].to_numpy()
            self._last_context[b] = x

            B_inv = np.linalg.inv(self.B[b])
            mu_b = B_inv @ self.f_vec[b]
            cov_b = self.v_sq * B_inv

            # Sample weight vector theta from posterior Gaussian distribution
            sampled_theta = self.rng.multivariate_normal(mu_b, cov_b)
            sampled_payoffs[b] = float(sampled_theta @ x)

        best_band = max(sampled_payoffs.keys(), key=lambda b: sampled_payoffs[b])
        return best_band

    def update(self, observation: ReceiverObservation) -> None:
        b_id = observation.band_id
        x = self._last_context.get(
            b_id,
            self.feature_extractor.extract_features(
                b_id, observation.timestamp, self.num_bands
            ).to_numpy(),
        )

        self.feature_extractor.update(observation)

        reward = 1.0 if (observation.detected and not observation.is_false_alarm) else 0.0
        if observation.is_false_alarm:
            reward = -0.5
        reward -= 0.1 * observation.tuning_cost_paid

        self.B[b_id] += np.outer(x, x)
        self.f_vec[b_id] += reward * x

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.feature_extractor.reset()
        self._last_context.clear()
        self.B = {b: np.eye(self.feature_dim, dtype=np.float32) for b in range(self.num_bands)}
        self.f_vec = {b: np.zeros(self.feature_dim, dtype=np.float32) for b in range(self.num_bands)}
