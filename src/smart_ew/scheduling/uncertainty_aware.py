"""
Uncertainty-Aware Exploration Scheduler integrating Bayesian Knowledge Map state into Contextual Bandits.
"""

from typing import Optional, Dict
import numpy as np

from smart_ew.scheduling.base import BaseScheduler
from smart_ew.scheduling.contextual_bandit import LinUCBScheduler
from smart_ew.knowledge.map import DynamicRFKnowledgeMap
from smart_ew.receiver.schema import ReceiverObservation


class UncertaintyAwareScheduler(BaseScheduler):
    """
    Uncertainty-Aware Contextual Scheduler.
    Fuses LinUCB activity predictions with Bayesian Knowledge Map uncertainty (U_b)
    and recency deltas (time since last scan) to prevent premature convergence
    and guarantee spectrum exploration.
    """

    def __init__(
        self,
        num_bands: int,
        alpha_linucb: float = 0.5,
        beta_uncertainty: float = 0.5,
        gamma_recency: float = 0.2,
        seed: Optional[int] = None,
        name: str = "Uncertainty-Aware Scheduler",
    ):
        super().__init__(num_bands, name)
        self.alpha_linucb = alpha_linucb
        self.beta_uncertainty = beta_uncertainty
        self.gamma_recency = gamma_recency
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.linucb_bandit = LinUCBScheduler(
            num_bands=num_bands, alpha=alpha_linucb, seed=seed
        )
        self.knowledge_map = DynamicRFKnowledgeMap(num_bands=num_bands)

    def select_next_band(self, time_step: int) -> int:
        features = self.linucb_bandit.feature_extractor.extract_all_features(
            time_step, self.num_bands
        )
        scores: Dict[int, float] = {}

        for b in range(self.num_bands):
            x = features[b].to_numpy()
            self.linucb_bandit._last_context[b] = x

            # LinUCB payoff & variance bound
            A_inv = np.linalg.inv(self.linucb_bandit.A[b])
            theta_hat = A_inv @ self.linucb_bandit.b_vec[b]

            linucb_pred = float(theta_hat @ x)
            linucb_var = float(self.alpha_linucb * np.sqrt(x.T @ A_inv @ x))

            # Knowledge map Bayesian uncertainty & recency
            bk = self.knowledge_map.get_band_knowledge(b, time_step)
            unc_score = bk.uncertainty
            recency_norm = min(1.0, bk.time_since_last_scan / 50.0)

            # Combined multi-factor decision score Q(b)
            q_score = (
                linucb_pred
                + linucb_var
                + self.beta_uncertainty * unc_score
                + self.gamma_recency * recency_norm
            )
            scores[b] = float(q_score)

        # Select band maximizing Q(b)
        max_score = max(scores.values())
        best_bands = [b for b, s in scores.items() if np.isclose(s, max_score)]
        return int(self.rng.choice(best_bands))

    def update(self, observation: ReceiverObservation) -> None:
        # Get ML predicted probability before update for knowledge map calibration
        b_id = observation.band_id
        features = self.linucb_bandit.feature_extractor.extract_features(
            b_id, observation.timestamp, self.num_bands
        )
        x = features.to_numpy()

        A_inv = np.linalg.inv(self.linucb_bandit.A[b_id])
        theta_hat = A_inv @ self.linucb_bandit.b_vec[b_id]
        pred_prob = float(1.0 / (1.0 + np.exp(-theta_hat @ x)))

        # Update both LinUCB bandit and Bayesian Knowledge Map
        self.linucb_bandit.update(observation)
        self.knowledge_map.update(observation, predicted_prob=pred_prob)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.linucb_bandit.reset()
        self.knowledge_map.reset()
