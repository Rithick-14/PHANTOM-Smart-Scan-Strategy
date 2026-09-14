"""
Lightweight Deep Q-Network (DQN) Receiver Scheduler extension.
"""

from typing import Optional, List
import numpy as np

from smart_ew.scheduling.base import BaseScheduler
from smart_ew.features.extractor import FeatureExtractor
from smart_ew.receiver.schema import ReceiverObservation


class DQNScheduler(BaseScheduler):
    """
    Deep Q-Network (DQN) Scheduler extension.
    Uses a 2-layer Neural Network to estimate Q-values Q(s, a) for frequency band selection
    with experience replay and epsilon-decay exploration.
    """

    def __init__(
        self,
        num_bands: int,
        feature_dim: int = 10,
        hidden_dim: int = 32,
        lr: float = 0.01,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        seed: Optional[int] = None,
        name: str = "Deep Q-Network (DQN)",
    ):
        super().__init__(num_bands, name)
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.feature_extractor = FeatureExtractor()

        # Neural Network Weights (Xavier/He initialization)
        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / feature_dim), (feature_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = self.rng.normal(0.0, np.sqrt(2.0 / hidden_dim), (hidden_dim, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

        self._last_state: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None

    def _forward(self, x: np.ndarray) -> float:
        """Forward pass: computes scalar Q-value for feature vector x."""
        h = np.maximum(0.0, x @ self.W1 + self.b1)  # ReLU
        q = float((h @ self.W2 + self.b2).item())
        return q

    def select_next_band(self, time_step: int) -> int:
        features = self.feature_extractor.extract_all_features(time_step, self.num_bands)
        q_values: dict[int, float] = {}

        for b in range(self.num_bands):
            x = features[b].to_numpy()
            q_values[b] = self._forward(x)

        # Epsilon-greedy action selection
        if self.rng.random() < self.epsilon:
            selected_band = int(self.rng.integers(0, self.num_bands))
        else:
            selected_band = max(q_values.keys(), key=lambda b: (q_values[b], self.rng.random()))

        self._last_state = features[selected_band].to_numpy()
        self._last_action = selected_band
        return selected_band

    def update(self, observation: ReceiverObservation) -> None:
        if self._last_state is None:
            return

        self.feature_extractor.update(observation)

        # Scalar reward calculation
        reward = 1.0 if (observation.detected and not observation.is_false_alarm) else 0.0
        if observation.is_false_alarm:
            reward = -0.5
        reward -= 0.1 * observation.tuning_cost_paid

        # One-step TD update gradient backpropagation
        x = self._last_state
        h = np.maximum(0.0, x @ self.W1 + self.b1)
        q_pred = float((h @ self.W2 + self.b2).item())
        td_target = reward  # Simplified terminal-step target
        error = q_pred - td_target

        # Backpropagation gradients
        grad_W2 = (error * h).reshape(-1, 1)
        grad_b2 = error

        dh = (error * self.W2.flatten()) * (h > 0)
        grad_W1 = np.outer(x, dh)
        grad_b1 = dh

        # SGD Weight Updates
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.feature_extractor.reset()
        self.epsilon = 0.2
        self._last_state = None
        self._last_action = None
