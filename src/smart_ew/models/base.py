"""
Abstract Base Class interface for Machine Learning Activity Predictors.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseActivityPredictor(ABC):
    """Abstract Base Class for predicting spectrum band activity probability."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits model on feature matrix X (shape: [N, D]) and binary targets y (shape: [N]).
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts activity probability P(active = 1 | X) for each sample in X.
        Returns 1D numpy array of probabilities in range [0.0, 1.0].
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets predictor model state."""
        pass
