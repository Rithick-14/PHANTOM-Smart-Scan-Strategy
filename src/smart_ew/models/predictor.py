"""
Concrete Machine Learning Activity Predictors (Logistic Regression, Random Forest, Gradient Boosting).
"""

from typing import Optional, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from smart_ew.models.base import BaseActivityPredictor
from smart_ew.features.schema import BandFeatureVector


class LogisticRegressionPredictor(BaseActivityPredictor):
    """Logistic Regression activity predictor baseline."""

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.model = LogisticRegression(random_state=seed, max_iter=200)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(np.unique(y)) < 2:
            return  # Need at least two classes to fit logistic regression
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.full(X.shape[0], 0.5, dtype=np.float32)
        probas = self.model.predict_proba(X)
        if probas.shape[1] == 1:
            return np.full(X.shape[0], float(probas[0, 0]), dtype=np.float32)
        return probas[:, 1].astype(np.float32)

    def reset(self) -> None:
        self.model = LogisticRegression(random_state=self.seed, max_iter=200)
        self.is_fitted = False


class RandomForestPredictor(BaseActivityPredictor):
    """Random Forest activity predictor."""

    def __init__(self, n_estimators: int = 50, seed: Optional[int] = None):
        super().__init__(seed)
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(np.unique(y)) < 2:
            return
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.full(X.shape[0], 0.5, dtype=np.float32)
        probas = self.model.predict_proba(X)
        if probas.shape[1] == 1:
            return np.full(X.shape[0], float(probas[0, 0]), dtype=np.float32)
        return probas[:, 1].astype(np.float32)

    def reset(self) -> None:
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.seed)
        self.is_fitted = False


class GradientBoostingPredictor(BaseActivityPredictor):
    """Gradient Boosting activity predictor."""

    def __init__(self, n_estimators: int = 50, seed: Optional[int] = None):
        super().__init__(seed)
        self.n_estimators = n_estimators
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(np.unique(y)) < 2:
            return
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.full(X.shape[0], 0.5, dtype=np.float32)
        probas = self.model.predict_proba(X)
        if probas.shape[1] == 1:
            return np.full(X.shape[0], float(probas[0, 0]), dtype=np.float32)
        return probas[:, 1].astype(np.float32)

    def reset(self) -> None:
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators, random_state=self.seed
        )
        self.is_fitted = False


class MLActivityPredictor:
    """
    High-level online learning manager. Buffers feature vectors and updates
    the underlying model periodically.
    """

    def __init__(
        self,
        predictor: Optional[BaseActivityPredictor] = None,
        fit_interval: int = 5,
        min_samples: int = 10,
        seed: Optional[int] = None,
    ):
        self.predictor = predictor or RandomForestPredictor(seed=seed)
        self.fit_interval = fit_interval
        self.min_samples = min_samples
        self.seed = seed

        self._x_buffer: List[np.ndarray] = []
        self._y_buffer: List[int] = []
        self._sample_count: int = 0

    def add_sample(self, x: np.ndarray, y: int) -> None:
        """Adds a new feature sample and binary target (1 = detected hit, 0 = quiet/miss)."""
        self._x_buffer.append(x)
        self._y_buffer.append(y)
        self._sample_count += 1

        if (
            self._sample_count >= self.min_samples
            and self._sample_count % self.fit_interval == 0
        ):
            self.fit()

    def fit(self) -> None:
        """Fits the underlying ML model on accumulated feature buffer."""
        if len(self._y_buffer) < self.min_samples:
            return
        X = np.array(self._x_buffer, dtype=np.float32)
        y = np.array(self._y_buffer, dtype=int)
        self.predictor.fit(X, y)

    def predict_band_activity(self, feature_vector: BandFeatureVector) -> float:
        """Predicts probability P(active = 1 | feature_vector)."""
        X = feature_vector.to_numpy().reshape(1, -1)
        proba = self.predictor.predict_proba(X)
        return float(proba[0])

    def predict_all_bands(
        self, feature_vectors: dict[int, BandFeatureVector]
    ) -> dict[int, float]:
        """Predicts activity probabilities for all frequency bands."""
        if not feature_vectors:
            return {}
        band_ids = list(feature_vectors.keys())
        X_matrix = np.array([feature_vectors[b].to_numpy() for b in band_ids], dtype=np.float32)
        probas = self.predictor.predict_proba(X_matrix)
        return {b: float(probas[i]) for i, b in enumerate(band_ids)}

    def reset(self) -> None:
        """Resets sample buffers and model."""
        self._x_buffer.clear()
        self._y_buffer.clear()
        self._sample_count = 0
        self.predictor.reset()
