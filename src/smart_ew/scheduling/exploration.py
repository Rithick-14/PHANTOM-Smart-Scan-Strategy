"""
Uncertainty Estimation and Exploration Policies (UCB, Epsilon-Greedy, Softmax).
"""

from typing import Dict, List, Optional
import numpy as np

from smart_ew.knowledge.schema import BandKnowledge


class UncertaintyEstimator:
    """
    Computes multi-source uncertainty estimates combining Bayesian posterior variance,
    ensemble model disagreement, and observation count sparsity.
    """

    @staticmethod
    def compute_bayesian_uncertainty(bk: BandKnowledge) -> float:
        """Computes standard deviation of Beta posterior distribution."""
        total = bk.beta_alpha + bk.beta_beta
        var = (bk.beta_alpha * bk.beta_beta) / ((total ** 2) * (total + 1.0))
        return float(np.sqrt(var))

    @staticmethod
    def compute_ensemble_disagreement(model_predictions: List[float]) -> float:
        """Computes prediction variance across ensemble models (LR, RF, GB)."""
        if len(model_predictions) < 2:
            return 0.0
        return float(np.var(model_predictions))

    @staticmethod
    def compute_count_sparsity_uncertainty(total_scans: int) -> float:
        """Computes count-based epistemic uncertainty 1 / sqrt(N + 1)."""
        return float(1.0 / np.sqrt(total_scans + 1.0))

    def compute_combined_uncertainty(
        self,
        bk: BandKnowledge,
        model_predictions: Optional[List[float]] = None,
        w_bayes: float = 0.4,
        w_ensemble: float = 0.3,
        w_sparsity: float = 0.3,
    ) -> float:
        """
        Fuses Bayesian variance, ensemble disagreement, and observation sparsity into a [0, 1] score.
        """
        u_bayes = self.compute_bayesian_uncertainty(bk)
        u_sparsity = self.compute_count_sparsity_uncertainty(bk.total_observations)
        
        if model_predictions and len(model_predictions) >= 2:
            u_ensemble = float(np.std(model_predictions))
        else:
            u_ensemble = u_bayes

        combined = w_bayes * u_bayes + w_ensemble * u_ensemble + w_sparsity * u_sparsity
        return float(np.clip(combined, 0.0, 1.0))


class ExplorationPolicy:
    """
    Exploration Strategy Engine implementing UCB-style scoring, Epsilon-Greedy, and Softmax.
    """

    def __init__(
        self,
        strategy: str = "ucb",
        c_ucb: float = 1.5,
        epsilon: float = 0.1,
        temperature: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.strategy = strategy.lower()
        self.c_ucb = c_ucb
        self.epsilon = epsilon
        self.temperature = temperature
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def select_action(
        self, predicted_probas: Dict[int, float], uncertainties: Dict[int, float]
    ) -> int:
        """
        Selects frequency band according to configured exploration policy.
        """
        band_ids = list(predicted_probas.keys())
        if not band_ids:
            return 0

        if self.strategy == "epsilon_greedy":
            if self.rng.random() < self.epsilon:
                return int(self.rng.choice(band_ids))
            else:
                max_p = max(predicted_probas.values())
                best_bands = [b for b, p in predicted_probas.items() if np.isclose(p, max_p)]
                return int(self.rng.choice(best_bands))

        elif self.strategy == "softmax":
            logits = np.array([predicted_probas[b] / max(0.01, self.temperature) for b in band_ids])
            # Numerically stable softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            return int(self.rng.choice(band_ids, p=probs))

        else:  # Default: UCB (Upper Confidence Bound)
            scores = {
                b: predicted_probas[b] + self.c_ucb * uncertainties.get(b, 0.0)
                for b in band_ids
            }
            max_score = max(scores.values())
            best_bands = [b for b, s in scores.items() if np.isclose(s, max_score)]
            return int(self.rng.choice(best_bands))

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
