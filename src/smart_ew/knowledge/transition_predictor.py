"""
Transition Predictor for predicting frequency-hopping behavior of agile emitters.
"""

from typing import Dict, Optional
from smart_ew.knowledge.map import DynamicRFKnowledgeMap


class TransitionPredictor:
    """
    Predicts next likely frequency band for frequency-hopping emitters
    using empirical state transition matrices in DynamicRFKnowledgeMap.
    """

    def predict_next_band_probabilities(
        self, knowledge_map: DynamicRFKnowledgeMap, current_band: int
    ) -> Dict[int, float]:
        """
        Returns probability distribution P(next_band | current_band) across all bands.
        """
        probs = {}
        for b in range(knowledge_map.num_bands):
            probs[b] = knowledge_map.get_transition_probability(from_band=current_band, to_band=b)
        return probs

    def get_most_likely_next_band(
        self, knowledge_map: DynamicRFKnowledgeMap, current_band: int
    ) -> Optional[int]:
        """
        Returns the band ID with highest empirical transition probability from current_band.
        """
        probs = self.predict_next_band_probabilities(knowledge_map, current_band)
        if not probs or max(probs.values()) == 0.0:
            return None
        return max(probs.keys(), key=lambda b: probs[b])
