"""
Change-Adaptive Scheduler integrating Page-Hinkley drift detection.
"""

from typing import Optional, List
import numpy as np

from smart_ew.scheduling.base import BaseScheduler
from smart_ew.scheduling.uncertainty_aware import UncertaintyAwareScheduler
from smart_ew.knowledge.change_detector import PageHinkleyChangeDetector
from smart_ew.receiver.schema import ReceiverObservation


class ChangeAdaptiveScheduler(BaseScheduler):
    """
    Change-Adaptive Scheduler.
    Integrates Page-Hinkley drift detection to automatically adapt when
    emitter behaviors shift mid-simulation.
    """

    def __init__(
        self,
        num_bands: int,
        alpha_linucb: float = 0.5,
        beta_uncertainty: float = 0.5,
        gamma_recency: float = 0.2,
        change_threshold: float = 4.0,
        seed: Optional[int] = None,
        name: str = "Change-Adaptive Scheduler",
    ):
        super().__init__(num_bands, name)
        self.seed = seed
        self.change_detector = PageHinkleyChangeDetector(threshold=change_threshold)
        self.base_scheduler = UncertaintyAwareScheduler(
            num_bands=num_bands,
            alpha_linucb=alpha_linucb,
            beta_uncertainty=beta_uncertainty,
            gamma_recency=gamma_recency,
            seed=seed,
        )

        self.change_history: List[int] = []
        self._exploration_boost_steps: int = 0

    def select_next_band(self, time_step: int) -> int:
        if self._exploration_boost_steps > 0:
            self._exploration_boost_steps -= 1
            # Temporary exploration boost
            self.base_scheduler.beta_uncertainty = 1.5
        else:
            self.base_scheduler.beta_uncertainty = 0.5

        return self.base_scheduler.select_next_band(time_step)

    def update(self, observation: ReceiverObservation) -> None:
        b_id = observation.band_id
        bk = self.base_scheduler.knowledge_map.get_band_knowledge(b_id, observation.timestamp)
        pred_prob = bk.activity_probability
        target = 1 if (observation.detected and not observation.is_false_alarm) else 0

        # Check for environment change event
        is_change = self.change_detector.update(target, pred_prob)

        if is_change:
            self.change_history.append(observation.timestamp)
            # Decay stale historical knowledge and trigger 10-step exploration boost
            self.base_scheduler.knowledge_map.decay_stale_knowledge(decay_factor=0.3)
            self._exploration_boost_steps = 10

        self.base_scheduler.update(observation)

    def reset(self) -> None:
        self.change_detector.reset()
        self.base_scheduler.reset()
        self.change_history.clear()
        self._exploration_boost_steps = 0
