"""
Multi-Objective Reward Engine calculating scalar training and evaluation rewards.
"""

from typing import Dict, Optional

from smart_ew.receiver.schema import ReceiverObservation
from smart_ew.knowledge.map import DynamicRFKnowledgeMap
from smart_ew.rewards.schema import RewardConfig


class MultiObjectiveRewardEngine:
    """
    Transparent multi-objective reward calculator.
    Combines intercept value, threat priority, false alarm risk, retuning penalty,
    and discovery bonuses into a unified scalar reward signal.
    """

    def __init__(self, config: RewardConfig = RewardConfig()):
        self.config = config

    def compute_components(
        self,
        observation: ReceiverObservation,
        knowledge_map: Optional[DynamicRFKnowledgeMap] = None,
    ) -> Dict[str, float]:
        """
        Computes detailed breakdown of all reward components for observation.
        """
        b_id = observation.band_id
        is_hit = observation.detected and not observation.is_false_alarm

        hit_val = self.config.w_hit if is_hit else 0.0

        if is_hit:
            priority_mult = self.config.band_priorities.get(b_id, 1.0)
            threat_val = self.config.w_threat_priority * priority_mult
        else:
            threat_val = 0.0

        fa_penalty = self.config.w_false_alarm if observation.is_false_alarm else 0.0
        tuning_penalty = self.config.w_retuning_cost * observation.tuning_cost_paid

        if is_hit and knowledge_map is not None:
            bk = knowledge_map.get_band_knowledge(b_id, observation.timestamp)
            disc_bonus = self.config.w_discovery_bonus * bk.uncertainty
        else:
            disc_bonus = 0.0

        total = hit_val + threat_val + fa_penalty + tuning_penalty + disc_bonus

        return {
            "hit_reward": float(hit_val),
            "threat_priority_reward": float(threat_val),
            "false_alarm_penalty": float(fa_penalty),
            "retuning_penalty": float(tuning_penalty),
            "discovery_bonus": float(disc_bonus),
            "total_reward": float(total),
        }

    def compute_reward(
        self,
        observation: ReceiverObservation,
        knowledge_map: Optional[DynamicRFKnowledgeMap] = None,
    ) -> float:
        """Returns total scalar multi-objective reward."""
        comps = self.compute_components(observation, knowledge_map)
        return comps["total_reward"]
