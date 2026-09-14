"""
Reward configuration schema for multi-objective scan evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RewardConfig:
    """Configurable weights for transparent multi-objective reward calculation."""
    w_hit: float = 2.0
    w_threat_priority: float = 1.0
    w_false_alarm: float = -1.0
    w_retuning_cost: float = -0.5
    w_discovery_bonus: float = 0.5
    band_priorities: Dict[int, float] = field(default_factory=dict)
