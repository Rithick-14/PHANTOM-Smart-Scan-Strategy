"""
Data structure for per-band knowledge state maintained in the Dynamic RF Knowledge Map.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class BandKnowledge:
    """Per-band dynamic knowledge state combining ML predictions, Bayesian belief, and statistics."""
    band_id: int
    activity_probability: float = 0.5
    confidence: float = 0.0
    uncertainty: float = 1.0
    total_observations: int = 0
    hits: int = 0
    misses: int = 0
    last_scan_time: int = -1
    last_hit_time: int = -1
    time_since_last_scan: float = 0.0
    time_since_last_hit: float = 0.0
    beta_alpha: float = 1.0
    beta_beta: float = 1.0
    transition_counts: Dict[int, float] = field(default_factory=dict)
