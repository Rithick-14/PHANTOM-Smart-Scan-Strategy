"""
Metrics calculation data structures for evaluating receiver and scheduler performance.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from smart_ew.receiver.schema import ReceiverObservation


@dataclass
class EvaluationMetrics:
    """Consolidated performance metrics for an ES receiver scheduling run."""
    scheduler_name: str
    total_steps: int
    num_bands: int
    total_scans: int
    total_active_scans: int
    total_intercepts: int
    total_quiet_scans: int
    total_false_alarms: int
    total_transmission_opportunities: int
    
    # 1. Probability of Detection (Pd)
    probability_of_detection: float  # total_intercepts / total_active_scans
    
    # 2. Probability of False Alarm (Pfa)
    probability_of_false_alarm: float  # total_false_alarms / total_quiet_scans
    
    # 3. Intercept Rate
    intercept_rate: float  # total_intercepts / total_transmission_opportunities
    
    # 4. Average Intercept Time
    average_intercept_time: float  # mean delay from signal start to first detection
    
    # 5. Intercept Time Error
    intercept_time_error_std: float  # standard deviation of intercept delays
    
    # 6. Prediction Accuracy / Brier Score
    prediction_accuracy: float = 0.0
    
    # 7. Cumulative Reward
    cumulative_reward: float = 0.0
    
    # 8. Spectrum Coverage Ratio
    spectrum_coverage_ratio: float = 1.0  # Temporal windowed coverage ratio
    spatial_coverage_ratio: float = 1.0   # Unique bands discovered ratio
    temporal_coverage_ratio: float = 1.0  # Sliding window coverage duty cycle ratio
    
    # 9. New Emitter Discovery Time
    new_emitter_discovery_time: float = 0.0
    
    # 10. Recovery Time After Environment Change
    recovery_time: float = 0.0
    
    total_retuning_cost: float = 0.0
    observations: List[ReceiverObservation] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, Any]:
        """Returns a concise dictionary summary of key metrics."""
        return {
            "scheduler_name": self.scheduler_name,
            "total_steps": self.total_steps,
            "num_bands": self.num_bands,
            "pd": round(self.probability_of_detection, 4),
            "pfa": round(self.probability_of_false_alarm, 4),
            "intercept_rate": round(self.intercept_rate, 4),
            "avg_intercept_time": round(self.average_intercept_time, 2),
            "intercept_time_error_std": round(self.intercept_time_error_std, 2),
            "spectrum_coverage": round(self.spectrum_coverage_ratio, 4),
            "total_retuning_cost": round(self.total_retuning_cost, 2),
            "cumulative_reward": round(self.cumulative_reward, 2),
            "total_intercepts": self.total_intercepts,
            "total_false_alarms": self.total_false_alarms,
        }
