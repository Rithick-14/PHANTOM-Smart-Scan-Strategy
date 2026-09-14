"""
Data structures for ES Receiver configuration and observation outputs.
"""

from dataclasses import dataclass


@dataclass
class ReceiverConfig:
    """Hardware and operational configuration for Electronic Support receiver."""
    instantaneous_bw_mhz: float = 50.0
    retuning_time_cost: float = 1.0
    noise_floor_dbm: float = -100.0
    snr_threshold_db: float = 10.0
    p_false_alarm: float = 0.01
    probabilistic_detection: bool = True
    noise_std_db: float = 1.0
    total_spectrum_start_mhz: float = 1000.0
    total_spectrum_end_mhz: float = 1500.0


@dataclass
class ReceiverObservation:
    """Result of an ES receiver scan action on a specific frequency band at a timestamp."""
    band_id: int
    timestamp: int
    detected: bool
    measured_power_dbm: float
    estimated_snr_db: float
    confidence: float
    tuning_cost_paid: float
    is_false_alarm: bool
    observation_duration: float = 1.0
