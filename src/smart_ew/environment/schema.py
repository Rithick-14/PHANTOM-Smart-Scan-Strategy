"""
Data structures and schema definitions for RF Environment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass(frozen=True)
class FrequencyBand:
    """Represents a discrete frequency band in the RF spectrum."""
    band_id: int
    f_center_mhz: float
    bandwidth_mhz: float


@dataclass
class EmitterConfig:
    """Configuration for an individual emitter."""
    emitter_id: str
    emitter_type: str  # e.g., 'continuous', 'intermittent', 'periodic', 'agile'
    assigned_bands: List[int]
    signal_power_dbm: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimConfig:
    """Configuration for the RF Environment simulation."""
    num_bands: int = 10
    f_start_mhz: float = 1000.0  # 1.0 GHz
    f_step_mhz: float = 50.0     # 50 MHz spacing
    bandwidth_mhz: float = 50.0  # 50 MHz per band
    total_time_steps: int = 1000
    noise_floor_dbm: float = -100.0
    seed: Optional[int] = 42


@dataclass
class GroundTruthFrame:
    """Ground truth state of the RF spectrum at a single discrete time step."""
    time_step: int
    active_bands: Dict[int, bool]
    signal_power_dbm: Dict[int, float]
    active_emitter_ids: Dict[int, List[str]]
