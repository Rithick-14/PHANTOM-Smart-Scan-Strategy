"""
Abstract base class interface for RF Emitters.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from smart_ew.environment.schema import EmitterConfig


class BaseEmitter(ABC):
    """Abstract Base Class for all emitter behavior models."""

    def __init__(self, config: EmitterConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @property
    def emitter_id(self) -> str:
        return self.config.emitter_id

    @property
    def emitter_type(self) -> str:
        return self.config.emitter_type

    @property
    def assigned_bands(self) -> list[int]:
        return self.config.assigned_bands

    @property
    def signal_power_dbm(self) -> float:
        return self.config.signal_power_dbm

    @abstractmethod
    def is_active(self, time_step: int, band_id: int) -> bool:
        """Returns True if the emitter is active on band_id at time_step."""
        pass

    @abstractmethod
    def get_signal_power_dbm(self, time_step: int, band_id: int) -> float:
        """Returns the signal power in dBm if active, otherwise -infinity or noise level."""
        pass

    @abstractmethod
    def step(self, time_step: int) -> None:
        """Advances internal emitter state to time_step."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets emitter state to initial conditions."""
        pass
