"""
Abstract Base Class interface for Scan Schedulers.
"""

from abc import ABC, abstractmethod
from typing import Optional

from smart_ew.receiver.schema import ReceiverObservation


class BaseScheduler(ABC):
    """Abstract Base Class for receiver frequency sweep schedulers."""

    def __init__(self, num_bands: int, name: str = "BaseScheduler"):
        self.num_bands = num_bands
        self.name = name

    @abstractmethod
    def select_next_band(self, time_step: int) -> int:
        """
        Selects the next frequency band ID (0 to num_bands - 1) to scan at the given time step.
        """
        pass

    def update(self, observation: ReceiverObservation) -> None:
        """
        Receives feedback from the receiver's scan observation.
        Baseline open-loop schedulers ignore feedback; adaptive schedulers override this.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets internal state of the scheduler."""
        pass
