"""
Uniform Random baseline scan scheduler.
"""

from typing import Optional
import numpy as np

from smart_ew.scheduling.base import BaseScheduler


class RandomScheduler(BaseScheduler):
    """
    Uniform random baseline scheduler.
    Selects a frequency band uniformly at random (0 to N-1) at each time step.
    """

    def __init__(
        self,
        num_bands: int,
        seed: Optional[int] = None,
        name: str = "Uniform Random Sweep",
    ):
        super().__init__(num_bands, name)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def select_next_band(self, time_step: int) -> int:
        return int(self.rng.integers(0, self.num_bands))

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
