"""
Sequential Sweep baseline scheduler.
"""

from smart_ew.scheduling.base import BaseScheduler


class SequentialScheduler(BaseScheduler):
    """
    Fixed/open-loop sequential sweep scheduler.
    Scans frequency bands in a deterministic cyclic sequence (0, 1, ..., N-1, 0, ...).
    """

    def __init__(
        self,
        num_bands: int,
        start_band: int = 0,
        step_size: int = 1,
        name: str = "Sequential Sweep",
    ):
        super().__init__(num_bands, name)
        self.start_band = start_band % num_bands
        self.step_size = step_size
        self.current_band = self.start_band

    def select_next_band(self, time_step: int) -> int:
        selected = self.current_band
        self.current_band = (self.current_band + self.step_size) % self.num_bands
        return selected

    def reset(self) -> None:
        self.current_band = self.start_band
