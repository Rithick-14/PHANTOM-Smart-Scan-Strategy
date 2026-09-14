"""
Continuous emitter model.
"""

from typing import Optional
from smart_ew.emitters.base import BaseEmitter
from smart_ew.environment.schema import EmitterConfig


class ContinuousEmitter(BaseEmitter):
    """Emitter that transmits continuously on assigned bands during active time window."""

    def __init__(self, config: EmitterConfig, seed: Optional[int] = None):
        super().__init__(config, seed)
        self.start_time: int = self.config.parameters.get("start_time", 0)
        self.end_time: Optional[int] = self.config.parameters.get("end_time", None)

    def is_active(self, time_step: int, band_id: int) -> bool:
        if band_id not in self.assigned_bands:
            return False
        if time_step < self.start_time:
            return False
        if self.end_time is not None and time_step > self.end_time:
            return False
        return True

    def get_signal_power_dbm(self, time_step: int, band_id: int) -> float:
        if self.is_active(time_step, band_id):
            return self.signal_power_dbm
        return -float("inf")

    def step(self, time_step: int) -> None:
        pass

    def reset(self) -> None:
        pass
