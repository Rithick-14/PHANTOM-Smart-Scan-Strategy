"""
Periodic emitter model simulating cyclic radar sweeps or pulsed transmissions.
"""

from typing import Optional
from smart_ew.emitters.base import BaseEmitter
from smart_ew.environment.schema import EmitterConfig


class PeriodicEmitter(BaseEmitter):
    """
    Emitter that transmits periodically with a fixed period, duration, and phase offset.
    Simulates rotating radar antenna beams or cyclic pulsed communications.
    """

    def __init__(self, config: EmitterConfig, seed: Optional[int] = None):
        super().__init__(config, seed)
        self.period: int = int(self.config.parameters.get("period", 10))
        self.duty_duration: int = int(self.config.parameters.get("duty_duration", 2))
        self.phase_offset: int = int(self.config.parameters.get("phase_offset", 0))
        self.start_time: int = int(self.config.parameters.get("start_time", 0))
        self.end_time: Optional[int] = self.config.parameters.get("end_time", None)

    def is_active(self, time_step: int, band_id: int) -> bool:
        if band_id not in self.assigned_bands:
            return False
        if time_step < self.start_time:
            return False
        if self.end_time is not None and time_step > self.end_time:
            return False

        cycle_pos = (time_step - self.phase_offset - self.start_time) % self.period
        return cycle_pos < self.duration_bounds()

    def duration_bounds(self) -> int:
        return max(1, min(self.duty_duration, self.period))

    def get_signal_power_dbm(self, time_step: int, band_id: int) -> float:
        if self.is_active(time_step, band_id):
            return self.signal_power_dbm
        return -float("inf")

    def step(self, time_step: int) -> None:
        pass

    def reset(self) -> None:
        pass
