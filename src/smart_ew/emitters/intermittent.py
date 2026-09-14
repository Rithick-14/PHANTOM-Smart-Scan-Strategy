"""
Intermittent emitter model using a Markov switching process.
"""

from typing import Optional
import numpy as np

from smart_ew.emitters.base import BaseEmitter
from smart_ew.environment.schema import EmitterConfig


class IntermittentEmitter(BaseEmitter):
    """
    Emitter that switches between Active and Quiet states based on a 2-state Markov process
    or predefined duty cycle parameters.
    """

    def __init__(self, config: EmitterConfig, seed: Optional[int] = None):
        super().__init__(config, seed)
        # p_on: probability of transitioning from Quiet -> Active
        # p_off: probability of transitioning from Active -> Quiet
        self.p_on: float = float(self.config.parameters.get("p_on", 0.2))
        self.p_off: float = float(self.config.parameters.get("p_off", 0.4))
        self.initial_active: bool = bool(self.config.parameters.get("initial_active", False))

        self.start_time: int = self.config.parameters.get("start_time", 0)
        self.end_time: Optional[int] = self.config.parameters.get("end_time", None)

        self._active_state: bool = self.initial_active
        self._history: dict[int, bool] = {}

    def is_active(self, time_step: int, band_id: int) -> bool:
        if band_id not in self.assigned_bands:
            return False
        if time_step < self.start_time:
            return False
        if self.end_time is not None and time_step > self.end_time:
            return False

        if time_step in self._history:
            return self._history[time_step]

        # Simulate steps up to time_step if not already computed
        return self._evaluate_to_time(time_step)

    def get_signal_power_dbm(self, time_step: int, band_id: int) -> float:
        if self.is_active(time_step, band_id):
            return self.signal_power_dbm
        return -float("inf")

    def _evaluate_to_time(self, target_step: int) -> bool:
        # Determine highest evaluated step in history
        last_step = max(self._history.keys()) if self._history else self.start_time - 1
        current_state = self._history.get(last_step, self.initial_active)

        for step in range(last_step + 1, target_step + 1):
            if step < self.start_time or (self.end_time is not None and step > self.end_time):
                current_state = False
            else:
                if current_state:
                    # Active -> Quiet transition check
                    if self.rng.random() < self.p_off:
                        current_state = False
                else:
                    # Quiet -> Active transition check
                    if self.rng.random() < self.p_on:
                        current_state = True

            self._history[step] = current_state

        return self._history[target_step]

    def step(self, time_step: int) -> None:
        self.is_active(time_step, self.assigned_bands[0] if self.assigned_bands else 0)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._active_state = self.initial_active
        self._history.clear()
