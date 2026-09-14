"""
Frequency-Agile / Hopping emitter model simulating multi-band state transitions.
"""

from typing import Optional, List
import numpy as np

from smart_ew.emitters.base import BaseEmitter
from smart_ew.environment.schema import EmitterConfig


class FrequencyAgileEmitter(BaseEmitter):
    """
    Emitter that hops across frequency bands according to a predefined hopping sequence
    or a Markov transition probability matrix.
    """

    def __init__(self, config: EmitterConfig, seed: Optional[int] = None):
        super().__init__(config, seed)
        self.dwell_time: int = int(self.config.parameters.get("dwell_time", 1))
        self.hop_sequence: Optional[List[int]] = self.config.parameters.get(
            "hop_sequence", None
        )
        self.transition_matrix: Optional[np.ndarray] = self.config.parameters.get(
            "transition_matrix", None
        )
        self.start_time: int = int(self.config.parameters.get("start_time", 0))
        self.end_time: Optional[int] = self.config.parameters.get("end_time", None)

        self._active_band_history: dict[int, int] = {}
        self._current_band_idx: int = 0

    def is_active(self, time_step: int, band_id: int) -> bool:
        if time_step < self.start_time:
            return False
        if self.end_time is not None and time_step > self.end_time:
            return False

        active_band = self._get_active_band(time_step)
        return band_id == active_band

    def _get_active_band(self, target_step: int) -> int:
        if target_step in self._active_band_history:
            return self._active_band_history[target_step]

        # Calculate dwell epoch index
        epoch = (target_step - self.start_time) // self.dwell_time

        if self.hop_sequence and len(self.hop_sequence) > 0:
            band = self.hop_sequence[epoch % len(self.hop_sequence)]
        elif self.transition_matrix is not None and len(self.assigned_bands) > 1:
            # Simulate Markov transitions up to target epoch
            curr_b = self.assigned_bands[0]
            rng = np.random.default_rng(self.seed + epoch)
            for _ in range(epoch):
                curr_idx = self.assigned_bands.index(curr_b) if curr_b in self.assigned_bands else 0
                probs = self.transition_matrix[curr_idx]
                probs = probs / np.sum(probs)
                curr_b = rng.choice(self.assigned_bands, p=probs)
            band = curr_b
        else:
            # Fallback: pseudo-random hop among assigned bands
            rng = np.random.default_rng(self.seed + epoch)
            band = rng.choice(self.assigned_bands)

        self._active_band_history[target_step] = int(band)
        return int(band)

    def get_signal_power_dbm(self, time_step: int, band_id: int) -> float:
        if self.is_active(time_step, band_id):
            return self.signal_power_dbm
        return -float("inf")

    def step(self, time_step: int) -> None:
        self._get_active_band(time_step)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._active_band_history.clear()
        self._current_band_idx = 0
