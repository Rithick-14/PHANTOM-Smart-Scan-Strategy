"""
RF Environment Simulator maintaining ground truth spectrum activity.
"""

from typing import Dict, List, Optional
import numpy as np

from smart_ew.environment.schema import (
    FrequencyBand,
    SimConfig,
    GroundTruthFrame,
)
from smart_ew.emitters.base import BaseEmitter


class RFEnvironment:
    """
    Simulates a multi-band RF Environment containing ground truth emitter activity.
    Ground truth information is maintained separately from receiver observations.
    """

    def __init__(self, config: SimConfig = SimConfig()):
        self.config = config
        self.bands: Dict[int, FrequencyBand] = self._create_bands()
        self.emitters: List[BaseEmitter] = []
        self.current_time_step: int = 0
        self._history: Dict[int, GroundTruthFrame] = {}

    def _create_bands(self) -> Dict[int, FrequencyBand]:
        bands = {}
        for b in range(self.config.num_bands):
            f_center = self.config.f_start_mhz + b * self.config.f_step_mhz
            bands[b] = FrequencyBand(
                band_id=b,
                f_center_mhz=f_center,
                bandwidth_mhz=self.config.bandwidth_mhz,
            )
        return bands

    def add_emitter(self, emitter: BaseEmitter) -> None:
        """Registers an emitter in the environment."""
        self.emitters.append(emitter)

    def step(self) -> GroundTruthFrame:
        """Advances simulation by one time step and generates the ground truth frame."""
        frame = self._compute_frame(self.current_time_step)
        self._history[self.current_time_step] = frame
        self.current_time_step += 1
        return frame

    def _compute_frame(self, time_step: int) -> GroundTruthFrame:
        active_bands: Dict[int, bool] = {b: False for b in self.bands}
        signal_power: Dict[int, float] = {b: self.config.noise_floor_dbm for b in self.bands}
        active_emitters: Dict[int, List[str]] = {b: [] for b in self.bands}

        for emitter in self.emitters:
            emitter.step(time_step)
            for band_id in emitter.assigned_bands:
                if band_id in self.bands and emitter.is_active(time_step, band_id):
                    active_bands[band_id] = True
                    active_emitters[band_id].append(emitter.emitter_id)

                    # Power accumulation (in linear mW scale, then convert back to dBm)
                    p_dbm = emitter.get_signal_power_dbm(time_step, band_id)
                    p_mw = 10.0 ** (p_dbm / 10.0)
                    existing_mw = 10.0 ** (signal_power[band_id] / 10.0)
                    combined_mw = existing_mw + p_mw
                    signal_power[band_id] = 10.0 * np.log10(combined_mw)

        return GroundTruthFrame(
            time_step=time_step,
            active_bands=active_bands,
            signal_power_dbm=signal_power,
            active_emitter_ids=active_emitters,
        )

    def get_ground_truth(self, time_step: int) -> GroundTruthFrame:
        """Retrieves ground truth frame for a specified time step."""
        if time_step in self._history:
            return self._history[time_step]
        return self._compute_frame(time_step)

    def get_ground_truth_matrix(self, total_steps: Optional[int] = None) -> np.ndarray:
        """
        Generates a 2D numpy matrix of shape (num_bands, total_steps)
        containing binary activity ground truth (1 = active, 0 = quiet).
        """
        steps = total_steps if total_steps is not None else self.config.total_time_steps
        matrix = np.zeros((self.config.num_bands, steps), dtype=int)

        for t in range(steps):
            frame = self.get_ground_truth(t)
            for b_id, active in frame.active_bands.items():
                matrix[b_id, t] = 1 if active else 0

        return matrix

    def get_power_matrix(self, total_steps: Optional[int] = None) -> np.ndarray:
        """
        Generates a 2D numpy matrix of shape (num_bands, total_steps)
        containing total signal power in dBm.
        """
        steps = total_steps if total_steps is not None else self.config.total_time_steps
        matrix = np.zeros((self.config.num_bands, steps), dtype=float)

        for t in range(steps):
            frame = self.get_ground_truth(t)
            for b_id, power in frame.signal_power_dbm.items():
                matrix[b_id, t] = power

        return matrix

    def reset(self) -> None:
        """Resets environment and all registered emitters."""
        self.current_time_step = 0
        self._history.clear()
        for emitter in self.emitters:
            emitter.reset()
