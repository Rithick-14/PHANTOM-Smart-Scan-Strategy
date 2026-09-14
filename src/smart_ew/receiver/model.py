"""
Simulated Electronic Support (ES) Receiver Model.
"""

from typing import Optional

from smart_ew.environment.schema import GroundTruthFrame
from smart_ew.receiver.schema import ReceiverConfig, ReceiverObservation
from smart_ew.detection.model import DetectionModel


class ESReceiver:
    """
    Simulates a narrow instantaneous-bandwidth ES receiver sweeping across frequency bands.
    Applies retuning costs, noise limits, and probabilistic signal detection.
    """

    def __init__(self, config: ReceiverConfig = ReceiverConfig(), seed: Optional[int] = None):
        self.config = config
        self.seed = seed
        self.current_tuned_band: Optional[int] = None
        self.detection_model = DetectionModel(
            snr_threshold_db=config.snr_threshold_db,
            p_false_alarm=config.p_false_alarm,
            noise_std_db=config.noise_std_db,
            probabilistic=config.probabilistic_detection,
            seed=seed,
        )

    def scan(
        self,
        band_id: int,
        time_step: int,
        ground_truth_frame: GroundTruthFrame,
        observation_duration: float = 1.0,
    ) -> ReceiverObservation:
        """
        Executes a scan action on band_id at time_step against the provided ground truth frame.
        Computes retuning delay penalty if tuning to a different band.
        """
        # Calculate retuning cost
        if self.current_tuned_band is not None and self.current_tuned_band != band_id:
            tuning_cost = self.config.retuning_time_cost
        else:
            tuning_cost = 0.0

        self.current_tuned_band = band_id

        # Inspect ground truth for this band (simulated physical environment state)
        is_truly_active = ground_truth_frame.active_bands.get(band_id, False)
        true_power_dbm = ground_truth_frame.signal_power_dbm.get(
            band_id, self.config.noise_floor_dbm
        )

        # Run detection evaluation
        det = self.detection_model.evaluate(
            true_power_dbm=true_power_dbm,
            noise_floor_dbm=self.config.noise_floor_dbm,
            is_truly_active=is_truly_active,
        )

        return ReceiverObservation(
            band_id=band_id,
            timestamp=time_step,
            detected=det.detected,
            measured_power_dbm=det.measured_power_dbm,
            estimated_snr_db=det.estimated_snr_db,
            confidence=det.confidence,
            tuning_cost_paid=tuning_cost,
            is_false_alarm=det.is_false_alarm,
            observation_duration=observation_duration,
        )

    def reset(self) -> None:
        """Resets receiver tuning state and detection RNG."""
        self.current_tuned_band = None
        self.detection_model.reset()
