"""
Simulation Runner executing closed-loop evaluation between RFEnvironment, ESReceiver, and Schedulers.
"""

from typing import Optional, List
import numpy as np

from smart_ew.environment.simulator import RFEnvironment
from smart_ew.receiver.model import ESReceiver
from smart_ew.scheduling.base import BaseScheduler
from smart_ew.evaluation.metrics import EvaluationMetrics
from smart_ew.receiver.schema import ReceiverObservation


class SimulationRunner:
    """
    Orchestrates closed-loop simulation runs and computes performance figures of merit.
    """

    def run(
        self,
        environment: RFEnvironment,
        receiver: ESReceiver,
        scheduler: BaseScheduler,
        total_steps: Optional[int] = None,
        reset_components: bool = True,
    ) -> EvaluationMetrics:
        """
        Executes simulation loop across total_steps and returns calculated evaluation metrics.
        """
        if reset_components:
            environment.reset()
            receiver.reset()
            scheduler.reset()

        steps = total_steps if total_steps is not None else environment.config.total_time_steps
        num_bands = environment.config.num_bands

        observations: List[ReceiverObservation] = []
        unique_bands_scanned = set()

        total_active_scans = 0
        total_quiet_scans = 0
        total_intercepts = 0
        total_false_alarms = 0
        total_retuning_cost = 0.0
        cumulative_reward = 0.0

        total_transmission_opportunities = 0

        for t in range(steps):
            frame = environment.step()
            
            # Count ground truth active opportunities across spectrum
            for b_id, active in frame.active_bands.items():
                if active:
                    total_transmission_opportunities += 1

            action_band = scheduler.select_next_band(t)
            obs = receiver.scan(action_band, t, frame)
            scheduler.update(obs)

            observations.append(obs)
            unique_bands_scanned.add(action_band)
            total_retuning_cost += obs.tuning_cost_paid

            is_truly_active = frame.active_bands.get(action_band, False)

            if is_truly_active:
                total_active_scans += 1
                if obs.detected and not obs.is_false_alarm:
                    total_intercepts += 1
                    cumulative_reward += 10.0  # Positive hit reward
                else:
                    cumulative_reward -= 5.0   # Miss penalty
            else:
                total_quiet_scans += 1
                if obs.detected or obs.is_false_alarm:
                    total_false_alarms += 1
                    cumulative_reward -= 8.0   # False alarm penalty
                else:
                    cumulative_reward += 0.5   # Quiet confirmation reward

            # Subtract retuning cost penalty
            cumulative_reward -= obs.tuning_cost_paid * 2.0

        # 1. Probability of Detection (Pd)
        pd = (
            float(total_intercepts / total_active_scans)
            if total_active_scans > 0
            else 0.0
        )

        # 2. Probability of False Alarm (Pfa)
        pfa = (
            float(total_false_alarms / total_quiet_scans)
            if total_quiet_scans > 0
            else 0.0
        )

        # 3. Intercept Rate
        intercept_rate = (
            float(total_intercepts / total_transmission_opportunities)
            if total_transmission_opportunities > 0
            else 0.0
        )

        # 4 & 5. Average Intercept Time and Intercept Time Error (std)
        intercept_delays = self._compute_intercept_delays(environment, observations, steps)
        avg_intercept_time = (
            float(np.mean(intercept_delays)) if len(intercept_delays) > 0 else float(steps)
        )
        intercept_time_error_std = (
            float(np.std(intercept_delays)) if len(intercept_delays) > 0 else 0.0
        )

        # 8. Spectrum Coverage Ratio
        coverage_ratio = float(len(unique_bands_scanned) / num_bands)

        return EvaluationMetrics(
            scheduler_name=scheduler.name,
            total_steps=steps,
            num_bands=num_bands,
            total_scans=steps,
            total_active_scans=total_active_scans,
            total_intercepts=total_intercepts,
            total_quiet_scans=total_quiet_scans,
            total_false_alarms=total_false_alarms,
            total_transmission_opportunities=total_transmission_opportunities,
            probability_of_detection=pd,
            probability_of_false_alarm=pfa,
            intercept_rate=intercept_rate,
            average_intercept_time=avg_intercept_time,
            intercept_time_error_std=intercept_time_error_std,
            cumulative_reward=cumulative_reward,
            spectrum_coverage_ratio=coverage_ratio,
            total_retuning_cost=total_retuning_cost,
            observations=observations,
        )

    def _compute_intercept_delays(
        self, environment: RFEnvironment, observations: List[ReceiverObservation], steps: int
    ) -> List[float]:
        """
        Computes delay from emitter burst activation to first positive receiver detection.
        """
        delays = []
        matrix = environment.get_ground_truth_matrix(total_steps=steps)

        for band_id in range(environment.config.num_bands):
            active_series = matrix[band_id, :]
            # Find burst activation time steps (0 -> 1 transitions or step 0 active)
            burst_starts = []
            for t in range(steps):
                if active_series[t] == 1:
                    if t == 0 or active_series[t - 1] == 0:
                        burst_starts.append(t)

            # For each burst start, find first time receiver scanned band_id and detected signal
            for start_t in burst_starts:
                intercepted_t = None
                for obs in observations[start_t:]:
                    if obs.band_id == band_id and obs.detected and not obs.is_false_alarm:
                        intercepted_t = obs.timestamp
                        break

                if intercepted_t is not None:
                    delays.append(float(intercepted_t - start_t))

        return delays
