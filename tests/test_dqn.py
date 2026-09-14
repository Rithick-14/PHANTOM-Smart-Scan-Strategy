"""
Unit tests for DQNScheduler.
"""

import pytest

from smart_ew.scheduling.dqn_agent import DQNScheduler
from smart_ew.receiver.schema import ReceiverObservation


def test_dqn_scheduler_forward_and_update():
    scheduler = DQNScheduler(num_bands=4, seed=42)

    action = scheduler.select_next_band(time_step=0)
    assert 0 <= action < 4

    obs = ReceiverObservation(
        band_id=action,
        timestamp=0,
        detected=True,
        measured_power_dbm=-60.0,
        estimated_snr_db=20.0,
        confidence=0.9,
        tuning_cost_paid=0.0,
        is_false_alarm=False,
    )

    # Update weights via SGD
    scheduler.update(obs)
    assert scheduler.epsilon < 0.2  # Epsilon decayed
