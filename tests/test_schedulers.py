"""
Unit tests for baseline scan schedulers.
"""

import pytest

from smart_ew.scheduling.sequential import SequentialScheduler
from smart_ew.scheduling.random_scan import RandomScheduler
from smart_ew.receiver.schema import ReceiverObservation


def test_sequential_scheduler_cycle():
    scheduler = SequentialScheduler(num_bands=4, start_band=0)

    actions = [scheduler.select_next_band(t) for t in range(10)]
    expected = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    assert actions == expected


def test_sequential_scheduler_reset():
    scheduler = SequentialScheduler(num_bands=3, start_band=1)

    a1 = [scheduler.select_next_band(t) for t in range(3)]
    assert a1 == [1, 2, 0]

    scheduler.reset()
    a2 = [scheduler.select_next_band(t) for t in range(3)]
    assert a2 == [1, 2, 0]


def test_random_scheduler_reproducibility():
    s1 = RandomScheduler(num_bands=10, seed=42)
    actions1 = [s1.select_next_band(t) for t in range(20)]

    s2 = RandomScheduler(num_bands=10, seed=42)
    actions2 = [s2.select_next_band(t) for t in range(20)]

    assert actions1 == actions2
    # Ensure all selected bands are within range [0, 9]
    assert all(0 <= a < 10 for a in actions1)


def test_scheduler_update_interface():
    scheduler = SequentialScheduler(num_bands=5)
    obs = ReceiverObservation(
        band_id=0,
        timestamp=0,
        detected=True,
        measured_power_dbm=-60.0,
        estimated_snr_db=20.0,
        confidence=0.9,
        tuning_cost_paid=0.0,
        is_false_alarm=False,
    )
    # Updating baseline open-loop scheduler should execute safely without exception
    scheduler.update(obs)
