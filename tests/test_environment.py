"""
Unit tests for RF Environment Simulator and Emitter Models.
"""

import pytest
import numpy as np

from smart_ew.environment.schema import SimConfig, EmitterConfig
from smart_ew.environment.simulator import RFEnvironment
from smart_ew.emitters.continuous import ContinuousEmitter
from smart_ew.emitters.intermittent import IntermittentEmitter


def test_sim_config_and_bands():
    config = SimConfig(num_bands=5, f_start_mhz=1000.0, f_step_mhz=100.0)
    env = RFEnvironment(config)

    assert len(env.bands) == 5
    assert env.bands[0].f_center_mhz == 1000.0
    assert env.bands[4].f_center_mhz == 1400.0


def test_continuous_emitter():
    config = EmitterConfig(
        emitter_id="E_CONT_1",
        emitter_type="continuous",
        assigned_bands=[1, 3],
        signal_power_dbm=-40.0,
        parameters={"start_time": 5, "end_time": 15},
    )
    emitter = ContinuousEmitter(config)

    assert not emitter.is_active(0, 1)
    assert not emitter.is_active(4, 1)
    assert emitter.is_active(5, 1)
    assert emitter.is_active(10, 3)
    assert not emitter.is_active(10, 2)
    assert not emitter.is_active(16, 1)
    assert emitter.get_signal_power_dbm(10, 1) == -40.0


def test_intermittent_emitter_reproducibility():
    config = EmitterConfig(
        emitter_id="E_INT_1",
        emitter_type="intermittent",
        assigned_bands=[2],
        signal_power_dbm=-50.0,
        parameters={"p_on": 0.3, "p_off": 0.5, "initial_active": True},
    )

    e1 = IntermittentEmitter(config, seed=123)
    states1 = [e1.is_active(t, 2) for t in range(50)]

    e2 = IntermittentEmitter(config, seed=123)
    states2 = [e2.is_active(t, 2) for t in range(50)]

    assert states1 == states2
    assert any(states1)  # At least some time steps active
    assert not all(states1)  # At least some time steps quiet


def test_rf_environment_matrix_generation():
    config = SimConfig(num_bands=4, total_time_steps=20)
    env = RFEnvironment(config)

    e_cont = ContinuousEmitter(
        EmitterConfig(
            emitter_id="C1",
            emitter_type="continuous",
            assigned_bands=[0],
            signal_power_dbm=-30.0,
        )
    )
    e_int = IntermittentEmitter(
        EmitterConfig(
            emitter_id="I1",
            emitter_type="intermittent",
            assigned_bands=[2],
            signal_power_dbm=-45.0,
            parameters={"p_on": 0.4, "p_off": 0.4},
        ),
        seed=42,
    )

    env.add_emitter(e_cont)
    env.add_emitter(e_int)

    matrix = env.get_ground_truth_matrix(total_steps=20)
    assert matrix.shape == (4, 20)
    # Band 0 should be active across all steps
    assert np.all(matrix[0, :] == 1)
    # Band 1 should be completely quiet
    assert np.all(matrix[1, :] == 0)


def test_power_accumulation():
    config = SimConfig(num_bands=2, noise_floor_dbm=-100.0)
    env = RFEnvironment(config)

    e1 = ContinuousEmitter(
        EmitterConfig(
            emitter_id="P1",
            emitter_type="continuous",
            assigned_bands=[0],
            signal_power_dbm=-50.0,
        )
    )
    e2 = ContinuousEmitter(
        EmitterConfig(
            emitter_id="P2",
            emitter_type="continuous",
            assigned_bands=[0],
            signal_power_dbm=-50.0,
        )
    )

    env.add_emitter(e1)
    env.add_emitter(e2)

    frame = env.get_ground_truth(0)
    # Two -50 dBm signals combined = 10*log10(10^-5 + 10^-5) = -50 + 3.01 = -46.99 dBm
    expected_power = 10.0 * np.log10(2 * (10.0 ** (-50.0 / 10.0)))
    assert pytest.approx(frame.signal_power_dbm[0], abs=0.1) == expected_power
