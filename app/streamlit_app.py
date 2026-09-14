import os
import sys

# Ensure src directory is on sys.path regardless of launch CWD
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from smart_ew.scheduling.base import BaseScheduler

from smart_ew.environment.schema import SimConfig, EmitterConfig
from smart_ew.environment.simulator import RFEnvironment
from smart_ew.emitters.continuous import ContinuousEmitter
from smart_ew.emitters.intermittent import IntermittentEmitter
from smart_ew.emitters.periodic import PeriodicEmitter
from smart_ew.emitters.agile import FrequencyAgileEmitter
from smart_ew.receiver.schema import ReceiverConfig
from smart_ew.receiver.model import ESReceiver
from smart_ew.scheduling.sequential import SequentialScheduler
from smart_ew.scheduling.random_scan import RandomScheduler
from smart_ew.scheduling.ml_greedy import MLGreedyScheduler
from smart_ew.scheduling.contextual_bandit import LinUCBScheduler
from smart_ew.scheduling.uncertainty_aware import UncertaintyAwareScheduler
from smart_ew.scheduling.change_adaptive import ChangeAdaptiveScheduler
from smart_ew.evaluation.runner import SimulationRunner


st.set_page_config(
    page_title="Smart Scan EW Scheduler",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Theme Aesthetics)
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E2640 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    .metric-title {
        color: #94A3B8;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        color: #38BDF8;
        font-size: 26px;
        font-weight: 700;
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def build_scenario_environment(
    scenario_name: str, num_bands: int, total_steps: int, noise_floor: float
) -> RFEnvironment:
    config = SimConfig(
        num_bands=num_bands,
        total_time_steps=total_steps,
        noise_floor_dbm=noise_floor,
    )
    env = RFEnvironment(config)

    if scenario_name == "Scenario 1: Static Continuous Emitters":
        env.add_emitter(
            ContinuousEmitter(
                EmitterConfig(
                    emitter_id="C1",
                    emitter_type="continuous",
                    assigned_bands=[1],
                    signal_power_dbm=-50.0,
                )
            )
        )
        if num_bands > 5:
            env.add_emitter(
                ContinuousEmitter(
                    EmitterConfig(
                        emitter_id="C2",
                        emitter_type="continuous",
                        assigned_bands=[num_bands - 2],
                        signal_power_dbm=-55.0,
                    )
                )
            )

    elif scenario_name == "Scenario 2: Intermittent Poisson Emitters":
        env.add_emitter(
            IntermittentEmitter(
                EmitterConfig(
                    emitter_id="I1",
                    emitter_type="intermittent",
                    assigned_bands=[1],
                    signal_power_dbm=-50.0,
                    parameters={"p_on": 0.3, "p_off": 0.4, "initial_active": True},
                ),
                seed=42,
            )
        )
        env.add_emitter(
            IntermittentEmitter(
                EmitterConfig(
                    emitter_id="I2",
                    emitter_type="intermittent",
                    assigned_bands=[max(0, num_bands - 3)],
                    signal_power_dbm=-55.0,
                    parameters={"p_on": 0.2, "p_off": 0.5, "initial_active": False},
                ),
                seed=123,
            )
        )

    elif scenario_name == "Scenario 3: Periodic Radar Scanning Emitter":
        env.add_emitter(
            PeriodicEmitter(
                EmitterConfig(
                    emitter_id="RADAR_1",
                    emitter_type="periodic",
                    assigned_bands=[2],
                    signal_power_dbm=-48.0,
                    parameters={"period": 10, "duty_duration": 2},
                )
            )
        )

    elif scenario_name == "Scenario 4: Frequency-Agile Hopping Emitter":
        env.add_emitter(
            FrequencyAgileEmitter(
                EmitterConfig(
                    emitter_id="HOPPER_1",
                    emitter_type="agile",
                    assigned_bands=[0, 2, max(0, num_bands - 1)],
                    signal_power_dbm=-50.0,
                    parameters={"dwell_time": 3, "hop_sequence": [0, 2, max(0, num_bands - 1)]},
                ),
                seed=42,
            )
        )

    elif scenario_name == "Scenario 5: Mid-Simulation Environment Shift":
        env.add_emitter(
            ContinuousEmitter(
                EmitterConfig(
                    emitter_id="E1_EARLY",
                    emitter_type="continuous",
                    assigned_bands=[0],
                    signal_power_dbm=-50.0,
                    parameters={"start_time": 0, "end_time": total_steps // 2},
                )
            )
        )
        env.add_emitter(
            ContinuousEmitter(
                EmitterConfig(
                    emitter_id="E2_LATE",
                    emitter_type="continuous",
                    assigned_bands=[max(0, num_bands - 1)],
                    signal_power_dbm=-50.0,
                    parameters={"start_time": total_steps // 2, "end_time": total_steps},
                )
            )
        )

    return env


def get_scheduler(
    scheduler_name: str, num_bands: int, seed: int
) -> BaseScheduler:
    if scheduler_name == "Sequential Sweep Baseline":
        return SequentialScheduler(num_bands=num_bands)
    elif scheduler_name == "Uniform Random Baseline":
        return RandomScheduler(num_bands=num_bands, seed=seed)
    elif scheduler_name == "ML-Greedy Baseline":
        return MLGreedyScheduler(num_bands=num_bands, seed=seed)
    elif scheduler_name == "LinUCB Contextual Bandit":
        return LinUCBScheduler(num_bands=num_bands, alpha=0.5, seed=seed)
    elif scheduler_name == "Uncertainty-Aware Scheduler":
        return UncertaintyAwareScheduler(num_bands=num_bands, seed=seed)
    elif scheduler_name == "Change-Adaptive Scheduler (Proposed)":
        return ChangeAdaptiveScheduler(num_bands=num_bands, seed=seed)
    else:
        return SequentialScheduler(num_bands=num_bands)


# Title Header
st.title("📡 Smart Scan Strategy for Electronic Warfare")
st.caption("Adaptive ES Receiver Scheduler Simulation & Machine Learning Benchmarking")

st.sidebar.header("🕹️ Simulation Settings")

scenario = st.sidebar.selectbox(
    "RF Environment Scenario",
    [
        "Scenario 1: Static Continuous Emitters",
        "Scenario 2: Intermittent Poisson Emitters",
        "Scenario 3: Periodic Radar Scanning Emitter",
        "Scenario 4: Frequency-Agile Hopping Emitter",
        "Scenario 5: Mid-Simulation Environment Shift",
    ],
)

scheduler_choice = st.sidebar.selectbox(
    "Receiver Scheduler Strategy",
    [
        "Sequential Sweep Baseline",
        "Uniform Random Baseline",
        "ML-Greedy Baseline",
        "LinUCB Contextual Bandit",
        "Uncertainty-Aware Scheduler",
        "Change-Adaptive Scheduler (Proposed)",
    ],
)

st.sidebar.subheader("🎛️ Spectrum & Receiver Specs")
num_bands = st.sidebar.slider("Number of Frequency Bands", 5, 20, 10)
total_steps = st.sidebar.slider("Simulation Time Horizon (Steps)", 30, 200, 100)
noise_floor = st.sidebar.number_input("Noise Floor (dBm)", -120.0, -80.0, -100.0)
snr_threshold = st.sidebar.number_input("SNR Threshold (dB)", 5.0, 25.0, 10.0)
retuning_cost = st.sidebar.number_input("Retuning Delay Cost", 0.0, 3.0, 1.0)
seed = st.sidebar.number_input("Random Seed", 1, 9999, 42)

st.sidebar.markdown("---")
debug_mode = st.sidebar.checkbox("🔍 Debug Mode (Show Ground Truth Spectrum)", value=False)


# Run Simulation
env = build_scenario_environment(scenario, num_bands, total_steps, noise_floor)
rx = ESReceiver(
    ReceiverConfig(
        noise_floor_dbm=noise_floor,
        snr_threshold_db=snr_threshold,
        retuning_time_cost=retuning_cost,
    ),
    seed=seed,
)
scheduler = get_scheduler(scheduler_choice, num_bands, seed)
runner = SimulationRunner()

metrics = runner.run(env, rx, scheduler, total_steps=total_steps)

# KPI Cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-title">Prob of Detection</div>
            <div class="metric-value">{metrics.probability_of_detection * 100:.1f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-title">False Alarm Rate</div>
            <div class="metric-value">{metrics.probability_of_false_alarm * 100:.1f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-title">Avg Intercept Delay</div>
            <div class="metric-value">{metrics.average_intercept_time:.1f} steps</div>
        </div>""",
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-title">Spectrum Coverage</div>
            <div class="metric-value">{metrics.spectrum_coverage_ratio * 100:.0f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )
with col5:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-title">Total Retuning Cost</div>
            <div class="metric-value">{metrics.total_retuning_cost:.1f}</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Waterfall Charts
st.subheader("📊 Spectrum Observation Waterfall Map")

# Matrix construction for observations
obs_matrix = np.zeros((num_bands, total_steps))
for obs in metrics.observations:
    if obs.detected and not obs.is_false_alarm:
        obs_matrix[obs.band_id, obs.timestamp] = 2.0  # Hit
    elif obs.is_false_alarm:
        obs_matrix[obs.band_id, obs.timestamp] = 1.0  # False alarm
    else:
        obs_matrix[obs.band_id, obs.timestamp] = 0.5  # Scanned Quiet

fig_obs = px.imshow(
    obs_matrix,
    labels=dict(x="Time Step", y="Frequency Band", color="Observation State"),
    x=list(range(total_steps)),
    y=[f"Band {b}" for b in range(num_bands)],
    color_continuous_scale=[[0.0, "#0F172A"], [0.25, "#334155"], [0.5, "#F59E0B"], [1.0, "#10B981"]],
    aspect="auto",
    title=f"Receiver Scans ({scheduler_choice}) - Green: Hit, Yellow: False Alarm, Dark: Quiet/Unobserved",
)
fig_obs.update_layout(template="plotly_dark", height=300)
st.plotly_chart(fig_obs, use_container_width=True)

if debug_mode:
    st.markdown("---")
    st.subheader("🔬 Ground Truth Matrix (Debug View Only)")
    gt_matrix = env.get_ground_truth_matrix(total_steps=total_steps)
    fig_gt = px.imshow(
        gt_matrix,
        labels=dict(x="Time Step", y="Frequency Band", color="Active"),
        x=list(range(total_steps)),
        y=[f"Band {b}" for b in range(num_bands)],
        color_continuous_scale=[[0.0, "#0F172A"], [1.0, "#EF4444"]],
        aspect="auto",
        title="Hidden Environment Ground Truth Matrix (Red: Transmitting)",
    )
    fig_gt.update_layout(template="plotly_dark", height=250)
    st.plotly_chart(fig_gt, use_container_width=True)

# Performance Breakdown
st.markdown("---")
st.subheader("📈 Cumulative Intercept Performance Comparison")

# Run baselines for comparison chart
seq_metrics = SimulationRunner().run(env, rx, SequentialScheduler(num_bands), total_steps)
rand_metrics = SimulationRunner().run(env, rx, RandomScheduler(num_bands, seed=seed), total_steps)

cum_hits_selected = np.cumsum([1 if o.detected and not o.is_false_alarm else 0 for o in metrics.observations])
cum_hits_seq = np.cumsum([1 if o.detected and not o.is_false_alarm else 0 for o in seq_metrics.observations])
cum_hits_rand = np.cumsum([1 if o.detected and not o.is_false_alarm else 0 for o in rand_metrics.observations])

df_perf = pd.DataFrame(
    {
        "Time Step": list(range(total_steps)),
        f"Selected ({scheduler_choice})": cum_hits_selected,
        "Sequential Sweep": cum_hits_seq,
        "Uniform Random": cum_hits_rand,
    }
)

fig_line = px.line(
    df_perf,
    x="Time Step",
    y=[f"Selected ({scheduler_choice})", "Sequential Sweep", "Uniform Random"],
    title="Cumulative Intercepts Over Time",
    color_discrete_sequence=["#38BDF8", "#F59E0B", "#94A3B8"],
)
fig_line.update_layout(template="plotly_dark", height=350, yaxis_title="Cumulative Intercepted Transmissions")
st.plotly_chart(fig_line, use_container_width=True)
