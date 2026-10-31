"""Persistent, software-only live simulation view for the Streamlit dashboard."""

from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from live_band_chart import render_live_band_time_chart

from smart_ew.emitters.agile import FrequencyAgileEmitter
from smart_ew.emitters.continuous import ContinuousEmitter
from smart_ew.emitters.intermittent import IntermittentEmitter
from smart_ew.emitters.periodic import PeriodicEmitter
from smart_ew.environment.schema import EmitterConfig
from smart_ew.receiver.model import ESReceiver
from smart_ew.receiver.schema import ReceiverConfig


def _new_run(
    scenario: str,
    scheduler_name: str,
    num_bands: int,
    total_steps: int,
    noise_floor: float,
    snr_threshold: float,
    retuning_cost: float,
    seed: int,
    build_environment: Callable,
    get_scheduler: Callable,
) -> dict:
    environment = build_environment(scenario, num_bands, total_steps, noise_floor)
    receiver = ESReceiver(
        ReceiverConfig(
            noise_floor_dbm=noise_floor,
            snr_threshold_db=snr_threshold,
            retuning_time_cost=retuning_cost,
            noise_std_db=2.5,
        ),
        seed=seed,
    )
    scheduler = get_scheduler(scheduler_name, num_bands, seed)
    return {
        "environment": environment,
        "receiver": receiver,
        "scheduler": scheduler,
        "observations": [],
        "truth_active": [],
        "events": [],
        "running": False,
        "signature": (
            scenario,
            scheduler_name,
            num_bands,
            total_steps,
            noise_floor,
            snr_threshold,
            retuning_cost,
            seed,
        ),
    }


def _band_range(band_id: int) -> str:
    low = 1.0 + band_id * 0.05
    high = low + 0.05
    return f"{low:.2f}-{high:.2f}"


def _advance(run: dict, total_steps: int) -> None:
    environment = run["environment"]
    t = environment.current_time_step
    if t >= total_steps:
        run["running"] = False
        return

    frame = environment.step()
    band = run["scheduler"].select_next_band(t)
    observation = run["receiver"].scan(band, t, frame)
    run["scheduler"].update(observation)
    active = bool(frame.active_bands.get(band, False))
    run["observations"].append(observation)
    run["truth_active"].append(active)
    run["events"].append(
        {
            "Step": t,
            "Band": band,
            "Frequency Range (GHz)": _band_range(band),
            "Receiver Result": (
                "FALSE ALARM"
                if observation.is_false_alarm
                else "DETECTION"
                if observation.detected
                else "QUIET"
            ),
            "Measured Power (dBm)": "-",
            "Estimated SNR (dB)": "-",
        }
    )
    if environment.current_time_step >= total_steps:
        run["running"] = False


def _inject_emitter(
    run: dict, kind: str, band: int, power: float, num_bands: int, seed: int
) -> str:
    environment = run["environment"]
    start = environment.current_time_step
    emitter_id = f"LIVE-{sum(e.emitter_id.startswith('LIVE-') for e in environment.emitters) + 1}"
    params = {"start_time": start}
    assigned_bands = [band]

    if kind == "Intermittent":
        params.update({"p_on": 0.3, "p_off": 0.35, "initial_active": True})
        emitter = IntermittentEmitter(
            EmitterConfig(emitter_id, "intermittent", assigned_bands, power, params),
            seed=seed + start,
        )
    elif kind == "Periodic":
        params.update({"period": 8, "duty_duration": 2})
        emitter = PeriodicEmitter(
            EmitterConfig(emitter_id, "periodic", assigned_bands, power, params)
        )
    elif kind == "Frequency hopping":
        next_band = (band + max(1, num_bands // 3)) % num_bands
        assigned_bands = [band, next_band]
        params.update({"dwell_time": 2, "hop_sequence": assigned_bands})
        emitter = FrequencyAgileEmitter(
            EmitterConfig(emitter_id, "agile", assigned_bands, power, params),
            seed=seed + start,
        )
    else:
        emitter = ContinuousEmitter(
            EmitterConfig(emitter_id, "continuous", assigned_bands, power, params)
        )

    environment.add_emitter(emitter)
    run["events"].append(
        {
            "Step": start,
            "Band": band,
            "Frequency Range (GHz)": _band_range(band),
            "Receiver Result": f"SIMULATED {kind.upper()} EMITTER ADDED",
            "Measured Power (dBm)": "-",
            "Estimated SNR (dB)": "-",
        }
    )
    return emitter_id


def _render_live_panel(
    run: dict, total_steps: int, num_bands: int, noise_floor: float, seed: int
) -> None:
    @st.fragment(run_every="1s")
    def live_panel() -> None:
        if run["running"]:
            _advance(run, total_steps)

        st.subheader("Live scan controls")
        st.caption(
            "Software simulation only. Each timer tick advances one receiver scan; "
            "added emitters begin at the next simulation step."
        )
        start_col, pause_col, step_col, reset_col = st.columns(4)
        with start_col:
            if st.button("Start / Resume", use_container_width=True, key="live_start"):
                run["running"] = run["environment"].current_time_step < total_steps
        with pause_col:
            if st.button("Pause", use_container_width=True, key="live_pause"):
                run["running"] = False
        with step_col:
            if st.button("Step once", use_container_width=True, key="live_step"):
                run["running"] = False
                _advance(run, total_steps)
        with reset_col:
            if st.button("Reset run", use_container_width=True, key="live_reset"):
                st.session_state.pop("phantom_live_run", None)
                st.rerun()

        t = run["environment"].current_time_step
        status = "RUNNING" if run["running"] else ("COMPLETE" if t >= total_steps else "PAUSED")
        st.write(
            f"**Status:** {status}　·　**Step:** {t}/{total_steps}　·　"
            f"**Scheduler:** {run['scheduler'].name}"
        )

        with st.expander("Inject a simulated emitter into this run", expanded=True):
            with st.form("live_inject_form", clear_on_submit=True):
                inject_col1, inject_col2, inject_col3 = st.columns(3)
                kind = inject_col1.selectbox(
                    "Emitter behavior",
                    ["Continuous", "Intermittent", "Periodic", "Frequency hopping"],
                )
                band = inject_col2.selectbox("Initial band", list(range(num_bands)))
                power = inject_col3.number_input(
                    "Signal power (dBm)",
                    min_value=float(noise_floor - 5),
                    max_value=float(noise_floor + 40),
                    value=float(noise_floor + 12),
                    step=1.0,
                )
                submitted = st.form_submit_button(
                    "Add emitter to active run", disabled=t >= total_steps
                )
            if submitted:
                emitter_id = _inject_emitter(
                    run, kind, int(band), float(power), num_bands, seed
                )
                st.success(
                    f"{emitter_id} added at step {t} on Band {band}. "
                    "It will affect subsequent simulated scans."
                )

        observations = run["observations"]
        if observations:
            correct_hits = sum(
                1
                for obs, active in zip(observations, run["truth_active"])
                if active and obs.detected and not obs.is_false_alarm
            )
            active_scans = sum(run["truth_active"])
            quiet_scans = len(observations) - active_scans
            false_alarms = sum(
                1
                for obs, active in zip(observations, run["truth_active"])
                if not active and (obs.detected or obs.is_false_alarm)
            )
            pd_value = f"{correct_hits / active_scans:.1%}" if active_scans else "—"
            far_value = f"{false_alarms / quiet_scans:.1%}" if quiet_scans else "—"
            metric_cols = st.columns(4)
            metric_cols[0].metric("Total observations", len(observations))
            metric_cols[1].metric("Detections", sum(obs.detected for obs in observations))
            metric_cols[2].metric("Pd · simulation evaluation", pd_value)
            metric_cols[3].metric("False alarm rate · evaluation", far_value)

            matrix = [[None] * t for _ in range(num_bands)]
            for obs in observations:
                matrix[obs.band_id][obs.timestamp] = (
                    2
                    if obs.detected and not obs.is_false_alarm
                    else 1
                    if obs.is_false_alarm
                    else 0
                )
            fig = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    x=list(range(t)),
                    y=[f"Band {b} · {_band_range(b)} GHz" for b in range(num_bands)],
                    zmin=0,
                    zmax=2,
                    colorscale=[[0, "#132235"], [0.5, "#d69e2e"], [1, "#24d6bd"]],
                    colorbar={
                        "title": "Receiver result",
                        "tickvals": [0, 1, 2],
                        "ticktext": ["Quiet", "False alarm", "Detection"],
                    },
                    hovertemplate="%{y}<br>Step %{x}<br>Result: %{z}<extra></extra>",
                )
            )
            fig.update_layout(
                title="Frequency Band Observation - receiver observations",
                template="plotly_dark",
                height=max(280, min(560, 30 * num_bands + 100)),
                xaxis_title="Time step",
                yaxis_title="Frequency band",
                margin={"l": 10, "r": 10, "t": 48, "b": 35},
            )
            st.plotly_chart(fig, use_container_width=True, key="live_observation_chart")

        log_df = pd.DataFrame(run["events"])
        if not log_df.empty:
            st.subheader("Live event log")
            st.dataframe(
                log_df.tail(100).iloc[::-1],
                use_container_width=True,
                hide_index=True,
            )
            csv = log_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Export live log as CSV",
                csv,
                file_name="phantom_live_simulation.csv",
                mime="text/csv",
                key="live_export_csv",
            )
        else:
            st.info("No scans yet. Start the simulation or advance one step.")

        render_live_band_time_chart(
            log_df, num_bands=num_bands, total_steps=total_steps
        )

    live_panel()


def render_live(
    scenario: str,
    scheduler_name: str,
    num_bands: int,
    total_steps: int,
    noise_floor: float,
    snr_threshold: float,
    retuning_cost: float,
    seed: int,
    build_environment: Callable,
    get_scheduler: Callable,
) -> None:
    signature = (
        scenario,
        scheduler_name,
        num_bands,
        total_steps,
        noise_floor,
        snr_threshold,
        retuning_cost,
        seed,
    )
    run = st.session_state.get("phantom_live_run")
    if run is None or run["signature"] != signature:
        run = _new_run(
            scenario,
            scheduler_name,
            num_bands,
            total_steps,
            noise_floor,
            snr_threshold,
            retuning_cost,
            seed,
            build_environment,
            get_scheduler,
        )
        st.session_state["phantom_live_run"] = run

    st.header("PHANTOM · Live Observation Center")
    st.caption("Persistent in-memory simulation run with runtime emitter injection")
    _render_live_panel(run, total_steps, num_bands, noise_floor, seed)
