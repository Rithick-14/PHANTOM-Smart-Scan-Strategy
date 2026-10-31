"""Live Plotly chart for the receiver's observed band scan path."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from smart_ew.environment.schema import SimConfig


_RESULT_STYLES = {
    "DETECTION": ("Detection", "#24d6bd"),
    "FALSE ALARM": ("False alarm", "#f59e0b"),
    "QUIET": ("Quiet scan", "#94a3b8"),
}


def _band_range_labels(log_df: pd.DataFrame, num_bands: int) -> list[str]:
    """Build every Y tick label from the live log, with simulator defaults as fallback."""
    ranges: dict[int, str] = {}
    if {"Band", "Frequency Range (GHz)"}.issubset(log_df.columns):
        for band_value, range_value in zip(
            log_df["Band"], log_df["Frequency Range (GHz)"]
        ):
            band_number = pd.to_numeric(band_value, errors="coerce")
            if pd.isna(band_number) or pd.isna(range_value):
                continue
            text = str(range_value).strip()
            if text and text not in {"—", "nan", "None"}:
                ranges[int(band_number)] = text

    config = SimConfig()
    labels = []
    for band in range(num_bands):
        frequency = ranges.get(band)
        if frequency is None:
            low_mhz = config.f_start_mhz + band * config.f_step_mhz
            high_mhz = low_mhz + config.bandwidth_mhz
            frequency = f"{low_mhz / 1000:.2f}-{high_mhz / 1000:.2f} GHz"
        elif "ghz" not in frequency.lower():
            frequency = f"{frequency} GHz"
        labels.append(f"Band {band} ({frequency})")
    return labels


def _scan_rows(log_df: pd.DataFrame, num_bands: int) -> pd.DataFrame:
    """Keep actual scan observations, excluding non-scan log entries such as emitter adds."""
    required = {"Step", "Band", "Receiver Result"}
    if not isinstance(log_df, pd.DataFrame) or log_df.empty:
        return pd.DataFrame(
            columns=["Step", "Band", "Receiver Result", "Frequency Range (GHz)"]
        )
    if not required.issubset(log_df.columns):
        return pd.DataFrame(
            columns=["Step", "Band", "Receiver Result", "Frequency Range (GHz)"]
        )

    columns = ["Step", "Band", "Receiver Result"]
    if "Frequency Range (GHz)" in log_df.columns:
        columns.append("Frequency Range (GHz)")
    rows = log_df.loc[:, columns].copy()
    rows["Step"] = pd.to_numeric(rows["Step"], errors="coerce")
    rows["Band"] = pd.to_numeric(rows["Band"], errors="coerce")
    rows["_result"] = rows["Receiver Result"].fillna("").astype(str).str.strip().str.upper()
    valid = (
        rows["Step"].notna()
        & rows["Band"].notna()
        & rows["Band"].between(0, max(num_bands - 1, -1))
        & rows["_result"].isin(_RESULT_STYLES)
    )
    rows = rows.loc[valid].copy()
    if rows.empty:
        return rows
    rows["Step"] = rows["Step"].astype(int)
    rows["Band"] = rows["Band"].astype(int)
    return rows.sort_values("Step", kind="stable").reset_index(drop=True)


def build_live_band_time_figure(
    log_df: pd.DataFrame | None, num_bands: int, total_steps: int
) -> go.Figure:
    """Create a time-versus-band path from real receiver scan rows in the live event log."""
    band_count = max(0, int(num_bands))
    step_count = max(1, int(total_steps))
    source = log_df if isinstance(log_df, pd.DataFrame) else pd.DataFrame()
    rows = _scan_rows(source, band_count)
    labels = _band_range_labels(source, band_count)

    if "Frequency Range (GHz)" not in rows.columns:
        rows["Frequency Range (GHz)"] = pd.Series(dtype="object")
    fallback_labels = {
        band: label[label.find("(") + 1 : -1] for band, label in enumerate(labels)
    }
    rows["_frequency"] = rows["Band"].map(fallback_labels)
    if not rows.empty and "Frequency Range (GHz)" in rows.columns:
        logged_frequency = rows["Frequency Range (GHz)"].fillna("").astype(str).str.strip()
        rows["_frequency"] = logged_frequency.where(
            ~logged_frequency.isin(["", "—", "nan", "None"]),
            rows["_frequency"],
        )
        rows["_frequency"] = rows["_frequency"].map(
            lambda value: value if "ghz" in str(value).lower() else f"{value} GHz"
        )

    figure = go.Figure()
    if not rows.empty:
        figure.add_trace(
            go.Scatter(
                x=rows["Step"],
                y=rows["Band"],
                mode="lines",
                name="Scan path",
                line={"color": "#54738e", "width": 1.5},
                hoverinfo="skip",
                showlegend=False,
                connectgaps=False,
            )
        )
        for result, (legend_name, color) in _RESULT_STYLES.items():
            points = rows.loc[rows["_result"] == result]
            figure.add_trace(
                go.Scatter(
                    x=points["Step"],
                    y=points["Band"],
                    mode="markers",
                    name=legend_name,
                    marker={
                        "color": color,
                        "size": 9,
                        "line": {"color": "#0b1220", "width": 1},
                    },
                    customdata=points["_frequency"],
                    hovertemplate=(
                        "Step %{x}<br>Band %{y}<br>Frequency %{customdata}"
                        f"<br>Result: {legend_name}<extra></extra>"
                    ),
                )
            )

    max_step = max(step_count - 1, 1)
    figure.update_layout(
        title="Live Band vs Time",
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        height=390,
        margin={"l": 20, "r": 20, "t": 58, "b": 44},
        legend={
            "title": "Receiver observation",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        uirevision="live-band-time-chart",
        xaxis={
            "title": "Simulation step",
            "range": [0, max_step],
            "dtick": max(1, math.ceil(step_count / 10)),
            "showgrid": True,
            "gridcolor": "#263445",
            "zeroline": False,
        },
        yaxis={
            "title": "Frequency band",
            "range": [-0.5, max(band_count - 0.5, 0.5)],
            "tickmode": "array",
            "tickvals": list(range(band_count)),
            "ticktext": labels,
            "showgrid": True,
            "gridcolor": "#263445",
            "zeroline": False,
        },
    )

    if rows.empty:
        figure.add_annotation(
            text="Waiting for the first receiver scan",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": "#94a3b8", "size": 15},
        )
    return figure


def render_live_band_time_chart(
    log_df: pd.DataFrame | None, num_bands: int, total_steps: int
) -> None:
    """Render the live chart and make unscanned bands explicit to the reader."""
    figure = build_live_band_time_figure(log_df, num_bands, total_steps)
    has_scan_rows = any(trace.name == "Scan path" for trace in figure.data)
    if not has_scan_rows:
        st.info("Waiting for the first receiver scan. Unscanned bands have no plotted point.")
    st.caption(
        "Dots show scanned bands: teal = detection, amber = false alarm, gray = quiet scan. "
        "Bands without a point were not scanned at that step."
    )
    st.plotly_chart(figure, key="live_band_time_chart", width="stretch", theme=None)
