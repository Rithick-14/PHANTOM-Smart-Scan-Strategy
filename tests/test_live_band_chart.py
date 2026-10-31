from __future__ import annotations

import pandas as pd

from app.live_band_chart import build_live_band_time_figure


def test_live_band_figure_sorts_scan_rows_and_marks_observation_types():
    log_df = pd.DataFrame(
        [
            {"Step": 2, "Band": 1, "Frequency Range (GHz)": "1.05-1.10", "Receiver Result": "FALSE ALARM"},
            {"Step": 0, "Band": 3, "Frequency Range (GHz)": "1.15-1.20", "Receiver Result": "DETECTION"},
            {"Step": 1, "Band": 2, "Frequency Range (GHz)": "1.10-1.15", "Receiver Result": "QUIET"},
            {"Step": 3, "Band": 0, "Frequency Range (GHz)": "1.00-1.05", "Receiver Result": "SIMULATED CONTINUOUS EMITTER ADDED"},
        ]
    )

    figure = build_live_band_time_figure(log_df, num_bands=4, total_steps=10)

    scan_path = next(trace for trace in figure.data if trace.name == "Scan path")
    assert list(scan_path.x) == [0, 1, 2]
    assert list(scan_path.y) == [3, 2, 1]
    assert list(next(trace for trace in figure.data if trace.name == "Detection").x) == [0]
    assert list(next(trace for trace in figure.data if trace.name == "Quiet scan").x) == [1]
    assert list(next(trace for trace in figure.data if trace.name == "False alarm").x) == [2]
    assert "Band 3 (1.15-1.20 GHz)" in list(figure.layout.yaxis.ticktext)


def test_live_band_figure_handles_empty_and_incomplete_logs():
    for log_df in (pd.DataFrame(), pd.DataFrame({"Step": [0]})):
        figure = build_live_band_time_figure(log_df, num_bands=4, total_steps=10)
        assert len(figure.data) == 0
        assert "Waiting for the first receiver scan" in figure.layout.annotations[0].text
        assert list(figure.layout.yaxis.tickvals) == [0, 1, 2, 3]
