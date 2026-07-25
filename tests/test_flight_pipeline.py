from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from canyonbench.exceptions import DataValidationError
from canyonbench.pipeline.flight_log import audit_flight_segments, recover_operational_flight
from canyonbench.pipeline.sync import compute_anchor


def test_recover_operational_flight_selects_long_full_segment(tmp_path: Path) -> None:
    log = tmp_path / "WORLD10.txt"
    log.write_text(
        "device preamble\n"
        "Flight Phase,Elapsed Time,Latitude,Longitude,Altitude\n"
        "Launching,0,36.7,-111.1,100\n"
        "Floating,1,36.7,-111.1,200\n"
        "Terminating,2,36.7,-111.1,100\n"
        "Launching,0,36.8,-111.2,1000\n"
        "Flight Phase,Elapsed Time,Latitude,Longitude,Altitude\n"
        "Launching,2742,0,0,1200\n"
        "Floating,6806,36.81,-111.21,23000\n"
        "Floating,6807,36.82,-111.22,23010\n"
        "Terminating,25688,36.83,-111.23,22000\n",
        encoding="utf-8",
    )
    result = recover_operational_flight(log)
    assert result.elapsed_s.tolist() == [0, 6806, 6807, 25688]
    assert result.phase.tolist() == ["Launching", "Floating", "Floating", "Terminating"]
    assert result.lat.min() > 0
    audit = audit_flight_segments(log)
    assert len(audit) == 2
    assert audit["selected_operational"].tolist() == [False, True]
    assert audit["has_operational_sequence"].tolist() == [True, True]


def test_recover_operational_flight_accepts_worldview_prefixed_headers(
    tmp_path: Path,
) -> None:
    log = tmp_path / "world10.txt"
    log.write_text(
        "Flight Phase,Elapsed Time,Packets,WV Time,WV Latitude,WV Longitude,WV Altitude,"
        "WV Speed,WV Heading,WV Velocity Down,WV Pressure,WV Temperature\n"
        "Ground,0,1,946684800,0,0,0,0,0,0,12,20\n"
        "Launching,1,2,946684801,36.9,-111.4,1300,3,90,-3,11,19\n"
        "Floating,2,3,946684802,36.8,-111.5,1400,4,100,-4,10,18\n"
        "Terminating,3,4,946684803,36.7,-111.6,1350,5,110,5,11,19\n",
        encoding="utf-8",
    )

    result = recover_operational_flight(log)

    assert result[["phase", "elapsed_s", "lat", "lon", "alt_m"]].to_dict("records") == [
        {
            "phase": "Launching",
            "elapsed_s": 1,
            "lat": 36.9,
            "lon": -111.4,
            "alt_m": 1300,
        },
        {
            "phase": "Floating",
            "elapsed_s": 2,
            "lat": 36.8,
            "lon": -111.5,
            "alt_m": 1400,
        },
        {
            "phase": "Terminating",
            "elapsed_s": 3,
            "lat": 36.7,
            "lon": -111.6,
            "alt_m": 1350,
        },
    ]
    weather_columns = result[
        ["speed", "heading", "velocity_down", "pressure", "temperature"]
    ].columns.tolist()
    assert weather_columns == [
        "speed",
        "heading",
        "velocity_down",
        "pressure",
        "temperature",
    ]


def test_recover_requires_full_phase_sequence(tmp_path: Path) -> None:
    log = tmp_path / "bad.txt"
    log.write_text(
        "Flight Phase,Elapsed Time,Latitude,Longitude,Altitude\n"
        "Launching,1,36,-111,100\nFloating,2,36,-111,200\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="No contiguous"):
        recover_operational_flight(log)


def test_compute_anchor_uses_global_clip_clock() -> None:
    clips = pd.DataFrame(
        [
            {"clip": "a.avi", "duration_s": 100, "video_start_s": 0},
            {"clip": "b.avi", "duration_s": 100, "video_start_s": 100},
        ]
    )
    anchor = compute_anchor(clips, "b.avi", 20, 2742, "launch")
    assert anchor.video_elapsed_s == 120
    assert anchor.flight_offset_s == 2622


def test_compute_anchor_rejects_out_of_clip_offset() -> None:
    clips = pd.DataFrame([{"clip": "a.avi", "duration_s": 5, "video_start_s": 0}])
    with pytest.raises(DataValidationError, match="outside"):
        compute_anchor(clips, "a.avi", 7, 10, "event")
