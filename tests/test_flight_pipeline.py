from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from canyonbench.exceptions import DataValidationError
from canyonbench.pipeline.flight_log import recover_operational_flight
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
