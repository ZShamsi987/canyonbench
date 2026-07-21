from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import canyonbench.pipeline.clips as clips_module
from canyonbench.exceptions import DataValidationError, ExternalToolError
from canyonbench.pipeline.clips import inventory_clips
from canyonbench.pipeline.extract import extract_clips, extraction_command
from canyonbench.pipeline.join import (
    build_frames_table,
    candidate_exclusion_reason,
    discover_frames,
)
from canyonbench.pipeline.naming import materialize_frame_names, plan_frame_names
from canyonbench.pipeline.quality import image_quality_controls
from canyonbench.pipeline.sampling import add_perceptual_hashes
from canyonbench.pipeline.sync import SyncAnchor, load_anchor, save_anchor


def test_clip_inventory_and_missing_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "clip10.avi").touch()
    (tmp_path / "clip2.avi").touch()
    monkeypatch.setattr(clips_module.shutil, "which", lambda _: "/fake/ffprobe")
    monkeypatch.setattr(
        clips_module,
        "_probe",
        lambda path, _: {"duration_s": 10.0, "creation_time": None},
    )
    frame = inventory_clips(tmp_path)
    assert set(frame["clip"]) == {"clip2.avi", "clip10.avi"}
    assert frame.video_start_s.tolist() == [0, 10]
    monkeypatch.setattr(clips_module.shutil, "which", lambda _: None)
    with pytest.raises(ExternalToolError, match="ffprobe"):
        inventory_clips(tmp_path)


def test_extract_naming_join_and_quality(tmp_path: Path) -> None:
    source = tmp_path / "source.avi"
    source.touch()
    clips = pd.DataFrame(
        [
            {
                "clip": source.name,
                "path": str(source),
                "duration_s": 2,
                "clip_index": 0,
                "video_start_s": 0,
            }
        ]
    )
    anchor = SyncAnchor(source.name, 0, 6806, "float", 0, 6806)
    command = extraction_command(clips.iloc[0], anchor, tmp_path / "extracted")
    assert "crop=" in command[command.index("-vf") + 1]
    assert len(extract_clips(clips, anchor, tmp_path / "extracted", execute=False)) == 1

    extracted = tmp_path / "extracted"
    extracted.mkdir(exist_ok=True)
    for index, color in ((1, "brown"), (2, "green")):
        Image.new("RGB", (12, 8), color).save(extracted / f"clip_0000_{index:010d}.jpg")
    plan = plan_frame_names(extracted, clips, anchor)
    assert plan.image.tolist() == ["img_006806.jpg", "img_006807.jpg"]
    named = tmp_path / "named"
    materialize_frame_names(plan, named)
    assert (named / "img_006806.jpg").exists()
    assert (extracted / "clip_0000_0000000001.jpg").exists()  # source remains intact

    images = discover_frames(named)
    flight = pd.DataFrame(
        [
            {"elapsed_s": 6806, "phase": "Floating", "lat": 36.8, "lon": -111.5, "alt_m": 23000},
            {"elapsed_s": 6807, "phase": "Terminating", "lat": 36.8, "lon": -111.5, "alt_m": 22000},
        ]
    )
    joined = build_frames_table(images, flight)
    assert joined.image.tolist() == ["img_006806.jpg"]
    assert 0 <= joined.brightness_mean.iloc[0] <= 1
    assert len(add_perceptual_hashes(joined)) == 1
    assert image_quality_controls(named / "img_006806.jpg")["width_px"] == 12


def test_sync_roundtrip_and_join_failures(tmp_path: Path) -> None:
    anchor = SyncAnchor("a.avi", 1, 2742, "launch", 1, 2741)
    path = tmp_path / "sync.json"
    save_anchor(path, anchor)
    assert load_anchor(path) == anchor
    with pytest.raises(DataValidationError, match="No elapsed-second"):
        discover_frames(tmp_path / "empty")
    images = pd.DataFrame(
        [{"image": "img_000001.jpg", "image_path": "missing.jpg", "elapsed_s": 1}]
    )
    flight = pd.DataFrame(
        [{"elapsed_s": 2, "phase": "Floating", "lat": 36, "lon": -111, "alt_m": 1}]
    )
    with pytest.raises(DataValidationError, match="no matching"):
        build_frames_table(images, flight, add_quality_controls=False)
    assert (
        candidate_exclusion_reason(
            pd.Series({"cloud": "heavy", "clarity": "clear", "balloon": "none"})
        )
        == "cloud_heavy"
    )
