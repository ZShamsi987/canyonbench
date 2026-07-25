from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import canyonbench.pipeline.clips as clips_module
from canyonbench.exceptions import DataValidationError, ExternalToolError
from canyonbench.pipeline.clips import _probe, inventory_clips
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
        lambda path, ffprobe: {"duration_s": 10.0, "creation_time": None},
    )
    frame = inventory_clips(tmp_path)
    assert set(frame["clip"]) == {"clip2.avi", "clip10.avi"}
    assert frame.video_start_s.tolist() == [0, 10]
    explicit = inventory_clips(tmp_path, order_by="filename", workers=2)
    assert explicit["clip"].tolist() == ["clip2.avi", "clip10.avi"]
    assert explicit.order_source.unique().tolist() == ["filename_relative_sequence"]
    assert explicit.timeline_source.unique().tolist() == ["contiguous"]
    monkeypatch.setattr(clips_module.shutil, "which", lambda _: None)
    with pytest.raises(ExternalToolError, match="ffprobe"):
        inventory_clips(tmp_path)


def test_clip_inventory_can_use_relative_clip_end_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "clip1.avi"
    second = tmp_path / "clip2.avi"
    first.touch()
    second.touch()
    os.utime(first, (1_000, 1_000))
    os.utime(second, (1_012, 1_012))
    monkeypatch.setattr(clips_module.shutil, "which", lambda _: "/fake/ffprobe")
    monkeypatch.setattr(
        clips_module,
        "_probe",
        lambda path, ffprobe: {
            "duration_s": 10.0 if path == first else 8.0,
            "creation_time": None,
        },
    )

    frame = inventory_clips(
        tmp_path,
        order_by="filename",
        timeline_by="relative_mtime_end",
    )

    assert frame.video_start_s.tolist() == [0, 14]
    assert frame.video_end_s.tolist() == [10, 22]
    assert frame.timeline_source.unique().tolist() == ["relative_mtime_end"]


def test_probe_uses_last_decodable_frame_in_preallocated_avi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "streams": [{"r_frame_rate": "30/1", "width": 1920, "height": 1080}],
        "frames": [
            {"best_effort_timestamp_time": "55.000000"},
            {"best_effort_timestamp_time": "60.966667"},
        ],
        "format": {"duration": "60.000000", "tags": {}},
    }
    completed = subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")
    monkeypatch.setattr(
        "canyonbench.pipeline.clips.subprocess.run", lambda *args, **kwargs: completed
    )

    metadata = _probe(tmp_path / "preallocated.avi", "/fake/ffprobe")

    assert metadata["declared_duration_s"] == 60
    assert metadata["duration_s"] == pytest.approx(61, abs=1e-6)
    assert metadata["last_frame_pts_s"] == pytest.approx(60.966667)


def test_inventory_can_audit_an_undecodable_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "clip1.avi").touch()
    (tmp_path / "clip2.avi").touch()
    monkeypatch.setattr(clips_module.shutil, "which", lambda _: "/fake/ffprobe")

    def fake_probe(path: Path, ffprobe: str) -> dict[str, object]:
        if path.name == "clip2.avi":
            raise ExternalToolError("zero-filled")
        return {"duration_s": 1.0, "creation_time": None}

    monkeypatch.setattr(clips_module, "_probe", fake_probe)
    with pytest.raises(ExternalToolError, match="zero-filled"):
        inventory_clips(tmp_path)

    frame = inventory_clips(tmp_path, exclude_undecodable=True)
    assert frame["clip"].tolist() == ["clip1.avi"]
    assert frame.attrs["excluded_clips"][0]["clip"] == "clip2.avi"


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


def test_extract_resume_and_hardlink_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    anchor = SyncAnchor(source.name, 0, 1, "test", 0, 1)
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool, timeout: int) -> None:
        assert check
        assert timeout == 900
        calls.append(command)
        Path(command[-1].replace("%010d", "0000000001")).write_bytes(b"frame")

    monkeypatch.setattr("canyonbench.pipeline.extract.shutil.which", lambda _: "/fake/ffmpeg")
    monkeypatch.setattr("canyonbench.pipeline.extract.subprocess.run", fake_run)
    output = tmp_path / "extracted"
    extract_clips(clips, anchor, output, execute=True, resume=True)
    extract_clips(clips, anchor, output, execute=True, resume=True)
    assert len(calls) == 1
    checksum_manifest = tmp_path / "source-checksums.json"
    extract_clips(
        clips,
        anchor,
        output,
        execute=True,
        resume=True,
        checksum_manifest=checksum_manifest,
    )
    assert len(calls) == 1
    checksum_payload = json.loads(checksum_manifest.read_text(encoding="utf-8"))
    assert checksum_payload["clips"][0]["sha256"] == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )

    plan = plan_frame_names(output, clips, anchor)
    named = tmp_path / "named"
    materialize_frame_names(plan, named, mode="hardlink")
    assert (named / "img_000001.jpg").samefile(output / "clip_0000_0000000001.jpg")


def test_extract_resume_invalidates_changed_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    anchor = SyncAnchor(source.name, 0, 1, "test", 0, 1)
    calls = 0

    def fake_run(command: list[str], *, check: bool, timeout: int) -> None:
        nonlocal calls
        calls += 1
        Path(command[-1].replace("%010d", "0000000001")).write_bytes(b"frame")

    monkeypatch.setattr("canyonbench.pipeline.extract.shutil.which", lambda _: "/fake/ffmpeg")
    monkeypatch.setattr("canyonbench.pipeline.extract.subprocess.run", fake_run)
    output = tmp_path / "extracted"
    extract_clips(clips, anchor, output, execute=True, resume=True)
    stale = output / "clip_0000_0000000999.jpg"
    stale.write_bytes(b"stale")
    clips.loc[0, "video_start_s"] = 2

    extract_clips(clips, anchor, output, execute=True, resume=True)

    assert calls == 2
    assert not stale.exists()


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
    dropped = build_frames_table(
        images,
        flight,
        add_quality_controls=False,
        drop_unmatched=True,
    )
    assert dropped.empty
    assert (
        candidate_exclusion_reason(
            pd.Series({"cloud": "heavy", "clarity": "clear", "balloon": "none"})
        )
        == "cloud_heavy"
    )
