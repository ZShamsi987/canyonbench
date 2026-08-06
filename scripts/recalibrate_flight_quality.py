#!/usr/bin/env python3
"""Rebuild the real-flight calibration without retaining the World-X videos.

The camera clock is known to be invalid.  This script uses the audited clip
order and the verified video-to-flight offset instead, takes one *uncropped*
midpoint frame from each usable Launching/Floating clip, and immediately evicts
the cloud-provider copy after measuring it.  The 377-frame selection is a hard
contract: changing the clip audit, source timeline, or eligibility criteria
must be reviewed rather than silently producing a new calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from canyonbench.io import sha256_file, write_json
from canyonbench.pipeline.cloud_cache import evict_cloud_files
from canyonbench.trace.degradation import FrameQuality, measure_quality
from canyonbench.trace.schemas import QualityCalibration

ELIGIBLE_PHASES = {"Launching", "Floating"}
MINIMUM_CLIP_DURATION_S = 30.0
EXPECTED_FRAME_COUNT = 377


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid {key!r} in clip audit") from exc


def select_clips(clips_csv: Path, telemetry_csv: Path, sync_json: Path) -> list[dict[str, str]]:
    """Select the audited chronological, usable clips without camera timestamps."""

    with telemetry_csv.open(newline="", encoding="utf-8") as handle:
        telemetry = list(csv.DictReader(handle))
    eligible_times = [
        _float(row, "elapsed_s") for row in telemetry if row.get("phase") in ELIGIBLE_PHASES
    ]
    if not eligible_times:
        raise ValueError(f"no {sorted(ELIGIBLE_PHASES)} telemetry rows in {telemetry_csv}")
    with sync_json.open(encoding="utf-8") as handle:
        flight_offset_s = float(json.load(handle)["flight_offset_s"])
    lower, upper = min(eligible_times), max(eligible_times)

    with clips_csv.open(newline="", encoding="utf-8") as handle:
        clips = list(csv.DictReader(handle))
    selected = []
    for row in clips:
        start, end = _float(row, "video_start_s"), _float(row, "video_end_s")
        midpoint_flight_s = ((start + end) / 2.0) + flight_offset_s
        if (
            lower <= midpoint_flight_s <= upper
            and _float(row, "duration_s") >= MINIMUM_CLIP_DURATION_S
        ):
            selected.append(row)
    selected.sort(key=lambda row: int(row["clip_index"]))
    if len(selected) != EXPECTED_FRAME_COUNT:
        raise ValueError(
            f"expected {EXPECTED_FRAME_COUNT} usable chronological clips, found {len(selected)}; "
            "do not recalibrate until the clip/telemetry audit is reconciled"
        )
    return selected


def _extract_midpoint_frame(clip: dict[str, str], destination: Path) -> None:
    """Extract exactly one full-width source frame; intentionally no crop filter."""

    source = Path(clip["path"])
    midpoint_s = _float(clip, "duration_s") / 2.0
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{midpoint_s:.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-y",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True, timeout=300)
    finally:
        # The original clip stays in Drive; this returns a File Provider cache
        # range to cloud-only storage before the next clip can be materialized.
        evict_cloud_files([source])
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not produce a frame for {source}")


def calibrate(clips: list[dict[str, str]], output: Path, provenance: Path) -> QualityCalibration:
    """Measure one temporary full frame at a time, retaining no footage or frames."""

    rows: list[FrameQuality] = []
    hashes: list[str] = []
    provenance_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="canyonbench-quality-") as temporary:
        frame = Path(temporary) / "frame.jpg"
        for index, clip in enumerate(clips, start=1):
            _extract_midpoint_frame(clip, frame)
            with Image.open(frame) as source:
                rows.append(measure_quality(np.asarray(source.convert("RGB")), path=clip["clip"]))
            frame_hash = sha256_file(frame)
            hashes.append(frame_hash)
            provenance_rows.append(
                {
                    "ordinal": index,
                    "clip": clip["clip"],
                    "clip_index": int(clip["clip_index"]),
                    "video_midpoint_s": _float(clip, "duration_s") / 2.0,
                    "frame_sha256": frame_hash,
                }
            )
            frame.unlink()
            print(f"[{index}/{len(clips)}] calibrated {clip['clip']}", flush=True)

    fields = [field for field in FrameQuality.__dataclass_fields__ if field != "path"]
    quantiles = {
        field: {
            str(q): float(np.quantile([getattr(row, field) for row in rows], q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        }
        for field in fields
    }
    calibration = QualityCalibration(
        frame_count=len(rows), source_sha256=hashes, quantiles=quantiles
    )
    write_json(output, calibration.model_dump(mode="json"))
    write_json(
        provenance,
        {
            "schema_version": "1.0.0",
            "selection": {
                "clock": "verified_clip_order_and_sync_offset; camera timestamps ignored",
                "phases": sorted(ELIGIBLE_PHASES),
                "minimum_clip_duration_s": MINIMUM_CLIP_DURATION_S,
                "source_crop": "none (full-width source frame)",
            },
            "calibration_sha256": sha256_file(output),
            "frames": provenance_rows,
        },
    )
    return calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=Path("work/log-audit/world10_clips.csv"))
    parser.add_argument(
        "--telemetry", type=Path, default=Path("work/log-audit/world10_operational.csv")
    )
    parser.add_argument("--sync", type=Path, default=Path("work/log-audit/world10_sync.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../CanyonBench-data/calibration/flight-quality.json"),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("../CanyonBench-data/calibration/flight-quality-provenance.json"),
    )
    args = parser.parse_args()
    clips = select_clips(args.clips, args.telemetry, args.sync)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    calibration = calibrate(clips, args.output, args.provenance)
    print(f"Wrote {calibration.frame_count} full-frame measurements to {args.output}")


if __name__ == "__main__":
    main()
