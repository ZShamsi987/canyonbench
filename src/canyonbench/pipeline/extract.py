"""Build and optionally execute deterministic ffmpeg extraction commands."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

from canyonbench.exceptions import DataValidationError, ExternalToolError
from canyonbench.io import read_json, sha256_file, write_json
from canyonbench.pipeline.cloud_cache import evict_cloud_files
from canyonbench.pipeline.sync import SyncAnchor, clip_flight_start


def extraction_command(
    clip_row: pd.Series,
    anchor: SyncAnchor,
    output_dir: str | Path,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    start = clip_flight_start(clip_row, anchor)
    # output timestamps are shifted to the flight clock before integer-second selection.
    filter_graph = f"setpts=PTS+{start:.6f}/TB,fps=1:round=near,crop=trunc(iw*2/3/2)*2:ih:iw/3:0"
    pattern = Path(output_dir) / f"clip_{int(clip_row['clip_index']):04d}_%010d.jpg"
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(clip_row["path"]),
        "-vf",
        filter_graph,
        "-q:v",
        "2",
        "-vsync",
        "0",
        str(pattern),
    ]


def extract_clips(
    clips: pd.DataFrame,
    anchor: SyncAnchor,
    output_dir: str | Path,
    *,
    execute: bool,
    resume: bool = False,
    evict_source_cache: bool = False,
    eviction_batch_size: int = 10,
    checksum_manifest: str | Path | None = None,
    ffmpeg: str = "ffmpeg",
) -> list[list[str]]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if eviction_batch_size < 1:
        raise DataValidationError("eviction_batch_size must be positive")
    commands = [extraction_command(row, anchor, destination, ffmpeg) for _, row in clips.iterrows()]
    if not execute:
        return commands
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise ExternalToolError("ffmpeg is required for frame extraction")
    pending_evictions: list[Path] = []
    checksum_records: list[dict[str, str]] = []
    try:
        for (_, row), command in zip(clips.iterrows(), commands, strict=True):
            clip_index = int(row["clip_index"])
            marker = destination / f".clip_{clip_index:04d}.complete.json"
            frame_glob = f"clip_{clip_index:04d}_*.jpg"
            record: object = None
            valid_complete_marker = False
            if resume and marker.is_file():
                try:
                    record = read_json(marker)
                except DataValidationError:
                    record = None
                actual_count = len(list(destination.glob(frame_glob)))
                valid_complete_marker = (
                    isinstance(record, dict)
                    and record.get("clip") == str(row["clip"])
                    and record.get("frame_count") == actual_count
                    and actual_count > 0
                )
                checksum_present = (
                    isinstance(record, dict)
                    and isinstance(record.get("source_sha256"), str)
                    and len(record["source_sha256"]) == 64
                )
                if valid_complete_marker and (checksum_manifest is None or checksum_present):
                    if checksum_present:
                        assert isinstance(record, dict)
                        checksum_records.append(
                            {
                                "clip": str(row["clip"]),
                                "sha256": str(record["source_sha256"]),
                            }
                        )
                    continue
            if not valid_complete_marker:
                command[0] = executable
                command.insert(1, "-y")
                try:
                    subprocess.run(command, check=True, timeout=900)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    raise ExternalToolError(f"ffmpeg failed for {command[-1]}: {exc}") from exc
                frame_count = len(list(destination.glob(frame_glob)))
                if frame_count == 0:
                    raise ExternalToolError(f"ffmpeg produced no frames for {row['clip']}")
            else:
                assert isinstance(record, dict)
                frame_count = int(record["frame_count"])
            source_sha256 = (
                sha256_file(str(row["path"])) if checksum_manifest is not None else None
            )
            marker_record: dict[str, object] = {
                "clip": str(row["clip"]),
                "clip_index": clip_index,
                "duration_s": float(row["duration_s"]),
                "frame_count": frame_count,
            }
            if source_sha256 is not None:
                marker_record["source_sha256"] = source_sha256
                checksum_records.append(
                    {"clip": str(row["clip"]), "sha256": source_sha256}
                )
            write_json(
                marker,
                marker_record,
            )
            if evict_source_cache:
                pending_evictions.append(Path(str(row["path"])))
                if len(pending_evictions) >= eviction_batch_size:
                    evict_cloud_files(pending_evictions)
                    pending_evictions.clear()
    finally:
        if pending_evictions:
            evict_cloud_files(pending_evictions)
    if checksum_manifest is not None:
        write_json(
            checksum_manifest,
            {
                "algorithm": "sha256",
                "clips": checksum_records,
            },
        )
    return commands
