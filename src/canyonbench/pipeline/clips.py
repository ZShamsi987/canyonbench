"""Video inventory via ffprobe with deterministic fallback ordering."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from canyonbench.exceptions import DataValidationError, ExternalToolError
from canyonbench.pipeline.cloud_cache import evict_cloud_files


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)
    )


def _probe(path: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-read_intervals",
        "59%",
        "-show_entries",
        (
            "format=duration:format_tags=creation_time:"
            "stream=r_frame_rate,width,height:"
            "frame=best_effort_timestamp_time"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        value = json.loads(completed.stdout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise ExternalToolError(f"ffprobe failed for {path}: {exc}") from exc
    format_info = value.get("format", {})
    streams = value.get("streams", [])
    frames = value.get("frames", [])
    if not streams or not frames:
        raise ExternalToolError(f"ffprobe found no decodable video frames in {path}")
    try:
        frame_rate = Fraction(str(streams[0]["r_frame_rate"]))
        last_frame_pts_s = max(
            float(frame["best_effort_timestamp_time"])
            for frame in frames
            if "best_effort_timestamp_time" in frame
        )
        duration_s = last_frame_pts_s + float(1 / frame_rate)
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise ExternalToolError(f"ffprobe returned incomplete video timing for {path}") from exc
    return {
        "duration_s": duration_s,
        "declared_duration_s": float(format_info.get("duration", 0)),
        "last_frame_pts_s": last_frame_pts_s,
        "frame_rate": str(streams[0]["r_frame_rate"]),
        "width": int(streams[0]["width"]),
        "height": int(streams[0]["height"]),
        "creation_time": format_info.get("tags", {}).get("creation_time"),
    }


def inventory_clips(
    directory: str | Path,
    ffprobe: str = "ffprobe",
    *,
    order_by: Literal["auto", "filename", "relative_time"] = "auto",
    evict_source_cache: bool = False,
    eviction_batch_size: int = 10,
    workers: int = 1,
) -> pd.DataFrame:
    """Probe clips and place them on one continuous relative video clock.

    ``filename`` is the safest explicit choice for cameras that write a
    gap-free numeric sequence while carrying an incorrect wall clock.
    ``relative_time`` uses filesystem modification times only as relative
    evidence; it does not assert that their dates are accurate.
    """

    root = Path(directory)
    if not root.is_dir():
        raise DataValidationError(f"Video directory does not exist: {root}")
    if order_by not in {"auto", "filename", "relative_time"}:
        raise DataValidationError(f"Unsupported clip ordering policy: {order_by}")
    if eviction_batch_size < 1:
        raise DataValidationError("eviction_batch_size must be positive")
    if workers < 1:
        raise DataValidationError("workers must be positive")
    executable = shutil.which(ffprobe)
    if executable is None:
        raise ExternalToolError("ffprobe is required to inventory clips")
    paths = sorted(
        (path for path in root.iterdir() if path.suffix.lower() in {".avi", ".mp4", ".mov"}),
        key=_natural_key,
    )
    if not paths:
        raise DataValidationError(f"No supported video clips found in {root}")
    rows: list[dict[str, Any]] = []
    pending_evictions: list[Path] = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            metadata_values = executor.map(partial(_probe, ffprobe=executable), paths)
            for path, metadata in zip(paths, metadata_values, strict=True):
                rows.append(
                    {
                        "clip": path.name,
                        "path": str(path.resolve()),
                        "duration_s": metadata["duration_s"],
                        "declared_duration_s": metadata.get("declared_duration_s"),
                        "last_frame_pts_s": metadata.get("last_frame_pts_s"),
                        "frame_rate": metadata.get("frame_rate"),
                        "width": metadata.get("width"),
                        "height": metadata.get("height"),
                        "creation_time": metadata["creation_time"],
                        "mtime": path.stat().st_mtime,
                        "natural_order": _natural_key(path),
                    }
                )
                if evict_source_cache:
                    pending_evictions.append(path)
                    if len(pending_evictions) >= eviction_batch_size:
                        evict_cloud_files(pending_evictions)
                        pending_evictions.clear()
    finally:
        if pending_evictions:
            evict_cloud_files(pending_evictions)
    if order_by == "filename":
        rows.sort(key=lambda row: row["natural_order"])
        order_source = "filename_relative_sequence"
    elif order_by == "relative_time":
        rows.sort(key=lambda row: (row["mtime"], row["natural_order"]))
        order_source = "mtime_relative_only_then_filename"
    elif all(row["creation_time"] for row in rows):
        rows.sort(key=lambda row: (row["creation_time"], row["natural_order"]))
        order_source = "creation_time"
    else:
        rows.sort(key=lambda row: (row["mtime"], row["natural_order"]))
        order_source = "mtime_then_filename"
    elapsed = 0.0
    for index, row in enumerate(rows):
        row["clip_index"] = index
        row["video_start_s"] = elapsed
        row["video_end_s"] = elapsed + row["duration_s"]
        row["order_source"] = order_source
        row.pop("natural_order")
        elapsed = row["video_end_s"]
    return pd.DataFrame(rows)
