"""Video inventory via ffprobe with deterministic fallback ordering."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from canyonbench.exceptions import DataValidationError, ExternalToolError


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)
    )


def _probe(path: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:format_tags=creation_time",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        value = json.loads(completed.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ExternalToolError(f"ffprobe failed for {path}: {exc}") from exc
    format_info = value.get("format", {})
    return {
        "duration_s": float(format_info.get("duration", 0)),
        "creation_time": format_info.get("tags", {}).get("creation_time"),
    }


def inventory_clips(directory: str | Path, ffprobe: str = "ffprobe") -> pd.DataFrame:
    root = Path(directory)
    if not root.is_dir():
        raise DataValidationError(f"Video directory does not exist: {root}")
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
    for path in paths:
        metadata = _probe(path, executable)
        rows.append(
            {
                "clip": path.name,
                "path": str(path.resolve()),
                "duration_s": metadata["duration_s"],
                "creation_time": metadata["creation_time"],
                "mtime": path.stat().st_mtime,
                "natural_order": _natural_key(path),
            }
        )
    if all(row["creation_time"] for row in rows):
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
