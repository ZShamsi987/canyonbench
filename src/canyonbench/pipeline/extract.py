"""Build and optionally execute deterministic ffmpeg extraction commands."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

from canyonbench.exceptions import ExternalToolError
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
    ffmpeg: str = "ffmpeg",
) -> list[list[str]]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    commands = [extraction_command(row, anchor, destination, ffmpeg) for _, row in clips.iterrows()]
    if not execute:
        return commands
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise ExternalToolError("ffmpeg is required for frame extraction")
    for command in commands:
        command[0] = executable
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise ExternalToolError(f"ffmpeg failed for {command[-1]}: {exc}") from exc
    return commands
