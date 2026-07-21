"""Single-anchor video-to-flight synchronization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from canyonbench.exceptions import DataValidationError
from canyonbench.io import read_json, write_json


@dataclass(frozen=True)
class SyncAnchor:
    clip: str
    anchor_offset_s: float
    flight_elapsed_s: int
    event: str
    video_elapsed_s: float
    flight_offset_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_anchor(
    clips: pd.DataFrame,
    clip: str,
    anchor_offset_s: float,
    flight_elapsed_s: int,
    event: str,
) -> SyncAnchor:
    matches = clips.loc[clips["clip"] == clip]
    if len(matches) != 1:
        raise DataValidationError(f"Anchor clip must match exactly once: {clip}")
    row = matches.iloc[0]
    if not 0 <= anchor_offset_s <= float(row["duration_s"]):
        raise DataValidationError("Anchor offset lies outside the selected clip")
    video_elapsed_s = float(row["video_start_s"]) + anchor_offset_s
    return SyncAnchor(
        clip=clip,
        anchor_offset_s=anchor_offset_s,
        flight_elapsed_s=flight_elapsed_s,
        event=event,
        video_elapsed_s=video_elapsed_s,
        flight_offset_s=flight_elapsed_s - video_elapsed_s,
    )


def save_anchor(path: str | Path, anchor: SyncAnchor) -> None:
    write_json(path, anchor.to_dict())


def load_anchor(path: str | Path) -> SyncAnchor:
    value = read_json(path)
    try:
        return SyncAnchor(**value)
    except TypeError as exc:
        raise DataValidationError(f"Invalid synchronization file {path}: {exc}") from exc


def clip_flight_start(clip_row: pd.Series, anchor: SyncAnchor) -> float:
    return float(clip_row["video_start_s"]) + anchor.flight_offset_s
