"""Lossless frame-to-telemetry join and release-table assembly."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from canyonbench.constants import SCORED_PHASES
from canyonbench.exceptions import DataValidationError
from canyonbench.pipeline.quality import image_quality_controls

IMAGE_PATTERN = re.compile(r"^img_(\d{6,})\.jpg$")


def discover_frames(directory: str | Path) -> pd.DataFrame:
    root = Path(directory)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("img_*.jpg")):
        match = IMAGE_PATTERN.match(path.name)
        if match is None:
            continue
        rows.append(
            {
                "image": path.name,
                "image_path": str(path.resolve()),
                "elapsed_s": int(match.group(1)),
            }
        )
    if not rows:
        raise DataValidationError(f"No elapsed-second frames found in {root}")
    return pd.DataFrame(rows)


def build_frames_table(
    images: pd.DataFrame,
    flight: pd.DataFrame,
    *,
    phases: tuple[str, ...] = SCORED_PHASES,
    add_quality_controls: bool = True,
) -> pd.DataFrame:
    if images["elapsed_s"].duplicated().any():
        raise DataValidationError("Image table contains duplicate elapsed seconds")
    if flight["elapsed_s"].duplicated().any():
        raise DataValidationError("Flight table contains duplicate elapsed seconds")
    joined = images.merge(flight, on="elapsed_s", how="left", validate="one_to_one", indicator=True)
    unmatched = joined.loc[joined["_merge"] != "both", "image"].tolist()
    if unmatched:
        preview = unmatched[:5]
        raise DataValidationError(f"Frames have no matching valid flight row: {preview}")
    joined = joined.drop(columns="_merge")
    joined = joined.loc[joined["phase"].isin(phases)].copy()
    if add_quality_controls:
        controls = pd.DataFrame(
            [image_quality_controls(path) for path in joined["image_path"]],
            index=joined.index,
        )
        joined = pd.concat([joined, controls], axis=1)
    return joined.sort_values("elapsed_s").reset_index(drop=True)


def candidate_exclusion_reason(row: pd.Series) -> str | None:
    reasons: list[str] = []
    if row.get("cloud") == "heavy":
        reasons.append("cloud_heavy")
    if row.get("clarity") == "heavy":
        reasons.append("clarity_heavy")
    if row.get("balloon") == "partial":
        reasons.append("balloon_partial")
    return "+".join(reasons) or None
