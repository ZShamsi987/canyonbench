"""Cross-file release validation with fail-closed benchmark invariants."""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from canyonbench.constants import FEATURES, SCORED_PHASES
from canyonbench.groundtruth.masks import load_binary_mask
from canyonbench.io import read_json

IMAGE_PATTERN = re.compile(r"^img_(\d{6,})\.jpg$")


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    image: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cells(value: Any) -> list[str] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None
    return parsed if isinstance(parsed, list) else None


def validate_release(directory: str | Path, *, require_files: bool = True) -> list[ValidationIssue]:
    root = Path(directory)
    issues: list[ValidationIssue] = []
    frames_path = root / "frames.csv"
    if not frames_path.exists():
        return [ValidationIssue("error", "missing_frames", f"Missing {frames_path}")]
    try:
        frames = pd.read_csv(frames_path)
    except (OSError, pd.errors.ParserError) as exc:
        return [ValidationIssue("error", "invalid_frames_csv", str(exc))]
    required = {
        "image",
        "elapsed_s",
        "phase",
        "lat",
        "lon",
        "alt_m",
        "segment_id",
        "split",
        "vegetation_fraction",
        "registration_reliable",
        "vegetation_cells",
        *FEATURES,
    }
    missing = sorted(required - set(frames.columns))
    if missing:
        issues.append(ValidationIssue("error", "missing_columns", f"Missing columns: {missing}"))
        return issues
    if frames["image"].duplicated().any():
        issues.append(ValidationIssue("error", "duplicate_image", "Image keys are not unique"))
    if frames["elapsed_s"].duplicated().any():
        issues.append(
            ValidationIssue("error", "duplicate_second", "Elapsed seconds are not unique")
        )
    for row in frames.to_dict(orient="records"):
        image = str(row["image"])
        match = IMAGE_PATTERN.match(image)
        if match is None or int(match.group(1)) != int(row["elapsed_s"]):
            issues.append(
                ValidationIssue(
                    "error", "image_second_mismatch", "Filename and elapsed_s differ", image
                )
            )
        if row["phase"] not in SCORED_PHASES:
            issues.append(ValidationIssue("error", "unscored_phase", str(row["phase"]), image))
        if not (-90 <= float(row["lat"]) <= 90 and -180 <= float(row["lon"]) <= 180):
            issues.append(ValidationIssue("error", "invalid_coordinate", "Invalid lat/lon", image))
        if float(row["lat"]) == 0 or float(row["lon"]) == 0:
            issues.append(
                ValidationIssue("error", "zero_gps", "GPS dropout was not filtered", image)
            )
        if not 0 <= float(row["vegetation_fraction"]) <= 1:
            issues.append(
                ValidationIssue("error", "invalid_cover", "Cover is outside [0,1]", image)
            )
        for feature in FEATURES:
            if row[feature] not in {"yes", "no", "uncertain"}:
                issues.append(
                    ValidationIssue("error", "invalid_presence", f"{feature}={row[feature]}", image)
                )
        reliable = str(row["registration_reliable"]).lower() in {"true", "1"}
        cells = _cells(row["vegetation_cells"])
        if reliable and cells is None:
            issues.append(
                ValidationIssue("error", "missing_grid", "Reliable registration has no grid", image)
            )
        if not reliable and cells is not None:
            issues.append(
                ValidationIssue("error", "unreliable_grid", "Unreliable frame has a grid", image)
            )
        if cells is not None:
            valid_cells = {f"{r},{c}" for r in range(4) for c in range(4)}
            if not set(cells) <= valid_cells or len(cells) != len(set(cells)):
                issues.append(ValidationIssue("error", "invalid_grid", str(cells), image))
        if require_files:
            frame_path = root / "frames" / image
            mask_path = root / "masks" / f"{Path(image).stem}.png"
            if not frame_path.exists():
                issues.append(
                    ValidationIssue("error", "missing_image_file", str(frame_path), image)
                )
            if not mask_path.exists():
                issues.append(ValidationIssue("error", "missing_mask_file", str(mask_path), image))
            elif frame_path.exists():
                try:
                    with Image.open(frame_path) as frame_image:
                        load_binary_mask(mask_path, frame_image.size)
                except Exception as exc:
                    issues.append(ValidationIssue("error", "invalid_mask", str(exc), image))
    if "spatial_block" in frames:
        leakage = frames.groupby("spatial_block")["split"].nunique()
        if (leakage > 1).any():
            issues.append(
                ValidationIssue(
                    "error",
                    "spatial_leakage",
                    "At least one spatial block occurs in multiple splits",
                )
            )
    segment_leakage = frames.groupby("segment_id")["split"].nunique()
    if (segment_leakage > 1).any():
        issues.append(
            ValidationIssue(
                "error",
                "segment_leakage",
                "At least one trajectory segment occurs in multiple splits",
            )
        )
    manifest_path = root / "release_manifest.json"
    if not manifest_path.exists():
        issues.append(ValidationIssue("error", "missing_manifest", str(manifest_path)))
    else:
        try:
            manifest = read_json(manifest_path)
            if int(manifest["n_frames"]) != len(frames):
                issues.append(
                    ValidationIssue(
                        "error", "manifest_count", "Manifest frame count does not match"
                    )
                )
            if int(manifest["effective_segment_count"]) != frames["segment_id"].nunique():
                issues.append(
                    ValidationIssue("error", "manifest_segments", "Segment count does not match")
                )
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(ValidationIssue("error", "invalid_manifest", str(exc)))
    return issues
