"""Batch registration with auditable residual and matrix outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from canyonbench.exceptions import DataValidationError
from canyonbench.io import write_json
from canyonbench.registration.homography import fit_homography, read_control_points


def register_manifest(
    manifest: pd.DataFrame,
    *,
    output_dir: str | Path,
    default_threshold_m: float,
) -> pd.DataFrame:
    required = {"image", "points_path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise DataValidationError(f"Registration manifest is missing: {missing}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in manifest.to_dict(orient="records"):
        points = read_control_points(record["points_path"])
        record_threshold = record.get("threshold_m")
        threshold = float(record_threshold) if pd.notna(record_threshold) else default_threshold_m
        result = fit_homography(
            record["image"],
            points,
            threshold_m=threshold,
        )
        row = result.to_dict()
        homography = row.pop("homography")
        rows.append(row)
        if homography is not None:
            stem = Path(record["image"]).stem
            write_json(
                output / f"{stem}.homography.json",
                {"image": record["image"], "H": homography},
            )
    residuals = pd.DataFrame(rows)
    residuals.to_csv(output / "residuals.csv", index=False)
    return residuals
