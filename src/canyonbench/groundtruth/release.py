"""Merge adjudicated labels and registration into the master release table."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, cast

import pandas as pd
from PIL import Image

from canyonbench.constants import FEATURES
from canyonbench.exceptions import DataValidationError
from canyonbench.groundtruth.labels import load_grid, load_presence, load_quality
from canyonbench.groundtruth.masks import grid_labels, load_binary_mask, vegetation_fraction
from canyonbench.io import write_json, write_jsonl
from canyonbench.pipeline.join import candidate_exclusion_reason


def _one_record_per_image(records: list[Any], label: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for record in records:
        if record.image in output:
            raise DataValidationError(f"Duplicate adjudicated {label} label: {record.image}")
        output[record.image] = record
    return output


def _as_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def build_release(
    sampled_frames: pd.DataFrame,
    data_repository: str | Path,
    output_dir: str | Path,
) -> pd.DataFrame:
    data_root = Path(data_repository)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    presence = _one_record_per_image(
        load_presence(data_root / "labels/adjudicated/presence.jsonl"), "presence"
    )
    quality = _one_record_per_image(
        load_quality(data_root / "labels/adjudicated/quality.jsonl"), "quality"
    )
    grid_path = data_root / "labels/adjudicated/grid.jsonl"
    grids = _one_record_per_image(load_grid(grid_path), "grid") if grid_path.exists() else {}
    residual_path = data_root / "registration/residuals.csv"
    if not residual_path.exists():
        raise DataValidationError(f"Missing registration residuals: {residual_path}")
    residuals = pd.read_csv(residual_path)
    residual_by_image = residuals.set_index("image").to_dict(orient="index")

    rows: list[dict[str, Any]] = []
    release_masks = output / "masks"
    release_frames = output / "frames"
    release_masks.mkdir(exist_ok=True)
    release_frames.mkdir(exist_ok=True)
    for raw_row in sampled_frames.to_dict(orient="records"):
        row = cast(dict[str, Any], raw_row)
        image = str(row["image"])
        if image not in presence or image not in quality:
            raise DataValidationError(f"Missing adjudicated presence or quality labels for {image}")
        mask_path = data_root / "masks/adjudicated" / f"{Path(image).stem}.png"
        frame_path = Path(str(row["image_path"]))
        with Image.open(frame_path) as frame_image:
            mask = load_binary_mask(mask_path, frame_image.size)
        shutil.copy2(mask_path, release_masks / mask_path.name)
        shutil.copy2(frame_path, release_frames / image)
        record = dict(row)
        record["image_path"] = str(Path("frames") / image)
        record.update({feature: getattr(presence[image], feature) for feature in FEATURES})
        record.update(quality[image].model_dump(exclude={"image", "annotator"}))
        record["vegetation_fraction"] = vegetation_fraction(mask)
        registration = residual_by_image.get(image)
        record["registration_reliable"] = bool(
            registration and _as_boolean(registration.get("reliable"))
        )
        record["holdout_rmse_m"] = registration.get("holdout_rmse_m") if registration else None
        expected_grid = grid_labels(mask)
        if record["registration_reliable"]:
            if image not in grids:
                raise DataValidationError(f"Reliable frame is missing a grounding grid: {image}")
            if grids[image].cells != expected_grid:
                record["grid_override"] = True
            record["vegetation_cells"] = sorted(
                cell for cell, present in grids[image].cells.items() if present
            )
        else:
            if image in grids:
                raise DataValidationError(
                    f"Unreliable frame must not have a grounding grid: {image}"
                )
            record["vegetation_cells"] = None
        reason = candidate_exclusion_reason(pd.Series(record))
        record["exclusion_candidate"] = reason is not None
        record["exclusion_reason"] = reason
        rows.append(record)

    master = pd.DataFrame(rows)
    master.to_csv(output / "frames.csv", index=False)
    write_jsonl(output / "labels.jsonl", rows)
    write_json(
        output / "release_manifest.json",
        {
            "schema_version": "1.0.0",
            "n_frames": len(master),
            "n_registered": int(master["registration_reliable"].sum()),
            "raw_frame_count": int(
                sampled_frames["sampling_raw_count"].iloc[0]
                if "sampling_raw_count" in sampled_frames and len(sampled_frames)
                else sampled_frames.attrs.get("raw_frame_count", len(sampled_frames))
            ),
            "effective_segment_count": int(master["segment_id"].nunique()),
        },
    )
    return master
