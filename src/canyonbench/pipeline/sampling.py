"""Distance-aware and perceptual sampling for correlated trajectory frames."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from canyonbench.exceptions import DataValidationError

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = np.radians([a[0], a[1], b[0], b[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(value)))


def perceptual_hash(path: str | Path) -> str:
    with Image.open(path) as image:
        return str(imagehash.phash(image.convert("RGB")))


def hash_distance(left: str, right: str) -> int:
    return imagehash.hex_to_hash(left) - imagehash.hex_to_hash(right)


def add_perceptual_hashes(
    frame: pd.DataFrame, image_root: str | Path | None = None
) -> pd.DataFrame:
    output = frame.copy()
    root = Path(image_root) if image_root is not None else None

    def resolve(row: pd.Series) -> Path:
        if "image_path" in row and pd.notna(row["image_path"]):
            return Path(str(row["image_path"]))
        if root is None:
            raise DataValidationError(
                "image_root or image_path is required to compute perceptual hashes"
            )
        return root / str(row["image"])

    output["phash"] = [perceptual_hash(resolve(row)) for _, row in output.iterrows()]
    return output


def sample_frames(
    frame: pd.DataFrame,
    *,
    distance_m: float = 500,
    phash_distance: int = 8,
    min_interval_s: int = 60,
    max_interval_s: int | None = None,
) -> pd.DataFrame:
    """Keep non-adjacent frames when movement or perceptual change is sufficient.

    The minimum interval is applied first, which prevents brief camera motion from
    admitting dense bursts. ``max_interval_s`` can force periodic coverage during
    a genuinely static stretch but is disabled by default.
    """

    required = {"elapsed_s", "lat", "lon", "phash"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"Sampling input is missing columns: {missing}")
    if not 30 <= min_interval_s <= 120:
        raise DataValidationError("min_interval_s must be between 30 and 120 seconds")
    if distance_m <= 0 or phash_distance < 0:
        raise DataValidationError("distance and perceptual thresholds must be non-negative")

    ordered = frame.sort_values("elapsed_s").reset_index(drop=True)
    keep_indices: list[int] = []
    last: pd.Series | None = None
    reasons: dict[int, str] = {}
    for index, (_, current) in enumerate(ordered.iterrows()):
        if last is None:
            keep_indices.append(index)
            reasons[index] = "first"
            last = current
            continue
        elapsed = int(current["elapsed_s"]) - int(last["elapsed_s"])
        if elapsed < min_interval_s:
            continue
        moved = haversine_m(
            (float(last["lat"]), float(last["lon"])),
            (float(current["lat"]), float(current["lon"])),
        )
        changed = hash_distance(str(last["phash"]), str(current["phash"]))
        forced = max_interval_s is not None and elapsed >= max_interval_s
        if moved >= distance_m or changed > phash_distance or forced:
            keep_indices.append(index)
            reason_parts = []
            if moved >= distance_m:
                reason_parts.append("distance")
            if changed > phash_distance:
                reason_parts.append("perceptual")
            if forced:
                reason_parts.append("max_interval")
            reasons[index] = "+".join(reason_parts)
            last = current
    sampled = ordered.loc[keep_indices].copy()
    sampled["sample_reason"] = [reasons[index] for index in keep_indices]
    return sampled.reset_index(drop=True)


def assign_segments(
    frame: pd.DataFrame,
    *,
    max_gap_s: int = 600,
    max_jump_m: float = 10_000,
    max_duration_s: int | None = 600,
) -> pd.DataFrame:
    required = {"elapsed_s", "lat", "lon", "phase"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"Segment input is missing columns: {missing}")
    ordered = frame.sort_values("elapsed_s").reset_index(drop=True).copy()
    ids: list[str] = []
    segment_number = 0
    previous: pd.Series | None = None
    segment_start_s: int | None = None
    for _, current in ordered.iterrows():
        current_elapsed_s = int(current["elapsed_s"])
        if segment_start_s is None:
            segment_start_s = current_elapsed_s
        if previous is not None:
            gap = current_elapsed_s - int(previous["elapsed_s"])
            jump = haversine_m(
                (float(previous["lat"]), float(previous["lon"])),
                (float(current["lat"]), float(current["lon"])),
            )
            duration_boundary = (
                max_duration_s is not None and current_elapsed_s - segment_start_s >= max_duration_s
            )
            if (
                gap > max_gap_s
                or jump > max_jump_m
                or current["phase"] != previous["phase"]
                or duration_boundary
            ):
                segment_number += 1
                segment_start_s = current_elapsed_s
        ids.append(f"seg_{segment_number:04d}")
        previous = current
    ordered["segment_id"] = ids
    return ordered


def _spatial_block(lat: float, lon: float, block_size_m: float) -> str:
    lat_size = block_size_m / 111_320.0
    lon_size = block_size_m / (111_320.0 * max(math.cos(math.radians(lat)), 0.1))
    return f"{math.floor(lat / lat_size)}:{math.floor(lon / lon_size)}"


def assign_geographic_splits(
    frame: pd.DataFrame,
    *,
    block_size_m: float = 5_000,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 2026,
) -> pd.DataFrame:
    """Assign whole spatial blocks to splits, preventing adjacent-frame leakage."""

    if train_fraction + validation_fraction >= 1:
        raise DataValidationError(
            "train + validation fractions must leave a non-empty test fraction"
        )
    output = frame.copy()
    output["spatial_block"] = [
        _spatial_block(float(lat), float(lon), block_size_m)
        for lat, lon in zip(output["lat"], output["lon"], strict=True)
    ]

    def split(block: str) -> str:
        digest = hashlib.sha256(f"{seed}:{block}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        if value < train_fraction:
            return "train"
        if value < train_fraction + validation_fraction:
            return "validation"
        return "test"

    output["split"] = output["spatial_block"].map(split)
    if "segment_id" in output:
        # Geographic blocks own their deterministic split. Refine a trajectory
        # segment when it crosses a split boundary instead of overriding the
        # block split, which could collapse a long flight into a single subset.
        original_segments = output["segment_id"].astype(str)
        boundaries = original_segments.ne(original_segments.shift()) | output["split"].ne(
            output["split"].shift()
        )
        output["segment_id"] = [
            f"seg_{index:04d}" for index in boundaries.cumsum().sub(1).astype(int)
        ]
    return output
