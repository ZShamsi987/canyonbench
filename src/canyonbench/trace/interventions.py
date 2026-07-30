"""Target, distractor, random, and texture-matched causal image interventions."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
from PIL import Image

from canyonbench.exceptions import DataValidationError
from canyonbench.io import sha256_file, write_json
from canyonbench.trace.schemas import (
    FeatureClass,
    InterventionConfig,
    InterventionRecord,
    RegionCovariates,
    TraceSequence,
)


def _binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def region_covariates(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    depth: np.ndarray | None = None,
) -> RegionCovariates:
    binary = _binary(mask)
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(float) / 255
    pixels = binary > 0
    if not pixels.any():
        return RegionCovariates(
            area_px=0,
            texture_energy=0,
            edge_density=0,
            mean_brightness=0,
            centre_distance=0,
            boundary_complexity=0,
            mean_depth=None,
        )
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    edges = cv2.Canny((gray * 255).astype(np.uint8), 80, 160) > 0
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    rows, columns = np.where(pixels)
    centre_y, centre_x = (binary.shape[0] - 1) / 2, (binary.shape[1] - 1) / 2
    distance = np.hypot(np.mean(rows) - centre_y, np.mean(columns) - centre_x)
    diagonal = np.hypot(binary.shape[0], binary.shape[1])
    depth_values = depth[pixels] if depth is not None else np.array([])
    return RegionCovariates(
        area_px=int(pixels.sum()),
        texture_energy=float(np.mean(laplacian[pixels])),
        edge_density=float(np.mean(edges[pixels])),
        mean_brightness=float(np.mean(gray[pixels])),
        centre_distance=float(distance / diagonal),
        boundary_complexity=float(perimeter / max(np.sqrt(pixels.sum()), 1)),
        mean_depth=(
            float(np.nanmean(depth_values))
            if depth_values.size and np.isfinite(depth_values).any()
            else None
        ),
    )


def standardized_differences(
    target: RegionCovariates, control: RegionCovariates
) -> dict[str, float]:
    """Per-image scale-normalized balance diagnostic used for regeneration."""

    differences: dict[str, float] = {}
    for field in (
        "area_px",
        "texture_energy",
        "edge_density",
        "mean_brightness",
        "centre_distance",
        "boundary_complexity",
        "mean_depth",
    ):
        first, second = getattr(target, field), getattr(control, field)
        if first is None or second is None:
            continue
        denominator = max((abs(float(first)) + abs(float(second))) / 2, 1e-6)
        differences[field] = abs(float(first) - float(second)) / denominator
    return differences


def _candidate_patch(
    shape: tuple[int, int], target: np.ndarray, generator: np.random.Generator
) -> np.ndarray:
    """Translate the target shape so area and boundary complexity match exactly."""

    area = int(np.sum(target > 0))
    height, width = shape
    rows, columns = np.where(target > 0)
    if len(rows):
        target_height = int(rows.max() - rows.min() + 1)
        target_width = int(columns.max() - columns.min() + 1)
        if target_height <= height and target_width <= width:
            top = int(generator.integers(0, max(1, height - target_height + 1)))
            left = int(generator.integers(0, max(1, width - target_width + 1)))
            translated = np.zeros(shape, np.uint8)
            translated[
                rows - rows.min() + top,
                columns - columns.min() + left,
            ] = 1
            if not np.logical_and(translated > 0, target > 0).any():
                return translated
    # Fallback for a target whose bounding box covers almost the full image.
    radius = max(1, int(np.sqrt(area / np.pi)))
    radius = min(radius, max(1, min(height, width) // 3))
    centre_x = int(generator.integers(radius, max(radius + 1, width - radius)))
    centre_y = int(generator.integers(radius, max(radius + 1, height - radius)))
    patch = np.zeros(shape, np.uint8)
    cv2.circle(patch, (centre_x, centre_y), radius, 1, -1)
    overlap = np.logical_and(patch > 0, target > 0).sum()
    if overlap:
        patch[target > 0] = 0
    if patch.sum() > area:
        rows, columns = np.where(patch > 0)
        order = generator.permutation(len(rows))
        patch[:] = 0
        keep = order[:area]
        patch[rows[keep], columns[keep]] = 1
    return patch


def texture_replacement(
    image: np.ndarray, region: np.ndarray, donor_region: np.ndarray
) -> np.ndarray:
    """Copy statistically matched donor pixels into a target-shaped region."""

    output = image.copy()
    target_rows, target_columns = np.where(region > 0)
    donor_rows, donor_columns = np.where(donor_region > 0)
    count = min(len(target_rows), len(donor_rows))
    if count:
        output[target_rows[:count], target_columns[:count]] = image[
            donor_rows[:count], donor_columns[:count]
        ]
    return output


def match_distractor(
    image: np.ndarray,
    target_mask: np.ndarray,
    *,
    depth: np.ndarray | None,
    candidates: int,
    seed: int,
) -> tuple[np.ndarray, RegionCovariates, dict[str, float]]:
    """Find the non-overlapping patch minimizing registered covariate imbalance."""

    target = _binary(target_mask)
    target_values = region_covariates(image, target, depth=depth)
    generator = np.random.default_rng(seed)
    best: tuple[float, np.ndarray, RegionCovariates, dict[str, float]] | None = None
    for _ in range(candidates):
        candidate = _candidate_patch(target.shape, target, generator)
        if not candidate.any():
            continue
        values = region_covariates(image, candidate, depth=depth)
        differences = standardized_differences(target_values, values)
        score = max(differences.values(), default=float("inf"))
        if best is None or score < best[0]:
            best = (score, candidate, values, differences)
    if best is None:
        raise ValueError("could not construct a distractor region")
    return best[1], best[2], best[3]


def rank_target_pixels(image: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    """Rank target pixels from most locally visible to least locally visible."""

    target = _binary(target_mask)
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(float)
    local_mean = cv2.GaussianBlur(gray, (0, 0), 5)
    visibility = np.abs(gray - local_mean)
    rows, columns = np.where(target > 0)
    order = np.argsort(-visibility[rows, columns], kind="stable")
    ranking = np.column_stack([rows[order], columns[order]])
    return ranking


def fractional_region(mask: np.ndarray, ranking: np.ndarray, fraction: float) -> np.ndarray:
    binary = _binary(mask)
    if fraction <= 0:
        return np.zeros_like(binary)
    count = int(np.ceil(len(ranking) * fraction))
    selected = np.zeros_like(binary)
    chosen = ranking[:count]
    if len(chosen):
        selected[chosen[:, 0], chosen[:, 1]] = 1
    return selected


def _feather(mask: np.ndarray, feather_px: int) -> np.ndarray:
    if feather_px <= 0:
        return _binary(mask).astype(np.float32)
    return cv2.GaussianBlur(_binary(mask).astype(np.float32), (0, 0), feather_px / 2).clip(0, 1)


def apply_operator(
    image: np.ndarray,
    region: np.ndarray,
    operator: str,
    *,
    feather_px: int,
    texture_source: np.ndarray | None = None,
) -> np.ndarray:
    """Apply O1-O4 within a feathered region."""

    rgb = image.astype(np.uint8)
    alpha = _feather(region, feather_px)[..., None]
    if operator == "blur":
        replacement = cv2.GaussianBlur(rgb, (0, 0), 8)
    elif operator == "texture":
        replacement = (
            cv2.GaussianBlur(rgb, (0, 0), 12)
            if texture_source is None
            else texture_source.astype(np.uint8)
        )
    elif operator == "frequency":
        low = cv2.GaussianBlur(rgb, (0, 0), 4)
        replacement = np.clip(low.astype(float) + 0.15 * (rgb.astype(float) - low), 0, 255)
    elif operator == "inpaint":
        replacement = cv2.inpaint(rgb, _binary(region) * 255, 5, cv2.INPAINT_TELEA)
    else:
        raise ValueError(f"unknown intervention operator: {operator}")
    output: np.ndarray = np.clip(rgb * (1 - alpha) + replacement * alpha, 0, 255).astype(np.uint8)
    return output


def edit_artifact_score(original: np.ndarray, edited: np.ndarray, region: np.ndarray) -> float:
    """Conservative no-reference proxy; high boundary discontinuities are rejected."""

    binary = _binary(region)
    boundary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
    if not boundary.any():
        return 1.0
    delta = np.mean(np.abs(original.astype(float) - edited.astype(float)), axis=2) / 255
    interior = binary > 0
    boundary_delta = float(np.mean(delta[boundary]))
    interior_delta = float(np.mean(delta[interior])) if interior.any() else 0
    return float(np.clip(boundary_delta / max(interior_delta, 1e-6), 0, 1))


def build_interventions_for_view(
    view_directory: Path,
    *,
    target_class: str,
    config: InterventionConfig,
    depth: np.ndarray | None = None,
) -> list[InterventionRecord]:
    """Materialize every registered operator/fraction for target and matched control."""

    with Image.open(view_directory / "rgb.png") as source:
        image = np.asarray(source.convert("RGB"))
    with Image.open(view_directory / f"{target_class}_mask.png") as source:
        target = _binary(np.asarray(source))
    if not target.any():
        return []
    matched = None
    for attempt in range(20):
        candidate = match_distractor(
            image,
            target,
            depth=depth,
            candidates=config.random_candidates,
            seed=config.seed + attempt,
        )
        if max(candidate[2].values(), default=0) <= config.maximum_match_smd:
            matched = candidate
            break
    if matched is None:
        raise DataValidationError(
            f"{view_directory} could not produce a distractor with every "
            f"|SMD| <= {config.maximum_match_smd} after 20 seeded attempts"
        )
    distractor, control_values, differences = matched
    target_values = region_covariates(image, target, depth=depth)
    output = view_directory / "interventions"
    output.mkdir(parents=True, exist_ok=True)
    records: list[InterventionRecord] = []
    for operator in config.operators:
        for fraction in config.fractions:
            for sequence, full_mask in (
                ("oracle_deletion", target),
                ("distractor_deletion", distractor),
            ):
                donor_mask = distractor if sequence == "oracle_deletion" else target
                region = fractional_region(
                    full_mask, rank_target_pixels(image, full_mask), fraction
                )
                donor_region = fractional_region(
                    donor_mask, rank_target_pixels(image, donor_mask), fraction
                )
                texture_source = (
                    texture_replacement(image, region, donor_region)
                    if operator == "texture"
                    else None
                )
                edited = apply_operator(
                    image,
                    region,
                    operator,
                    feather_px=config.feather_px,
                    texture_source=texture_source,
                )
                filename = f"{sequence}__{operator}__{int(fraction * 100):03d}.png"
                path = output / filename
                Image.fromarray(edited).save(path, optimize=True)
                region_path = output / filename.replace(".png", "__region.png")
                Image.fromarray(region * 255).save(region_path)
                artifact = edit_artifact_score(image, edited, region)
                record = InterventionRecord(
                    view_id=view_directory.name,
                    sequence=cast(TraceSequence, sequence),
                    operator=operator,
                    fraction=fraction,
                    target_class=cast(FeatureClass, target_class),
                    image_path=Path("interventions") / filename,
                    image_sha256=sha256_file(path),
                    region_sha256=sha256_file(region_path),
                    target_covariates=target_values,
                    control_covariates=control_values,
                    standardized_mean_differences=differences,
                    artifact_score=artifact,
                    accepted=(
                        max(differences.values(), default=0) <= config.maximum_match_smd
                        and (operator != "inpaint" or artifact <= 0.75)
                    ),
                )
                records.append(record)
    write_json(
        output / "manifest.json",
        [record.model_dump(mode="json") for record in records],
    )
    write_json(
        output / "balance.json",
        {
            "target": target_values.model_dump(),
            "control": control_values.model_dump(),
            "standardized_mean_differences": differences,
            "maximum_allowed": config.maximum_match_smd,
            "accepted": max(differences.values(), default=0) <= config.maximum_match_smd,
        },
    )
    return records
