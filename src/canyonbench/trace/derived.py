"""Objective mask geometry, visibility, and grid-occupancy derivations."""

from __future__ import annotations

import cv2
import numpy as np

from canyonbench.trace.schemas import CaseType, FeatureDerived


def apparent_widths(mask: np.ndarray) -> tuple[float, float]:
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return 0.0, 0.0
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    widths = distance[binary > 0] * 2
    return float(np.min(widths)), float(np.median(widths))


def boundary_distance(mask: np.ndarray) -> float:
    """Closest approach of the mask to any frame border, in pixels."""

    binary = mask > 0
    if not binary.any():
        return 0.0
    rows, columns = np.where(binary)
    height, width = binary.shape
    return float(np.min(np.concatenate([rows, columns, height - rows - 1, width - columns - 1])))


def interior_distance(mask: np.ndarray) -> float:
    """Deepest penetration of the mask away from every frame border, in pixels.

    Resolvability must not depend on ``boundary_distance``: a through-going road
    or river always reaches the border, so its closest approach is zero even when
    it crosses the whole view. What separates a usable feature from a clipped
    corner sliver is how far inside the frame the feature actually reaches.
    """

    binary = mask > 0
    if not binary.any():
        return 0.0
    rows, columns = np.where(binary)
    height, width = binary.shape
    return float(
        np.max(
            np.minimum.reduce(
                [rows, columns, height - rows - 1, width - columns - 1],
            )
        )
    )


def grid_occupancy(mask: np.ndarray, grid_size: int) -> list[str]:
    binary = mask > 0
    height, width = binary.shape
    cells: list[str] = []
    for row in range(grid_size):
        y0, y1 = round(row * height / grid_size), round((row + 1) * height / grid_size)
        for column in range(grid_size):
            x0, x1 = round(column * width / grid_size), round((column + 1) * width / grid_size)
            if binary[y0:y1, x0:x1].any():
                cells.append(f"{row},{column}")
    return cells


def grid_target_pixel_counts(mask: np.ndarray, grid_size: int) -> dict[str, int]:
    """Count exact target-mask pixels in every occupied grid cell."""

    binary = mask > 0
    height, width = binary.shape
    counts: dict[str, int] = {}
    for row in range(grid_size):
        y0, y1 = round(row * height / grid_size), round((row + 1) * height / grid_size)
        for column in range(grid_size):
            x0, x1 = round(column * width / grid_size), round((column + 1) * width / grid_size)
            count = int(binary[y0:y1, x0:x1].sum())
            if count:
                counts[f"{row},{column}"] = count
    return counts


def local_contrast(image: np.ndarray, mask: np.ndarray, ring_px: int = 8) -> float:
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return 0.0
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(float) / 255
    kernel = np.ones((ring_px * 2 + 1, ring_px * 2 + 1), np.uint8)
    ring = cv2.dilate(binary, kernel) > 0
    ring &= binary == 0
    if not ring.any():
        return 0.0
    return float(min(1.0, abs(np.mean(gray[binary > 0]) - np.mean(gray[ring]))))


def aliasing_risk(mask: np.ndarray) -> float:
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return 0.0
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    area = float(np.sum(binary))
    return float(np.clip(perimeter / max(4 * np.sqrt(area), 1) / 8, 0, 1))


def derive_feature(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    case_type: CaseType,
    occlusion_fraction: float = 0,
    minimum_resolvable_width_px: float = 1,
    extinction_width_px: float = 1,
    minimum_component_px: int = 4,
    minimum_boundary_distance_px: float = 2,
    minimum_local_contrast: float = 0.03,
    maximum_occlusion_fraction: float = 0.25,
    detector_score: np.ndarray | None = None,
    detector_min_score: float = 0.25,
) -> FeatureDerived:
    binary = mask > 0
    area = int(binary.sum())
    minimum, median = apparent_widths(binary)
    contrast = local_contrast(image, binary)
    component_total, _, component_stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8)
    )
    component_areas = (
        component_stats[1:, cv2.CC_STAT_AREA] if component_total > 1 else np.asarray([], dtype=int)
    )
    valid_components = component_areas[component_areas >= minimum_component_px]
    edge_distance = boundary_distance(binary)
    interior_reach = interior_distance(binary)
    detector_values = detector_score[binary] if detector_score is not None and area else None
    detector_mean = (
        float(np.mean(detector_values))
        if detector_values is not None and detector_values.size
        else None
    )
    score = float(
        np.clip(
            0.45 * min(1, median / max(minimum_resolvable_width_px, 1e-9))
            + 0.35 * min(1, contrast / 0.1)
            + 0.2 * (1 - occlusion_fraction),
            0,
            1,
        )
    )
    extinction = bool(
        area > 0
        and median <= extinction_width_px
        and contrast < minimum_local_contrast
        and detector_mean is not None
        and detector_mean < detector_min_score
    )
    resolvable = bool(
        area > 0
        and len(valid_components) > 0
        and median >= minimum_resolvable_width_px
        and interior_reach >= minimum_boundary_distance_px
        and contrast >= minimum_local_contrast
        and occlusion_fraction <= maximum_occlusion_fraction
    )
    return FeatureDerived(
        present=bool(area),
        case_type=(
            "extinction" if extinction else "negative" if case_type == "negative" else "positive"
        ),
        area_px=area,
        area_fraction=area / binary.size,
        component_count=len(valid_components),
        largest_component_px=(int(np.max(component_areas)) if len(component_areas) else 0),
        minimum_width_px=minimum,
        median_width_px=median,
        boundary_distance_px=edge_distance,
        interior_distance_px=interior_reach,
        occlusion_fraction=occlusion_fraction,
        local_contrast=contrast,
        detector_mean_score=detector_mean,
        aliasing_risk=aliasing_risk(binary),
        resolvability_score=score,
        resolvable=resolvable,
        extinction=extinction,
        grid_occupancy={f"{size}x{size}": grid_occupancy(binary, size) for size in (4, 6, 8)},
        grid_target_pixel_counts={
            f"{size}x{size}": grid_target_pixel_counts(binary, size) for size in (4, 6, 8)
        },
    )
