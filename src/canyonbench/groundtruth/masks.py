"""Mask validation, cover fractions, agreement, and 4x4 grounding grids."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image

from canyonbench.constants import GRID_THRESHOLD, GROUNDING_GRID_SIZE
from canyonbench.exceptions import DataValidationError


def load_binary_mask(path: str | Path, expected_size: tuple[int, int] | None = None) -> np.ndarray:
    source = Path(path)
    with Image.open(source) as image:
        if image.mode != "L":
            raise DataValidationError(f"Mask must be single-channel 8-bit PNG: {source}")
        if expected_size is not None and image.size != expected_size:
            raise DataValidationError(
                f"Mask size {image.size} does not match frame size {expected_size}: {source}"
            )
        array = np.asarray(image)
    values = set(np.unique(array).tolist())
    if not values <= {0, 255}:
        raise DataValidationError(f"Mask contains values other than 0 and 255: {source}")
    return cast(np.ndarray, array == 255)


def vegetation_fraction(mask: np.ndarray) -> float:
    if mask.ndim != 2 or mask.size == 0:
        raise DataValidationError("Vegetation mask must be a non-empty 2D array")
    return float(np.asarray(mask, dtype=bool).mean())


def grid_labels(
    mask: np.ndarray,
    *,
    n: int = GROUNDING_GRID_SIZE,
    threshold: float = GRID_THRESHOLD,
) -> dict[str, bool]:
    if mask.ndim != 2 or mask.size == 0:
        raise DataValidationError("Grounding mask must be a non-empty 2D array")
    if n <= 0 or not 0 <= threshold <= 1:
        raise DataValidationError("Invalid grid configuration")
    height, width = mask.shape
    output: dict[str, bool] = {}
    for row in range(n):
        for column in range(n):
            region = mask[
                row * height // n : (row + 1) * height // n,
                column * width // n : (column + 1) * width // n,
            ]
            output[f"{row},{column}"] = bool(np.asarray(region, dtype=bool).mean() >= threshold)
    return output


def dice(left: np.ndarray, right: np.ndarray) -> float:
    left_bool = np.asarray(left, dtype=bool)
    right_bool = np.asarray(right, dtype=bool)
    if left_bool.shape != right_bool.shape:
        raise DataValidationError("Dice masks must have the same shape")
    denominator = left_bool.sum() + right_bool.sum()
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(left_bool, right_bool).sum() / denominator)


def small_components(mask: np.ndarray, minimum_pixels: int = 4) -> list[int]:
    """Return sizes of connected foreground components below annotation rule A-5."""

    foreground = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(foreground, dtype=bool)
    violations: list[int] = []
    height, width = foreground.shape
    for start_y, start_x in zip(*np.where(foreground & ~visited), strict=True):
        if visited[start_y, start_x]:
            continue
        queue = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        count = 0
        while queue:
            y, x = queue.popleft()
            count += 1
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                new_y, new_x = y + dy, x + dx
                if (
                    0 <= new_y < height
                    and 0 <= new_x < width
                    and foreground[new_y, new_x]
                    and not visited[new_y, new_x]
                ):
                    visited[new_y, new_x] = True
                    queue.append((new_y, new_x))
        if count < minimum_pixels:
            violations.append(count)
    return violations
