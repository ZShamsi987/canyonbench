"""Viewpoint equivariance of model grounding under the known imaging transform.

The generator applies one transform to the image and to every mask, so two views
of the same site are related by a transform the benchmark knows exactly. Both
views are rendered from the same source raster, so if ``H_a`` and ``H_b`` are
their source-to-view homographies, the map between the views is ``H_b @ inv(H_a)``
with no estimation and no correspondence search.

That makes a grounding question available that costs no extra inference: push the
cells a model named in one view through that transform and compare them with the
cells it named in the other. A model that grounds on scene geometry should
commute with the transform; a model that pattern-matches the rendered frame need
not. Because Tier A already collects ``evidence_cells`` on every view, this is an
analysis over responses the run has already paid for.

Two view pairs matter and they probe different symmetries. Holding altitude fixed
and changing nadir to oblique is a projective action at constant ground sample
distance. Holding geometry fixed and changing altitude is a scale action, which
is the same manipulation the extinction ladder uses, so equivariance can be read
against apparent feature width.

This measures consistency, not correctness. A model that answers identically
everywhere is perfectly equivariant and useless, so every report here carries the
matched random-cell baseline and the selected-cell count beside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

__all__ = [
    "CellGrid",
    "ViewPair",
    "answer_invariance",
    "cell_centres",
    "grounding_equivariance",
    "map_cells",
]


@dataclass(frozen=True)
class CellGrid:
    """The evidence grid a response was constrained to."""

    grid_size: int
    width_px: int
    height_px: int

    def bounds(self, cell: str) -> tuple[float, float, float, float]:
        """Return the pixel bounds of one cell, matching ``_cell_area``."""

        row, column = (int(value) for value in cell.split(","))
        y0 = row * self.height_px / self.grid_size
        y1 = (row + 1) * self.height_px / self.grid_size
        x0 = column * self.width_px / self.grid_size
        x1 = (column + 1) * self.width_px / self.grid_size
        return x0, y0, x1, y1

    def locate(self, x: float, y: float) -> str | None:
        """Return the cell containing a pixel, or None if it left the frame."""

        if not (0 <= x < self.width_px and 0 <= y < self.height_px):
            return None
        column = min(self.grid_size - 1, int(x * self.grid_size / self.width_px))
        row = min(self.grid_size - 1, int(y * self.grid_size / self.height_px))
        return f"{row},{column}"

    @property
    def cells(self) -> list[str]:
        return [
            f"{row},{column}"
            for row in range(self.grid_size)
            for column in range(self.grid_size)
        ]


@dataclass(frozen=True)
class ViewPair:
    """Two views of one site and the exact transform between them.

    ``source_to_a`` and ``source_to_b`` are the ``PixelWindow.homography``
    matrices produced by ``camera_window`` for each view.
    """

    source_to_a: np.ndarray
    source_to_b: np.ndarray

    @property
    def a_to_b(self) -> np.ndarray:
        """The view-to-view homography, composed rather than estimated."""

        return np.asarray(self.source_to_b, dtype=float) @ np.linalg.inv(
            np.asarray(self.source_to_a, dtype=float)
        )


def cell_centres(grid: CellGrid, cells: list[str]) -> np.ndarray:
    """Return the pixel centres of the given cells as an (N, 2) array."""

    points = []
    for cell in cells:
        x0, y0, x1, y1 = grid.bounds(cell)
        points.append(((x0 + x1) / 2, (y0 + y1) / 2))
    return np.asarray(points, dtype=float).reshape(-1, 2)


def _apply(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(-1, 2)
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    projected = homogeneous @ np.asarray(homography, dtype=float).T
    denominator = projected[:, 2:3]
    # A point on or behind the horizon has no finite image; drop it rather than
    # letting a near-zero denominator manufacture a coordinate.
    safe = np.abs(denominator[:, 0]) > 1e-9
    result = np.full((len(points), 2), np.nan)
    result[safe] = projected[safe, :2] / denominator[safe]
    return result


def map_cells(
    cells: list[str],
    pair: ViewPair,
    grid_a: CellGrid,
    grid_b: CellGrid,
) -> tuple[set[str], int]:
    """Map cells from view A into view B.

    Returns the mapped cell set and the number of source cells that left view B's
    frame. Cells that leave the frame are reported rather than silently dropped:
    an oblique view does not cover the same ground as its nadir counterpart, so
    coverage is part of the measurement.
    """

    if not cells:
        return set(), 0
    centres = cell_centres(grid_a, cells)
    projected = _apply(pair.a_to_b, centres)
    mapped: set[str] = set()
    lost = 0
    for x, y in projected:
        if not np.isfinite(x) or not np.isfinite(y):
            lost += 1
            continue
        cell = grid_b.locate(float(x), float(y))
        if cell is None:
            lost += 1
        else:
            mapped.add(cell)
    return mapped, lost


def _jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else float("nan")


def grounding_equivariance(
    cells_a: list[str],
    cells_b: list[str],
    pair: ViewPair,
    grid_a: CellGrid,
    grid_b: CellGrid,
    *,
    seed: int = 2026,
    baseline_draws: int = 256,
) -> dict[str, float]:
    """Score whether a model's grounding commutes with the imaging transform.

    ``equivariance`` is the Jaccard overlap between the cells named in view A,
    carried into view B by the exact transform, and the cells the model named in
    view B. ``baseline`` is the same quantity for random cell sets of matched
    size, so a model that names many cells cannot score well by covering the
    frame. ``lift`` is the difference, and is the number to report.
    """

    mapped, lost = map_cells(cells_a, pair, grid_a, grid_b)
    observed = set(cells_b)
    equivariance = _jaccard(mapped, observed)

    generator = np.random.default_rng(seed)
    population = grid_b.cells
    draws = []
    if mapped and observed and len(population) >= max(len(mapped), len(observed)):
        for _ in range(baseline_draws):
            sample = set(
                generator.choice(len(population), size=len(mapped), replace=False).tolist()
            )
            draws.append(_jaccard({population[index] for index in sample}, observed))
    baseline = float(np.mean(draws)) if draws else float("nan")

    return {
        "equivariance": equivariance,
        "baseline": baseline,
        "lift": equivariance - baseline,
        "cells_a": float(len(cells_a)),
        "cells_b": float(len(observed)),
        "cells_mapped": float(len(mapped)),
        "cells_left_frame": float(lost),
    }


def answer_invariance(answers: dict[str, str]) -> dict[str, float]:
    """Fraction of view pairs for one site and class that answer identically.

    Callers pass only views where the feature is resolvable, so a change of
    answer is a change under a transform that preserved the evidence. Views in
    the extinction band must be excluded by the caller: there a changed answer is
    the correct behaviour, not an inconsistency.
    """

    labels = [value for value in answers.values() if value in ("yes", "no", "abstain")]
    pairs = list(combinations(labels, 2))
    if not pairs:
        return {"invariance": float("nan"), "views": float(len(labels))}
    agreeing = sum(1 for first, second in pairs if first == second)
    return {
        "invariance": agreeing / len(pairs),
        "views": float(len(labels)),
        "pairs": float(len(pairs)),
    }
