"""Tier selection and dynamic self-evidence intervention construction."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from canyonbench.io import write_json
from canyonbench.trace.interventions import apply_operator, texture_replacement
from canyonbench.trace.schemas import TraceResponse


def stratified_select(rows: list[dict[str, Any]], count: int, *, seed: int) -> list[dict[str, Any]]:
    """Round-robin sample across group/class/case/geometry/altitude strata."""

    if count > len(rows):
        raise ValueError(f"cannot select {count} rows from {len(rows)}")
    strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        altitude = (
            str(row.get("view_id", "")).split("_")[1] if "_" in str(row.get("view_id")) else ""
        )
        fixed_key = (
            str(row.get("group")),
            str(row.get("target_class")),
            str(row.get("case_type")),
            str(row.get("geometry", row.get("view_id", "").split("_")[-1])),
            altitude,
        )
        strata[fixed_key].append(row)
    generator = np.random.default_rng(seed)
    for stratum_key in strata:
        order = generator.permutation(len(strata[stratum_key]))
        strata[stratum_key] = [strata[stratum_key][int(index)] for index in order]
    selected: list[dict[str, Any]] = []
    keys = sorted(strata)
    while len(selected) < count:
        progressed = False
        for ordered_key in keys:
            if strata[ordered_key] and len(selected) < count:
                selected.append(strata[ordered_key].pop())
                progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda row: (str(row["site_id"]), str(row["view_id"])))


def cell_mask(shape: tuple[int, int], grid_size: int, cells: list[str]) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, np.uint8)
    for cell in cells:
        row, column = (int(value) for value in cell.split(","))
        if not (0 <= row < grid_size and 0 <= column < grid_size):
            raise ValueError(f"cell {cell} is outside {grid_size}x{grid_size}")
        y0, y1 = round(row * height / grid_size), round((row + 1) * height / grid_size)
        x0, x1 = round(column * width / grid_size), round((column + 1) * width / grid_size)
        mask[y0:y1, x0:x1] = 1
    return mask


def matched_control_cells(
    image: np.ndarray,
    *,
    selected: list[str],
    grid_size: int,
    seed: int,
    texture_matched: bool,
) -> list[str]:
    """Choose equally many unselected cells, random or nearest in texture energy."""

    all_cells = [f"{row},{column}" for row in range(grid_size) for column in range(grid_size)]
    candidates = [cell for cell in all_cells if cell not in selected]
    if len(candidates) < len(selected):
        raise ValueError("not enough unselected cells to construct a control")
    generator = np.random.default_rng(seed)
    if not texture_matched:
        return cast(
            list[str],
            generator.choice(candidates, size=len(selected), replace=False).tolist(),
        )
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)

    def energy(cell: str) -> float:
        region = cell_mask(gray.shape, grid_size, [cell]) > 0
        return float(cv2.Laplacian(gray, cv2.CV_64F)[region].var())

    target = np.mean([energy(cell) for cell in selected]) if selected else 0
    return sorted(candidates, key=lambda cell: abs(energy(cell) - target))[: len(selected)]


def materialize_self_sequences(
    image_path: Path,
    response: TraceResponse,
    output_dir: Path,
    *,
    grid_size: int,
    cell_budget: int,
    operator: str = "blur",
    feather_px: int = 5,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Create S3-S5 images using only the model's own claimed cells."""

    response.validate_protocol(grid_size=grid_size, cell_budget=cell_budget)
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    selected = response.cell_ranking
    random_cells = matched_control_cells(
        image, selected=selected, grid_size=grid_size, seed=seed, texture_matched=False
    )
    texture_cells = matched_control_cells(
        image, selected=selected, grid_size=grid_size, seed=seed, texture_matched=True
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for sequence, cells in (
        ("self_deletion", selected),
        ("random_control", random_cells),
        ("texture_control", texture_cells),
    ):
        for count in range(1, len(cells) + 1):
            region = cell_mask(image.shape[:2], grid_size, cells[:count])
            texture_source = None
            if operator == "texture":
                donor_cells = matched_control_cells(
                    image,
                    selected=cells,
                    grid_size=grid_size,
                    seed=seed + count,
                    texture_matched=True,
                )
                donor_region = cell_mask(image.shape[:2], grid_size, donor_cells[:count])
                texture_source = texture_replacement(image, region, donor_region)
            edited = apply_operator(
                image,
                region,
                operator,
                feather_px=feather_px,
                texture_source=texture_source,
            )
            filename = f"{sequence}__{count:02d}-of-{len(cells):02d}.png"
            Image.fromarray(edited).save(output_dir / filename, optimize=True)
            rows.append(
                {
                    "sequence": sequence,
                    "step": count,
                    "total_steps": len(cells),
                    "cells": cells[:count],
                    "image_path": str(output_dir / filename),
                }
            )

    # Sufficiency starts with one selected cell and restores one claimed cell per step.
    all_mask = np.ones(image.shape[:2], np.uint8)
    for count in range(1, len(selected) + 1):
        clear = cell_mask(image.shape[:2], grid_size, selected[:count])
        suppress = all_mask.copy()
        suppress[clear > 0] = 0
        # Sufficiency is defined as preserving selected cells while heavily
        # suppressing the remainder; transparent blur is fixed across operators.
        edited = apply_operator(image, suppress, "blur", feather_px=feather_px)
        filename = f"self_sufficiency__{count:02d}-of-{len(selected):02d}.png"
        Image.fromarray(edited).save(output_dir / filename, optimize=True)
        rows.append(
            {
                "sequence": "self_sufficiency",
                "step": count,
                "total_steps": len(selected),
                "cells": selected[:count],
                "image_path": str(output_dir / filename),
            }
        )
    write_json(output_dir / "manifest.json", rows)
    return rows


def request_id(*parts: object) -> str:
    return hashlib.sha256("\0".join(str(part) for part in parts).encode()).hexdigest()
