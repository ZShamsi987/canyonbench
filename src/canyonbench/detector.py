"""Non-language detector reference: what the imagery supports, without language.

Section 12 requires a detector or segmenter as an upper reference, so a failure
at 16 km can be attributed to the imagery rather than to the model. This module
holds everything about that reference that can be decided without a GPU: how a
segmentation map becomes the same structured answer a VLM returns, and how the
registered feature classes map onto a segmentation model's own label set.

The label mapping is resolved against the served model's `id2label` at startup
rather than hard-coded to index numbers, so a checkpoint with a different class
order cannot silently mis-map a class.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from canyonbench.constants import TRACE_FEATURES

# Substrings matched against the segmentation model's own label names. ADE20K
# labels are semicolon-joined synonym lists such as "road;route", so substring
# matching is the intended lookup.
SEGMENTATION_LABEL_TERMS: dict[str, tuple[str, ...]] = {
    "water": ("water", "sea", "river", "lake", "waterfall", "swimming pool"),
    "road": ("road", "route", "path", "runway", "dirt track", "highway"),
    "field": ("field", "grass", "farm"),
}
# A class must occupy at least this fraction of the view before the reference
# calls it present. Below it, the signal is indistinguishable from segmentation
# noise at these ground sample distances.
DEFAULT_PRESENCE_FRACTION = 0.002


@dataclass(frozen=True)
class DetectorQuery:
    """The parts of a benchmark prompt a non-language detector can act on."""

    target_class: str
    grid_size: int
    cell_budget: int


def parse_detector_prompt(
    prompt: str,
    *,
    default_grid_size: int = 6,
    default_cell_budget: int = 6,
) -> DetectorQuery:
    """Recover the target class, grid, and cell budget from a rendered prompt.

    The detector receives the same prompt string as every other model, so it must
    read the query out of it rather than being told separately.
    """

    lowered = prompt.lower()
    found = [feature for feature in TRACE_FEATURES if feature in lowered]
    if len(found) != 1:
        raise ValueError(
            f"prompt must name exactly one of {TRACE_FEATURES}; matched {found or 'none'}"
        )
    grid = re.search(r"(\d+)\s*x\s*\1\b", lowered)
    budget = re.search(r"at most (\d+)", lowered) or re.search(r"up to (\d+)", lowered)
    return DetectorQuery(
        target_class=found[0],
        grid_size=int(grid.group(1)) if grid else default_grid_size,
        cell_budget=int(budget.group(1)) if budget else default_cell_budget,
    )


def resolve_label_ids(id2label: dict[int, str], target_class: str) -> tuple[int, ...]:
    """Map a registered feature class onto a checkpoint's own label indices."""

    if target_class not in SEGMENTATION_LABEL_TERMS:
        raise KeyError(f"unknown target class {target_class!r}")
    terms = SEGMENTATION_LABEL_TERMS[target_class]
    matched = sorted(
        index
        for index, label in id2label.items()
        if any(term in str(label).lower() for term in terms)
    )
    if not matched:
        raise ValueError(
            f"no label in the segmentation checkpoint matches {target_class!r}; "
            f"expected one of {terms}"
        )
    return tuple(matched)


def _cell_scores(mask: np.ndarray, grid_size: int) -> dict[str, float]:
    height, width = mask.shape
    scores: dict[str, float] = {}
    for row in range(grid_size):
        y0, y1 = round(row * height / grid_size), round((row + 1) * height / grid_size)
        for column in range(grid_size):
            x0, x1 = round(column * width / grid_size), round((column + 1) * width / grid_size)
            window = mask[y0:y1, x0:x1]
            if window.size:
                scores[f"{row},{column}"] = float(np.mean(window))
    return scores


def response_from_segmentation(
    segmentation: np.ndarray,
    label_ids: tuple[int, ...],
    query: DetectorQuery,
    *,
    presence_fraction: float = DEFAULT_PRESENCE_FRACTION,
) -> dict[str, Any]:
    """Turn a predicted segmentation map into the benchmark's structured answer.

    The detector never abstains: it reports what the pixels support, which is
    exactly the reference the protocol wants it to provide.
    """

    if segmentation.ndim != 2:
        raise ValueError("segmentation must be a 2-D label map")
    mask = np.isin(segmentation, list(label_ids))
    coverage = float(np.mean(mask))
    scores = _cell_scores(mask, query.grid_size)
    ordered = sorted(scores.items(), key=lambda item: -item[1])
    ranked = [cell for cell, score in ordered if score > 0]
    selected = ranked[: query.cell_budget]
    present = coverage >= presence_fraction
    if not present:
        return {
            "answer": "no",
            # Confidence in the reported answer, not in presence: a clean, empty
            # view is a confident negative.
            "confidence": round(100 * (1 - min(1.0, coverage / presence_fraction))),
            "evidence_cells": [],
            "cell_ranking": [],
        }
    # Saturate at 20 percent coverage so a large feature is not over-credited.
    confidence = round(100 * min(1.0, 0.5 + coverage / 0.4))
    return {
        "answer": "yes",
        "confidence": max(50, min(100, confidence)),
        "evidence_cells": selected,
        "cell_ranking": selected,
    }
