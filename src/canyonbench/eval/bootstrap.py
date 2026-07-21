"""Cluster bootstrap over contiguous trajectory segments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from canyonbench.exceptions import DataValidationError


def segment_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    segment_column: str = "segment_id",
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> dict[str, Any]:
    if segment_column not in frame:
        raise DataValidationError(f"Bootstrap input has no {segment_column} column")
    if iterations < 100:
        raise DataValidationError("At least 100 bootstrap iterations are required")
    if not 0 < confidence < 1:
        raise DataValidationError("confidence must be between zero and one")
    clean = frame.loc[frame[segment_column].notna()].copy()
    segment_ids = clean[segment_column].drop_duplicates().tolist()
    if not segment_ids:
        raise DataValidationError("Bootstrap input has no non-null trajectory segments")
    grouped = {segment: clean.loc[clean[segment_column] == segment] for segment in segment_ids}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        sampled_ids = rng.choice(segment_ids, size=len(segment_ids), replace=True)
        sample_parts = []
        for instance, segment in enumerate(sampled_ids):
            part = grouped[segment].copy()
            part["_bootstrap_segment"] = f"{segment}:{instance}"
            sample_parts.append(part)
        value = float(statistic(pd.concat(sample_parts, ignore_index=True)))
        if np.isfinite(value):
            values.append(value)
    estimate = float(statistic(clean))
    if not values:
        return {
            "estimate": estimate,
            "lower": None,
            "upper": None,
            "confidence": confidence,
            "iterations": iterations,
            "effective_segments": len(segment_ids),
        }
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(values, [alpha, 1 - alpha])
    return {
        "estimate": estimate,
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "iterations": iterations,
        "effective_segments": len(segment_ids),
    }
