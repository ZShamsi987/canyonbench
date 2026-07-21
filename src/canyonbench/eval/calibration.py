"""Secondary calibration diagnostics, kept separate by confidence mechanism."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyonbench.exceptions import DataValidationError


def expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if confidence.shape != correct.shape or confidence.ndim != 1 or len(confidence) == 0:
        raise DataValidationError("Calibration inputs must be paired non-empty vectors")
    if np.any((confidence < 0) | (confidence > 1)):
        raise DataValidationError("Confidence values must lie in [0, 1]")
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.minimum(np.digitize(confidence, edges[1:], right=True), bins - 1)
    ece = 0.0
    reliability: list[dict[str, Any]] = []
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum())
        if not count:
            reliability.append(
                {"lower": float(edges[index]), "upper": float(edges[index + 1]), "n": 0}
            )
            continue
        mean_confidence = float(confidence[selected].mean())
        accuracy = float(correct[selected].mean())
        ece += count / len(confidence) * abs(accuracy - mean_confidence)
        reliability.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "n": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {"ece": float(ece), "bins": reliability, "n": len(confidence)}


def calibration_by_mechanism(
    frame: pd.DataFrame,
    *,
    confidence_column: str = "confidence",
    correct_column: str = "correct",
    mechanism_column: str = "mechanism",
    bins: int = 10,
) -> dict[str, Any]:
    missing = sorted({confidence_column, correct_column, mechanism_column} - set(frame.columns))
    if missing:
        raise DataValidationError(f"Calibration table is missing: {missing}")
    return {
        str(mechanism): expected_calibration_error(
            group[confidence_column].to_numpy(), group[correct_column].to_numpy(), bins=bins
        )
        for mechanism, group in frame.groupby(mechanism_column)
    }
