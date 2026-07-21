"""Evaluate weak label masks against human gold without promoting them to truth."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyonbench.exceptions import DataValidationError
from canyonbench.groundtruth.masks import dice
from canyonbench.groundtruth.vari import intersection_over_union


def mask_comparison(prediction: np.ndarray, human: np.ndarray) -> dict[str, float | int | None]:
    prediction = np.asarray(prediction, dtype=bool)
    human = np.asarray(human, dtype=bool)
    if prediction.shape != human.shape:
        raise DataValidationError("Weak-label and human masks must have the same shape")
    true_positive = int(np.logical_and(prediction, human).sum())
    false_positive = int(np.logical_and(prediction, ~human).sum())
    false_negative = int(np.logical_and(~prediction, human).sum())
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    )
    return {
        "human_fraction": float(human.mean()),
        "weak_fraction": float(prediction.mean()),
        "signed_cover_error": float(prediction.mean() - human.mean()),
        "iou": intersection_over_union(prediction, human),
        "dice": dice(prediction, human),
        "precision": precision,
        "recall": recall,
    }


def compare_weak_label_set(
    images: list[str],
    predictions: list[np.ndarray],
    humans: list[np.ndarray],
    *,
    source: str,
) -> pd.DataFrame:
    if not images or not (len(images) == len(predictions) == len(humans)):
        raise DataValidationError("Weak-label comparison inputs must be paired and non-empty")
    rows: list[dict[str, Any]] = []
    for image, prediction, human in zip(images, predictions, humans, strict=True):
        rows.append(
            {"image": image, "weak_label_source": source, **mask_comparison(prediction, human)}
        )
    return pd.DataFrame(rows)
