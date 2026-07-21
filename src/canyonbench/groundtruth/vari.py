"""Visible Atmospherically Resistant Index as a calibrated weak label only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import cast

import numpy as np

from canyonbench.exceptions import DataValidationError


@dataclass(frozen=True)
class VariCalibration:
    threshold: float
    mean_iou: float
    thresholds: list[float]
    mean_ious: list[float]
    n_calibration: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def vari(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise DataValidationError("VARI input must be an HxWx3 RGB image")
    if image.max(initial=0) > 1:
        image = image / 255.0
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    denominator = green + red - blue
    return cast(
        np.ndarray,
        np.divide(
            green - red,
            denominator,
            out=np.zeros_like(green),
            where=np.abs(denominator) > 1e-6,
        ),
    )


def intersection_over_union(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise DataValidationError("IoU arrays must have the same shape")
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, target).sum() / union)


def calibrate_vari(
    vari_images: list[np.ndarray],
    human_masks: list[np.ndarray],
    *,
    thresholds: np.ndarray | None = None,
) -> VariCalibration:
    if not vari_images or len(vari_images) != len(human_masks):
        raise DataValidationError("VARI calibration requires paired, non-empty images and masks")
    candidates = np.arange(-0.10, 0.301, 0.01) if thresholds is None else thresholds
    if candidates.ndim != 1 or len(candidates) == 0:
        raise DataValidationError("VARI thresholds must be a non-empty 1D array")
    means = [
        float(
            np.mean(
                [
                    intersection_over_union(index_image > threshold, mask)
                    for index_image, mask in zip(vari_images, human_masks, strict=True)
                ]
            )
        )
        for threshold in candidates
    ]
    best_index = int(np.argmax(means))
    return VariCalibration(
        threshold=float(candidates[best_index]),
        mean_iou=means[best_index],
        thresholds=[float(value) for value in candidates],
        mean_ious=means,
        n_calibration=len(vari_images),
    )
