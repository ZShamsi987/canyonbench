"""Single-factor image degradations calibrated from the real balloon flight."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from canyonbench.io import sha256_file
from canyonbench.trace.schemas import QualityCalibration, QualityParameters


@dataclass(frozen=True)
class FrameQuality:
    path: str
    laplacian_variance: float
    dark_channel_mean: float
    luminance_mean: float
    clipped_dark_fraction: float
    clipped_bright_fraction: float
    saturation_mean: float
    contrast_std: float
    jpeg_blockiness: float
    horizon_frequency: float
    obstruction_fraction: float


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")
    output: np.ndarray = np.clip(image, 0, 255).astype(np.uint8)
    return output


def measure_quality(image: np.ndarray, *, path: str = "") -> FrameQuality:
    """Measure the registered quality proxies from one RGB frame."""

    rgb = _as_rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    dark = cv2.erode(np.min(rgb, axis=2), np.ones((15, 15), np.uint8))
    vertical = np.abs(np.diff(gray.astype(np.float32), axis=0))
    horizontal = np.abs(np.diff(gray.astype(np.float32), axis=1))
    # JPEG blocking proxy: discontinuities at 8-pixel boundaries relative to neighbours.
    boundary = np.mean(
        [
            np.mean(np.abs(gray[:, offset].astype(float) - gray[:, offset - 1].astype(float)))
            for offset in range(8, gray.shape[1], 8)
        ]
        or [0.0]
    )
    ordinary = np.mean(horizontal) if horizontal.size else 0.0
    edge_density_rows = np.mean(vertical, axis=1) if vertical.size else np.zeros(1)
    horizon = float(np.max(edge_density_rows) / 255)
    lower = gray[int(gray.shape[0] * 0.88) :]
    obstruction = float(np.mean(lower < 12)) if lower.size else 0.0
    return FrameQuality(
        path=path,
        laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        dark_channel_mean=float(np.mean(dark) / 255),
        luminance_mean=float(np.mean(gray) / 255),
        clipped_dark_fraction=float(np.mean(gray <= 3)),
        clipped_bright_fraction=float(np.mean(gray >= 252)),
        saturation_mean=float(np.mean(hsv[..., 1]) / 255),
        contrast_std=float(np.std(gray) / 255),
        jpeg_blockiness=float(max(0.0, boundary - ordinary) / 255),
        horizon_frequency=horizon,
        obstruction_fraction=obstruction,
    )


def calibrate_from_frames(paths: list[Path]) -> QualityCalibration:
    """Return distribution quantiles used to sample realistic single degradations."""

    rows: list[FrameQuality] = []
    hashes: list[str] = []
    for path in sorted(paths):
        with Image.open(path) as source:
            rows.append(measure_quality(np.asarray(source.convert("RGB")), path=str(path)))
        hashes.append(sha256_file(path))
    if not rows:
        raise ValueError("at least one calibration frame is required")
    fields = [field for field in FrameQuality.__dataclass_fields__ if field != "path"]
    quantiles = {
        field: {
            str(q): float(np.quantile([getattr(row, field) for row in rows], q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        }
        for field in fields
    }
    return QualityCalibration(
        frame_count=len(rows),
        source_sha256=hashes,
        quantiles=quantiles,
    )


def apply_degradation(image: np.ndarray, quality: QualityParameters) -> np.ndarray:
    """Apply exactly one preregistered degradation to RGB pixels."""

    rgb = _as_rgb(image)
    if quality.degradation == "none":
        return rgb.copy()
    if quality.degradation == "blur":
        return cv2.GaussianBlur(rgb, (0, 0), quality.blur_sigma)
    if quality.degradation == "haze":
        airlight = np.full_like(rgb, 235)
        return cv2.addWeighted(rgb, 1 - quality.haze_strength, airlight, quality.haze_strength, 0)
    if quality.degradation == "exposure":
        linear = rgb.astype(np.float32) / 255
        return np.clip(linear * (2**quality.exposure_ev) * 255, 0, 255).astype(np.uint8)
    if quality.degradation in {"saturation", "contrast"}:
        value = rgb.astype(np.float32) / 255
        if quality.degradation == "saturation":
            gray = np.mean(value, axis=2, keepdims=True)
            value = gray + (value - gray) * quality.saturation_scale
        else:
            value = 0.5 + (value - 0.5) * quality.contrast_scale
        return np.clip(value * 255, 0, 255).astype(np.uint8)
    if quality.degradation == "jpeg":
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality.jpeg_quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return np.asarray(compressed.convert("RGB"))
    raise AssertionError(f"unhandled degradation {quality.degradation}")


def sample_degradations(
    view_ids: list[str],
    *,
    count: int,
    seed: int,
    calibration: QualityCalibration | None = None,
    calibration_source_sha256: str | None = None,
) -> dict[str, QualityParameters]:
    """Select a stratifiable fixed subset and one factor per selected view."""

    if count > len(view_ids):
        raise ValueError("degraded count cannot exceed view count")
    generator = np.random.default_rng(seed)
    selected = set(generator.choice(sorted(view_ids), size=count, replace=False).tolist())
    names = ["blur", "haze", "exposure", "saturation", "contrast", "jpeg"]
    output: dict[str, QualityParameters] = {}
    quantiles = calibration.quantiles if calibration is not None else {}

    def q(metric: str, probability: str, fallback: float) -> float:
        return float(quantiles.get(metric, {}).get(probability, fallback))

    sharpness_ratio = q("laplacian_variance", "0.95", 400) / max(
        q("laplacian_variance", "0.05", 50), 1
    )
    blur_max = float(np.clip(np.sqrt(sharpness_ratio) / 2, 1.2, 3.5))
    haze_max = float(
        np.clip(
            0.15 + q("dark_channel_mean", "0.95", 0.25) - q("dark_channel_mean", "0.25", 0.08),
            0.18,
            0.5,
        )
    )
    low_saturation = float(
        np.clip(
            q("saturation_mean", "0.05", 0.2) / max(q("saturation_mean", "0.5", 0.4), 1e-3),
            0.3,
            0.85,
        )
    )
    low_contrast = float(
        np.clip(
            q("contrast_std", "0.05", 0.08) / max(q("contrast_std", "0.5", 0.18), 1e-3),
            0.3,
            0.85,
        )
    )
    jpeg_min = int(np.clip(75 - q("jpeg_blockiness", "0.95", 0.05) * 600, 20, 60))
    dark_ev = float(
        np.clip(
            np.log2(max(q("luminance_mean", "0.05", 0.2), 0.02) / 0.5),
            -2,
            -0.4,
        )
    )
    bright_ev = float(
        np.clip(
            np.log2(max(q("luminance_mean", "0.95", 0.8), 0.5) / 0.5),
            0.4,
            2,
        )
    )
    for view_id in sorted(view_ids):
        if view_id not in selected:
            output[view_id] = QualityParameters()
            continue
        degradation = str(generator.choice(names))
        values: dict[str, object] = {
            "degradation": degradation,
            "calibration_source_sha256": calibration_source_sha256,
        }
        if degradation == "blur":
            values["blur_sigma"] = float(generator.uniform(0.8, blur_max))
        elif degradation == "haze":
            values["haze_strength"] = float(generator.uniform(0.12, haze_max))
        elif degradation == "exposure":
            values["exposure_ev"] = float(generator.choice([dark_ev, bright_ev]))
        elif degradation == "saturation":
            values["saturation_scale"] = float(generator.uniform(low_saturation, 0.9))
        elif degradation == "contrast":
            values["contrast_scale"] = float(generator.uniform(low_contrast, 0.9))
        else:
            values["jpeg_quality"] = int(generator.integers(jpeg_min, 66))
        output[view_id] = QualityParameters.model_validate(values)
    return output
