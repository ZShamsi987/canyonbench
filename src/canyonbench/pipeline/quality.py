"""Deterministic image-quality controls for ascent association analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def image_quality_controls(path: str | Path) -> dict[str, Any]:
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
        gray_image = source.convert("L")
        gray = np.asarray(gray_image, dtype=np.float32) / 255.0
        edges = np.asarray(gray_image.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        np.maximum(maximum, 1e-6),
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )
    return {
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "sharpness_edge_var": float(edges.var()),
        "saturation_mean": float(saturation.mean()),
        "clipped_high_fraction": float((gray >= 250 / 255).mean()),
        "clipped_low_fraction": float((gray <= 5 / 255).mean()),
        "width_px": int(rgb.shape[1]),
        "height_px": int(rgb.shape[0]),
    }
