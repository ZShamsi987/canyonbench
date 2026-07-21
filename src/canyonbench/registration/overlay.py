"""Warp registered reference rasters and vector layers into frame pixels."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import numpy as np

from canyonbench.exceptions import DataValidationError


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise DataValidationError("Overlay warping requires the registration extra") from exc
    return cv2


def warp_reference_to_frame(
    reference: np.ndarray,
    image_to_reference_h: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    nearest: bool = True,
) -> np.ndarray:
    if image_to_reference_h.shape != (3, 3):
        raise DataValidationError("Homography must be a 3x3 matrix")
    cv2 = _cv2()
    reference_to_image = np.linalg.inv(image_to_reference_h)
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cast(
        np.ndarray,
        cv2.warpPerspective(
            reference,
            reference_to_image,
            (image_width, image_height),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ),
    )


def rasterize_reference_shapes(
    shapes: Iterable[tuple[Any, int]],
    *,
    out_shape: tuple[int, int],
    transform: Any,
) -> np.ndarray:
    try:
        from rasterio.features import rasterize  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DataValidationError("Vector rasterization requires the registration extra") from exc
    return cast(
        np.ndarray,
        rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype="uint8",
        ),
    )
