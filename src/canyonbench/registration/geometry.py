"""Transparent nadir-view footprint and effective-GSD estimates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from canyonbench.exceptions import DataValidationError


@dataclass(frozen=True)
class GroundGeometry:
    altitude_agl_m: float
    horizontal_fov_deg: float
    vertical_fov_deg: float
    image_width_px: int
    image_height_px: int
    footprint_width_m: float
    footprint_height_m: float
    gsd_x_m_per_px: float
    gsd_y_m_per_px: float
    reliability_threshold_m: float
    assumption: str = "nadir_planar_ground"

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def estimate_ground_geometry(
    *,
    altitude_agl_m: float,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
    image_width_px: int,
    image_height_px: int,
) -> GroundGeometry:
    """Estimate footprint/GSD; use only when camera calibration supports the inputs."""

    if altitude_agl_m <= 0:
        raise DataValidationError("altitude_agl_m must be positive")
    if not 0 < horizontal_fov_deg < 180 or not 0 < vertical_fov_deg < 180:
        raise DataValidationError("camera field of view must lie between 0 and 180 degrees")
    if image_width_px <= 0 or image_height_px <= 0:
        raise DataValidationError("image dimensions must be positive")
    width = 2 * altitude_agl_m * math.tan(math.radians(horizontal_fov_deg) / 2)
    height = 2 * altitude_agl_m * math.tan(math.radians(vertical_fov_deg) / 2)
    return GroundGeometry(
        altitude_agl_m=altitude_agl_m,
        horizontal_fov_deg=horizontal_fov_deg,
        vertical_fov_deg=vertical_fov_deg,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        footprint_width_m=width,
        footprint_height_m=height,
        gsd_x_m_per_px=width / image_width_px,
        gsd_y_m_per_px=height / image_height_px,
        reliability_threshold_m=width / 16,
    )
