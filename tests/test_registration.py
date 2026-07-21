from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from canyonbench.exceptions import DataValidationError
from canyonbench.registration.geometry import estimate_ground_geometry
from canyonbench.registration.homography import (
    fit_homography,
    read_control_points,
    registration_threshold_m,
    validate_point_distribution,
)


def synthetic_points() -> pd.DataFrame:
    image = np.array(
        [[0, 0], [100, 0], [0, 100], [100, 100], [50, 50], [25, 75], [75, 25], [80, 80]],
        dtype=float,
    )
    mapped = image * 2 + np.array([500_000, 4_100_000])
    return pd.DataFrame(
        {
            "image_x": image[:, 0],
            "image_y": image[:, 1],
            "map_x": mapped[:, 0],
            "map_y": mapped[:, 1],
            "role": ["fit"] * 6 + ["holdout"] * 2,
        }
    )


def test_registration_threshold() -> None:
    assert registration_threshold_m(1600) == 100
    with pytest.raises(DataValidationError):
        registration_threshold_m(0)


def test_geometry_exposes_nadir_assumption_and_threshold() -> None:
    geometry = estimate_ground_geometry(
        altitude_agl_m=1000,
        horizontal_fov_deg=90,
        vertical_fov_deg=60,
        image_width_px=1000,
        image_height_px=500,
    )
    assert geometry.footprint_width_m == pytest.approx(2000)
    assert geometry.gsd_x_m_per_px == pytest.approx(2)
    assert geometry.reliability_threshold_m == pytest.approx(125)
    assert geometry.assumption == "nadir_planar_ground"


def test_read_control_points_and_distribution(tmp_path: Path) -> None:
    path = tmp_path / "points.csv"
    synthetic_points().to_csv(path, index=False)
    points = read_control_points(path)
    assert len(points) == 8
    assert validate_point_distribution(points, 100, 100) == []


def test_fit_homography_has_near_zero_holdout_error() -> None:
    pytest.importorskip("cv2")
    result = fit_homography("img_006806.jpg", synthetic_points(), threshold_m=1)
    assert result.reliable is True
    assert result.holdout_rmse_m < 1e-3
    assert result.n_holdout_points == 2


def test_read_control_points_requires_six(tmp_path: Path) -> None:
    path = tmp_path / "short.csv"
    synthetic_points().head(4).drop(columns="role").to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="six"):
        read_control_points(path)
