"""Homography estimation with held-out, metre-scale validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canyonbench.exceptions import DataValidationError


@dataclass(frozen=True)
class RegistrationResult:
    image: str
    n_points: int
    n_fit_points: int
    n_holdout_points: int
    holdout_rmse_m: float
    threshold_m: float
    reliable: bool
    homography: list[list[float]] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise DataValidationError(
            "Registration requires the 'registration' extra: "
            "pip install 'canyonbench[registration]'"
        ) from exc
    return cv2


def read_control_points(path: str | Path) -> pd.DataFrame:
    """Read QGIS ``.points`` or a canonical CSV control-point file.

    Canonical fields are ``image_x,image_y,map_x,map_y,role`` where role is
    ``fit`` or ``holdout``. QGIS fields ``sourceX/sourceY/mapX/mapY/enable``
    are accepted, but callers must identify two holdouts separately if role is
    absent; the last two enabled points become holdouts deterministically.
    """

    source = Path(path)
    try:
        frame = pd.read_csv(source, comment="#")
    except (OSError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Could not parse control points {source}: {exc}") from exc
    aliases = {
        "sourcex": "image_x",
        "sourcey": "image_y",
        "mapx": "map_x",
        "mapy": "map_y",
        "pixelx": "image_x",
        "pixely": "image_y",
    }
    renames = {
        column: aliases.get(str(column).lower().replace("_", ""), str(column).lower())
        for column in frame.columns
    }
    frame = frame.rename(columns=renames)
    required = {"image_x", "image_y", "map_x", "map_y"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"Control-point file is missing fields: {missing}")
    if "enable" in frame:
        frame = frame.loc[frame["enable"].astype(str).str.lower().isin({"1", "true"})]
    if "role" not in frame:
        if len(frame) < 6:
            raise DataValidationError("At least six enabled control points are required")
        frame["role"] = "fit"
        frame.loc[frame.index[-2:], "role"] = "holdout"
    invalid_roles = sorted(set(frame["role"]) - {"fit", "holdout"})
    if invalid_roles:
        raise DataValidationError(f"Unknown control-point roles: {invalid_roles}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame[[*list(required), "role"]].reset_index(drop=True)


def validate_point_distribution(
    points: pd.DataFrame, image_width: int, image_height: int
) -> list[str]:
    warnings: list[str] = []
    if len(points) < 6:
        warnings.append("fewer_than_six_points")
    x = points["image_x"].to_numpy()
    y = points["image_y"].to_numpy()
    quadrants = {
        (int(value_x >= image_width / 2), int(value_y >= image_height / 2))
        for value_x, value_y in zip(x, y, strict=True)
    }
    if len(quadrants) < 4:
        warnings.append("missing_quadrant")
    centre_distance = np.hypot(x - image_width / 2, y - image_height / 2)
    if not np.any(centre_distance <= 0.2 * min(image_width, image_height)):
        warnings.append("missing_centre_point")
    centered = np.column_stack((x - x.mean(), y - y.mean()))
    if len(points) >= 2 and np.linalg.matrix_rank(centered) < 2:
        warnings.append("collinear_points")
    return warnings


def registration_threshold_m(ground_width_m: float) -> float:
    """One quarter of a cell width for a 4x4 ground grid."""

    if ground_width_m <= 0:
        raise DataValidationError("ground_width_m must be positive")
    return ground_width_m / 16.0


def fit_homography(
    image: str,
    points: pd.DataFrame,
    *,
    threshold_m: float,
    ransac_reprojection_threshold: float = 5.0,
    seed: int = 2026,
) -> RegistrationResult:
    if len(points) < 6:
        raise DataValidationError("At least six total control points are required")
    fit = points.loc[points["role"] == "fit"]
    holdout = points.loc[points["role"] == "holdout"]
    if len(fit) < 4 or len(holdout) < 2:
        raise DataValidationError("Registration requires at least four fit and two holdout points")
    cv2 = _cv2()
    cv2.setRNGSeed(seed)
    source = fit[["image_x", "image_y"]].to_numpy(np.float64)
    target = fit[["map_x", "map_y"]].to_numpy(np.float64)
    homography, _ = cv2.findHomography(source, target, cv2.RANSAC, ransac_reprojection_threshold)
    if homography is None:
        return RegistrationResult(
            image=image,
            n_points=len(points),
            n_fit_points=len(fit),
            n_holdout_points=len(holdout),
            holdout_rmse_m=float("inf"),
            threshold_m=threshold_m,
            reliable=False,
            homography=None,
        )
    test_source = holdout[["image_x", "image_y"]].to_numpy(np.float64).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(test_source, homography).reshape(-1, 2)
    expected = holdout[["map_x", "map_y"]].to_numpy(np.float64)
    rmse = float(np.sqrt(np.mean(np.sum((projected - expected) ** 2, axis=1))))
    return RegistrationResult(
        image=image,
        n_points=len(points),
        n_fit_points=len(fit),
        n_holdout_points=len(holdout),
        holdout_rmse_m=rmse,
        threshold_m=threshold_m,
        reliable=bool(np.isfinite(rmse) and rmse <= threshold_m),
        homography=homography.tolist(),
    )
