"""Sim-to-real fidelity accounting for the orthophoto virtual camera.

A real camera at float altitude records relief displacement that a synthesized
nadir view over an already-orthorectified photo does not. In vertical aerial
geometry a point standing dh above the reference plane is imaged radially
outward from nadir by

    d = r * dh / H

with r the radial distance from the principal point and H the flying height.
The displacement corrupts no label, because image and masks pass through the
same transform, so it is a fidelity gap rather than an alignment error. This
module quantifies it per view and aggregates it across the dataset so the gap is
reported instead of concealed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from canyonbench.io import read_json, write_json
from canyonbench.trace.schemas import CameraSpec, ReliefDisplacement, ViewManifest


def relief_displacement_px(*, radial_px: float, relief_m: float, height_agl_m: float) -> float:
    """Return d = r * dh / H in pixels for one radial distance."""

    if height_agl_m <= 0:
        raise ValueError("flying height above ground must be positive")
    return float(abs(radial_px) * relief_m / height_agl_m)


def terrain_relief_m(depth: np.ndarray | None) -> float:
    """Robust within-view relief from the projected DEM, ignoring outliers."""

    if depth is None:
        return 0.0
    values = np.asarray(depth, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    low, high = np.percentile(finite, [2.0, 98.0])
    return float(max(0.0, high - low))


def view_relief_displacement(camera: CameraSpec, depth: np.ndarray | None) -> ReliefDisplacement:
    """Quantify the unmodelled relief displacement of one rendered view."""

    relief = terrain_relief_m(depth)
    half_width = (camera.width_px - 1) / 2
    half_height = (camera.height_px - 1) / 2
    corner_radius = math.hypot(half_width, half_height)
    ratio = relief / camera.altitude_agl_m
    return ReliefDisplacement(
        relief_m=relief,
        height_agl_m=camera.altitude_agl_m,
        displacement_ratio=float(ratio),
        edge_displacement_px=relief_displacement_px(
            radial_px=half_width,
            relief_m=relief,
            height_agl_m=camera.altitude_agl_m,
        ),
        corner_displacement_px=relief_displacement_px(
            radial_px=corner_radius,
            relief_m=relief,
            height_agl_m=camera.altitude_agl_m,
        ),
        corner_displacement_m=float(ratio * corner_radius * camera.gsd_m_per_px),
        dem_available=depth is not None,
        injected=False,
    )


def relief_map(
    camera: CameraSpec,
    depth: np.ndarray,
    *,
    reference_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the sampling grid that adds first-order relief displacement.

    An orthophoto places a point at its planimetric radius ``r``; a real frame
    images it at ``r * (1 + dh / H)``. The output pixel at radius ``r_out``
    therefore samples the source at ``r_out / (1 + dh / H)``. Applying the same
    grid to RGB and to every mask keeps the labels exact.
    """

    values = np.asarray(depth, dtype=float)
    if values.shape[:2] != (camera.height_px, camera.width_px):
        raise ValueError("depth raster must match the rendered view shape")
    reference = (
        float(np.nanmedian(values[np.isfinite(values)]))
        if reference_m is None and np.isfinite(values).any()
        else float(reference_m or 0.0)
    )
    elevation = np.where(np.isfinite(values), values, reference)
    scale = 1.0 + (elevation - reference) / camera.altitude_agl_m
    scale = np.clip(scale, 0.5, 1.5)
    centre_x = (camera.width_px - 1) / 2
    centre_y = (camera.height_px - 1) / 2
    grid_x, grid_y = np.meshgrid(
        np.arange(camera.width_px, dtype=np.float32),
        np.arange(camera.height_px, dtype=np.float32),
    )
    map_x = (centre_x + (grid_x - centre_x) / scale).astype(np.float32)
    map_y = (centre_y + (grid_y - centre_y) / scale).astype(np.float32)
    return map_x, map_y


def dataset_relief_report(dataset_dir: Path, output: Path | None = None) -> dict[str, Any]:
    """Aggregate the reported sim-to-real relief gap over every clean view."""

    index = [row for row in read_json(dataset_dir / "index.json") if row["variant"] == "clean"]
    records: list[dict[str, Any]] = []
    for row in index:
        manifest_path = dataset_dir / str(row["manifest_path"])
        if not manifest_path.exists():
            continue
        manifest = ViewManifest.model_validate(read_json(manifest_path))
        if manifest.relief_displacement is None:
            continue
        records.append(
            {
                "site_id": manifest.site_id,
                "view_id": manifest.view_id,
                "geometry": manifest.geometry,
                "altitude_agl_m": manifest.camera.altitude_agl_m,
                **manifest.relief_displacement.model_dump(mode="json"),
            }
        )

    def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
        if not rows:
            return {}
        edge = np.asarray([row["edge_displacement_px"] for row in rows], dtype=float)
        corner = np.asarray([row["corner_displacement_px"] for row in rows], dtype=float)
        relief = np.asarray([row["relief_m"] for row in rows], dtype=float)
        return {
            "views": float(len(rows)),
            "median_relief_m": float(np.median(relief)),
            "maximum_relief_m": float(np.max(relief)),
            "median_edge_displacement_px": float(np.median(edge)),
            "maximum_edge_displacement_px": float(np.max(edge)),
            "median_corner_displacement_px": float(np.median(corner)),
            "maximum_corner_displacement_px": float(np.max(corner)),
        }

    report: dict[str, Any] = {
        "schema_version": "4.0.0",
        "formula": "d = r * relief / height_agl (first-order vertical aerial geometry)",
        "interpretation": (
            "Relief displacement is absent from the synthesized orthophoto views and "
            "corrupts no label, because image and masks share one transform. It is a "
            "reported sim-to-real fidelity gap, not a registration error."
        ),
        "views_with_dem": sum(bool(row["dem_available"]) for row in records),
        "views_with_injection": sum(bool(row["injected"]) for row in records),
        "overall": summarize(records),
        "by_altitude_agl_m": {
            str(int(altitude)): summarize(
                [row for row in records if row["altitude_agl_m"] == altitude]
            )
            for altitude in sorted({row["altitude_agl_m"] for row in records})
        },
        "by_geometry": {
            geometry: summarize([row for row in records if row["geometry"] == geometry])
            for geometry in sorted({str(row["geometry"]) for row in records})
        },
        "views": records,
    }
    if output is not None:
        write_json(output, report)
    return report
