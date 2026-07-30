"""Deterministic virtual-camera geometry shared by RGB, masks, and depth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio  # type: ignore[import-untyped]
from rasterio.enums import Resampling  # type: ignore[import-untyped]
from rasterio.transform import rowcol  # type: ignore[import-untyped]
from rasterio.warp import transform as transform_coordinates  # type: ignore[import-untyped]
from rasterio.windows import Window  # type: ignore[import-untyped]

from canyonbench.exceptions import DataValidationError
from canyonbench.trace.fidelity import relief_map
from canyonbench.trace.schemas import CameraSpec


@dataclass(frozen=True)
class PixelWindow:
    """Source-space quadrilateral and corresponding output transform."""

    source_points: np.ndarray
    output_points: np.ndarray
    homography: np.ndarray


def _camera_centre_px(
    dataset: rasterio.io.DatasetReader, camera: CameraSpec
) -> tuple[float, float]:
    source_crs = dataset.crs
    if source_crs is None:
        raise DataValidationError(f"Raster has no CRS: {dataset.name}")
    xs, ys = transform_coordinates("EPSG:4326", source_crs, [camera.longitude], [camera.latitude])
    row, column = rowcol(dataset.transform, xs[0], ys[0], op=float)
    return float(column), float(row)


def _metres_per_pixel(dataset: rasterio.io.DatasetReader) -> tuple[float, float]:
    if dataset.crs is None or not dataset.crs.is_projected:
        raise DataValidationError(
            f"Source raster must use a projected metric CRS before rendering: {dataset.name}"
        )
    return abs(float(dataset.transform.a)), abs(float(dataset.transform.e))


def camera_window(dataset: rasterio.io.DatasetReader, camera: CameraSpec) -> PixelWindow:
    """Compute the source quadrilateral for nadir or moderate-oblique rendering."""

    centre_x, centre_y = _camera_centre_px(dataset, camera)
    resolution_x, resolution_y = _metres_per_pixel(dataset)
    half_width = camera.ground_width_m / (2 * resolution_x)
    half_height = camera.ground_height_m / (2 * resolution_y)

    # A pinhole approximation: pitch shifts the far edge and foreshortens the near edge.
    pitch = np.deg2rad(camera.pitch_deg)
    shift = np.tan(pitch) * half_height
    near_scale = max(0.55, 1 - 0.45 * np.sin(pitch))
    far_scale = 1 + 0.45 * np.sin(pitch)
    points = np.array(
        [
            [centre_x - half_width * far_scale, centre_y - half_height + shift],
            [centre_x + half_width * far_scale, centre_y - half_height + shift],
            [centre_x + half_width * near_scale, centre_y + half_height + shift],
            [centre_x - half_width * near_scale, centre_y + half_height + shift],
        ],
        dtype=np.float32,
    )
    if camera.yaw_deg:
        angle = np.deg2rad(camera.yaw_deg)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        points = (points - np.array([centre_x, centre_y], dtype=np.float32)) @ rotation.T
        points += np.array([centre_x, centre_y], dtype=np.float32)
    output = np.array(
        [
            [0, 0],
            [camera.width_px - 1, 0],
            [camera.width_px - 1, camera.height_px - 1],
            [0, camera.height_px - 1],
        ],
        dtype=np.float32,
    )
    return PixelWindow(points, output, cv2.getPerspectiveTransform(points, output))


def warp_array(
    array: np.ndarray,
    window: PixelWindow,
    camera: CameraSpec,
    *,
    interpolation: int,
    border_value: int | float = 0,
) -> np.ndarray:
    return cv2.warpPerspective(
        array,
        window.homography,
        (camera.width_px, camera.height_px),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def apply_relief_displacement(
    rgb: np.ndarray,
    masks: dict[str, np.ndarray],
    continuous: dict[str, np.ndarray],
    depth: np.ndarray,
    camera: CameraSpec,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Optionally inject first-order DEM relief displacement into every raster.

    One sampling grid is shared by RGB, masks, and continuous layers, so the
    labels stay pixel-exact even though the geometry is no longer orthographic.
    """

    map_x, map_y = relief_map(camera, depth)
    warped_rgb = cv2.remap(
        rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    warped_masks = {
        name: (
            cv2.remap(
                mask,
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REPLICATE,
            )
            > 0
        ).astype(np.uint8)
        for name, mask in masks.items()
    }
    warped_continuous = {
        name: cv2.remap(
            values,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).astype(np.float32)
        for name, values in continuous.items()
    }
    return warped_rgb, warped_masks, warped_continuous


def render_aligned_rasters(
    imagery_path: Path,
    mask_paths: dict[str, Path],
    camera: CameraSpec,
    *,
    dem_path: Path | None = None,
    continuous_paths: dict[str, Path] | None = None,
    inject_relief: bool = False,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray | None,
    PixelWindow,
]:
    """Apply one exact transform to source-aligned RGB, masks, and optional DEM."""

    with rasterio.open(imagery_path) as imagery_source:
        global_window = camera_window(imagery_source, camera)
        imagery_bounds = imagery_source.bounds
        imagery_crs = imagery_source.crs
        imagery_transform = imagery_source.transform
        imagery_width = imagery_source.width
        imagery_height = imagery_source.height
        minimum_x = math.floor(float(np.min(global_window.source_points[:, 0]))) - 2
        maximum_x = math.ceil(float(np.max(global_window.source_points[:, 0]))) + 2
        minimum_y = math.floor(float(np.min(global_window.source_points[:, 1]))) - 2
        maximum_y = math.ceil(float(np.max(global_window.source_points[:, 1]))) + 2
        source_window = Window(
            col_off=minimum_x,
            row_off=minimum_y,
            width=max(2, maximum_x - minimum_x),
            height=max(2, maximum_y - minimum_y),
        )
        intermediate_width = max(
            2,
            min(math.ceil(source_window.width), camera.width_px * 2),
        )
        intermediate_height = max(
            2,
            min(math.ceil(source_window.height), camera.height_px * 2),
        )
        scale_x = intermediate_width / source_window.width
        scale_y = intermediate_height / source_window.height
        local_points = global_window.source_points.copy()
        local_points[:, 0] = (local_points[:, 0] - float(source_window.col_off)) * scale_x
        local_points[:, 1] = (local_points[:, 1] - float(source_window.row_off)) * scale_y
        window = PixelWindow(
            source_points=local_points,
            output_points=global_window.output_points,
            homography=cv2.getPerspectiveTransform(
                local_points.astype(np.float32),
                global_window.output_points,
            ),
        )
        indexes = list(range(1, min(3, imagery_source.count) + 1))
        rgb = imagery_source.read(
            indexes,
            window=source_window,
            out_shape=(len(indexes), intermediate_height, intermediate_width),
            out_dtype="float32",
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0,
        )
        if imagery_source.count == 1:
            rgb = np.repeat(rgb, 3, axis=0)
        rgb = np.moveaxis(rgb, 0, -1)

    continuous_paths = continuous_paths or {}
    aligned_paths = [
        *mask_paths.items(),
        *continuous_paths.items(),
        *(([("depth", dem_path)]) if dem_path else []),
    ]
    for name, path in aligned_paths:
        assert path is not None
        with rasterio.open(path) as dataset:
            if (
                dataset.crs != imagery_crs
                or dataset.transform != imagery_transform
                or dataset.bounds != imagery_bounds
                or dataset.width != imagery_width
                or dataset.height != imagery_height
            ):
                raise DataValidationError(
                    f"{name} raster is not pixel-aligned with imagery; pre-align sources before "
                    f"rendering ({path})"
                )

    warped_rgb = warp_array(rgb, window, camera, interpolation=cv2.INTER_CUBIC)
    if warped_rgb.max(initial=0) > 255:
        warped_rgb = warped_rgb / max(float(warped_rgb.max()), 1) * 255
    warped_rgb = np.clip(warped_rgb, 0, 255).astype(np.uint8)

    masks: dict[str, np.ndarray] = {}
    for feature, path in mask_paths.items():
        with rasterio.open(path) as dataset:
            array = dataset.read(
                [1],
                window=source_window,
                out_shape=(1, intermediate_height, intermediate_width),
                out_dtype="float32",
                resampling=Resampling.nearest,
                boundless=True,
                fill_value=0,
            )[0]
        mask = warp_array(array, window, camera, interpolation=cv2.INTER_NEAREST)
        masks[feature] = (mask > 0).astype(np.uint8)

    continuous: dict[str, np.ndarray] = {}
    for name, path in continuous_paths.items():
        with rasterio.open(path) as dataset:
            array = dataset.read(
                [1],
                window=source_window,
                out_shape=(1, intermediate_height, intermediate_width),
                out_dtype="float32",
                resampling=Resampling.bilinear,
                boundless=True,
                fill_value=0,
            )[0]
        values = warp_array(
            array,
            window,
            camera,
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        maximum = float(np.nanmax(values)) if np.isfinite(values).any() else 0
        if maximum > 1:
            values /= 255 if maximum <= 255 else maximum
        continuous[name] = np.clip(values, 0, 1)

    depth: np.ndarray | None = None
    if dem_path is not None:
        with rasterio.open(dem_path) as dataset:
            dem = dataset.read(
                [1],
                window=source_window,
                out_shape=(1, intermediate_height, intermediate_width),
                out_dtype="float32",
                resampling=Resampling.bilinear,
                boundless=True,
                fill_value=np.nan,
            )[0]
        depth = warp_array(
            dem,
            window,
            camera,
            interpolation=cv2.INTER_LINEAR,
            border_value=np.nan,
        ).astype(np.float32)
    if inject_relief:
        if depth is None:
            raise DataValidationError(
                "relief-displacement injection requires a DEM for every rendered site"
            )
        warped_rgb, masks, continuous = apply_relief_displacement(
            warped_rgb, masks, continuous, depth, camera
        )
    return warped_rgb, masks, continuous, depth, global_window
