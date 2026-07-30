"""Source ingestion helpers for immutable rasters and vector feature layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio  # type: ignore[import-untyped]
from rasterio.features import rasterize  # type: ignore[import-untyped]
from rasterio.warp import Resampling, reproject  # type: ignore[import-untyped]

from canyonbench.exceptions import DataValidationError


def align_raster_to_reference(
    source: Path,
    reference: Path,
    output: Path,
    *,
    categorical: bool = True,
) -> Path:
    """Reproject a raster onto the exact reference pixel grid."""

    try:
        with rasterio.open(reference) as target, rasterio.open(source) as candidate:
            profile = target.profile.copy()
            profile.update(
                count=candidate.count, dtype=candidate.dtypes[0], nodata=candidate.nodata
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output, "w", **profile) as destination:
                for band in range(1, candidate.count + 1):
                    reproject(
                        source=rasterio.band(candidate, band),
                        destination=rasterio.band(destination, band),
                        src_transform=candidate.transform,
                        src_crs=candidate.crs,
                        src_nodata=candidate.nodata,
                        dst_transform=target.transform,
                        dst_crs=target.crs,
                        dst_nodata=candidate.nodata,
                        resampling=Resampling.nearest if categorical else Resampling.bilinear,
                    )
    except (OSError, rasterio.errors.RasterioError) as exc:
        raise DataValidationError(f"Could not align {source} to {reference}: {exc}") from exc
    return output


def rasterize_geojson(
    geojson: dict[str, Any],
    reference: Path,
    output: Path,
    *,
    all_touched: bool = False,
) -> Path:
    """Rasterize GeoJSON features onto the exact imagery grid without GIS bindings."""

    raw_features = geojson.get("features")
    if not isinstance(raw_features, list):
        raise DataValidationError("GeoJSON must contain a features list")
    shapes: list[tuple[dict[str, Any], int]] = []
    for feature in raw_features:
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
        if geometry:
            shapes.append((geometry, 1))
    try:
        with rasterio.open(reference) as source:
            mask = rasterize(
                shapes,
                out_shape=(source.height, source.width),
                transform=source.transform,
                fill=0,
                all_touched=all_touched,
                dtype="uint8",
            )
            profile = source.profile.copy()
            profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as destination:
            destination.write(mask, 1)
    except (OSError, rasterio.errors.RasterioError, ValueError) as exc:
        raise DataValidationError(f"Could not rasterize feature layer to {output}: {exc}") from exc
    return output


def assert_binary_mask(path: Path) -> None:
    with rasterio.open(path) as dataset:
        values = np.unique(dataset.read([1])[0])
    if not set(values.tolist()).issubset({0, 1, 255}):
        raise DataValidationError(f"Mask is not binary: {path}; values={values[:20].tolist()}")
