"""Reproducible discovery and materialization of CanyonBench-Trace source sites."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import cv2
import httpx
import numpy as np
import rasterio  # type: ignore[import-untyped]
import yaml
from rasterio.enums import Resampling  # type: ignore[import-untyped]
from rasterio.features import rasterize  # type: ignore[import-untyped]
from rasterio.io import MemoryFile  # type: ignore[import-untyped]
from rasterio.transform import from_origin, xy  # type: ignore[import-untyped]
from rasterio.vrt import WarpedVRT  # type: ignore[import-untyped]
from rasterio.warp import (  # type: ignore[import-untyped]
    reproject,
    transform,
    transform_bounds,
    transform_geom,
)
from rasterio.windows import Window  # type: ignore[import-untyped]
from shapely import box  # type: ignore[import-untyped]
from shapely.geometry import Polygon, shape  # type: ignore[import-untyped]
from shapely.geometry.base import BaseGeometry  # type: ignore[import-untyped]
from shapely.ops import unary_union  # type: ignore[import-untyped]
from shapely.strtree import STRtree  # type: ignore[import-untyped]

from canyonbench.exceptions import DataValidationError
from canyonbench.io import atomic_write_text, read_json, sha256_file, write_json
from canyonbench.trace.schemas import (
    CandidateSeed,
    SiteSpec,
    SourceAcquisitionConfig,
    SourceManifest,
    SourceRecord,
)

NAIP_STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PLANETARY_COMPUTER_SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
NAIP_COLLECTION = "naip"
NAIP_TERMS = (
    "https://www.usgs.gov/faqs/what-are-terms-uselicensing-map-services-and-data-national-map"
)
NAIP_SERVICE = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer"
NHD_SERVICE = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer"
NHD_TERMS = "https://www.usgs.gov/3d-hydrography-program/access-3dhp-data-products"
TIGER_ROAD_SERVICE = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Transportation/MapServer"
)
TIGER_HYDRO_SERVICE = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Hydro/MapServer"
)
TIGER_TERMS = "https://www.census.gov/data/developers/about/terms-of-service.html"
OSM_OVERPASS = "https://overpass-api.de/api/interpreter"
OSM_TERMS = "https://www.openstreetmap.org/copyright"
CDL_SERVICE = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"
CDL_RELEASE_URL = "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/"
CDL_TERMS = "https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php"
NLCD_TERMS = (
    "https://www.usgs.gov/centers/eros/science/usgs-eros-archive-land-cover-"
    "annual-nlcd-collection-1-land-cover"
)
NLCD_IMPERVIOUS_TERMS = (
    "https://www.usgs.gov/centers/eros/science/usgs-eros-archive-land-cover-"
    "annual-nlcd-collection-1-impervious-descriptor"
)
IO_LULC_COLLECTION = "io-lulc-9-class"
IO_LULC_TERMS = "https://creativecommons.org/licenses/by/4.0/"
THREE_DEP_SERVICE = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"
)
THREE_DEP_TERMS = "https://www.usgs.gov/3d-elevation-program"
ACCESS_DATE = date.today().isoformat()

WATER_LAYERS = (9, 12)
ROAD_LAYERS = (2, 3)
CULTIVATED_CDL_CODES = frozenset(
    [*range(1, 7), *range(10, 15), *range(21, 61), *range(66, 81), *range(200, 255)]
)


class _Grid:
    """Exact local site grid shared by every source artifact."""

    def __init__(
        self,
        longitude: float,
        latitude: float,
        resolution_m: float,
        half_extent_m: float,
    ) -> None:
        zone = min(60, max(1, int((longitude + 180) // 6) + 1))
        self.epsg = (32600 if latitude >= 0 else 32700) + zone
        xs, ys = transform("EPSG:4326", f"EPSG:{self.epsg}", [longitude], [latitude])
        size = math.ceil((2 * half_extent_m) / resolution_m)
        if size % 2:
            size += 1
        actual_extent = size * resolution_m / 2
        self.width = size
        self.height = size
        self.resolution_m = resolution_m
        self.left = xs[0] - actual_extent
        self.bottom = ys[0] - actual_extent
        self.right = xs[0] + actual_extent
        self.top = ys[0] + actual_extent
        self.transform = from_origin(self.left, self.top, resolution_m, resolution_m)
        self.crs = f"EPSG:{self.epsg}"
        self.wgs84_bounds = transform_bounds(
            self.crs,
            "EPSG:4326",
            self.left,
            self.bottom,
            self.right,
            self.top,
            densify_pts=21,
        )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.left, self.bottom, self.right, self.top)

    def profile(self, *, dtype: str, count: int = 1, nodata: int | float = 0) -> dict[str, Any]:
        return {
            "driver": "GTiff",
            "width": self.width,
            "height": self.height,
            "count": count,
            "dtype": dtype,
            "crs": self.crs,
            "transform": self.transform,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "deflate",
            "predictor": 2 if dtype != "float32" else 3,
            "bigtiff": "IF_SAFER",
            "nodata": nodata,
        }


def _http_client() -> httpx.Client:
    # Public imagery services frequently return a gateway timeout for a single
    # export tile.  Keep a failed candidate bounded: acquisition is resumable,
    # whereas five 300-second waits can stall the whole serial login-node run
    # for more than twenty minutes before the next candidate is tried.
    return httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(60, connect=20),
        headers={"User-Agent": "CanyonBench-Trace/4.0 source-curation"},
    )


def _retry_get(
    client: httpx.Client,
    url: str,
    *,
    params: Any = None,
    attempts: int = 3,
) -> httpx.Response:
    error: BaseException | None = None
    for attempt in range(attempts):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 8))
    raise DataValidationError(f"Source request failed after {attempts} attempts: {url}: {error}")


def _retry_post(
    client: httpx.Client,
    url: str,
    *,
    data: Any = None,
    json_value: Any = None,
    attempts: int = 3,
) -> httpx.Response:
    error: BaseException | None = None
    for attempt in range(attempts):
        try:
            response = client.post(url, data=data, json=json_value)
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 8))
    raise DataValidationError(f"Source request failed after {attempts} attempts: {url}: {error}")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _arcgis_features(
    client: httpx.Client,
    service: str,
    layer: int,
    bounds: tuple[float, float, float, float],
    *,
    out_fields: str = "*",
) -> list[dict[str, Any]]:
    query = f"{service}/{layer}/query"
    common = {
        "f": "json",
        "geometry": ",".join(f"{value:.10f}" for value in bounds),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "where": "1=1",
        "returnGeometry": "true",
        "outSR": "4326",
        "outFields": out_fields,
    }
    id_payload = _retry_get(
        client,
        query,
        params={
            **common,
            "returnIdsOnly": "true",
            "returnGeometry": "false",
        },
    ).json()
    if "error" in id_payload:
        raise DataValidationError(f"ArcGIS ID query failed: {id_payload['error']}")
    object_ids = sorted(int(value) for value in id_payload.get("objectIds") or [])
    features: list[dict[str, Any]] = []
    # ID-chunked requests are substantially more reliable than offset paging
    # for geometry-heavy nationwide layers. The initial spatial query fixes the
    # exact object set; subsequent requests cannot skip or duplicate records.
    for offset in range(0, len(object_ids), 250):
        chunk = object_ids[offset : offset + 250]
        payload = _retry_get(
            client,
            query,
            params={
                "f": "geojson",
                "objectIds": ",".join(str(value) for value in chunk),
                "returnGeometry": "true",
                "outSR": "4326",
                "outFields": out_fields,
            },
        ).json()
        if "error" in payload:
            raise DataValidationError(f"ArcGIS feature query failed: {payload['error']}")
        page = payload.get("features", [])
        if not isinstance(page, list):
            raise DataValidationError("ArcGIS GeoJSON response has no feature list")
        features.extend(cast(list[dict[str, Any]], page))
    return features


def _cached_arcgis_features(
    client: httpx.Client,
    service: str,
    layer: int,
    bounds: tuple[float, float, float, float],
    cache_path: Path,
) -> list[dict[str, Any]]:
    """Freeze a live ArcGIS query so discovery is cheap and reproducible on rerun."""

    if cache_path.is_file():
        payload = read_json(cache_path)
        if not isinstance(payload, list):
            raise DataValidationError(f"ArcGIS cache is not a feature list: {cache_path}")
        return cast(list[dict[str, Any]], payload)
    features = _arcgis_features(client, service, layer, bounds)
    write_json(cache_path, features)
    return features


def _feature_id(feature: dict[str, Any], *, prefix: str) -> str:
    properties = feature.get("properties") or {}
    for key in (
        "PERMANENT_IDENTIFIER",
        "permanent_identifier",
        "OID",
        "OBJECTID",
        "GLOBALID",
        "@id",
    ):
        value = properties.get(key)
        if value not in (None, ""):
            return f"{prefix}:{value}"
    geometry = json.dumps(
        feature.get("geometry"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}:sha256:{hashlib.sha256(geometry).hexdigest()[:24]}"


def _nlcd_url(product: str) -> tuple[str, str]:
    workspace = f"mrlc_{product}_conus_year_data"
    base = f"https://dmsdata.cr.usgs.gov/geoserver/{workspace}/wcs"
    coverage = f"{workspace}:{product}_conus_year_data"
    return base, coverage


def _fetch_nlcd(
    client: httpx.Client,
    product: str,
    bounds_wgs84: tuple[float, float, float, float],
    year: int,
    destination: Path,
) -> Path:
    if destination.is_file():
        return destination
    bounds = transform_bounds("EPSG:4326", "EPSG:5070", *bounds_wgs84, densify_pts=21)
    width = max(1, math.ceil((bounds[2] - bounds[0]) / 30))
    height = max(1, math.ceil((bounds[3] - bounds[1]) / 30))
    base, coverage = _nlcd_url(product)
    response = _retry_get(
        client,
        base,
        # The public WCS intermittently returns 500/timeout responses for
        # otherwise valid windows. This is bounded (1+2+4+8+8 seconds) and
        # avoids discarding a site solely because of a transient gateway fault.
        attempts=6,
        params={
            "service": "WCS",
            "version": "1.0.0",
            "request": "GetCoverage",
            "coverage": coverage,
            "format": "GeoTIFF",
            "bbox": ",".join(str(value) for value in bounds),
            "crs": "EPSG:5070",
            "response_crs": "EPSG:5070",
            "width": width,
            "height": height,
            "time": f"{year}-01-01T00:00:00.000Z",
        },
    )
    if "tiff" not in response.headers.get("content-type", "").lower():
        raise DataValidationError(
            f"Annual NLCD returned {response.headers.get('content-type')}: {response.text[:500]}"
        )
    _atomic_bytes(destination, response.content)
    return destination


def _fetch_cdl(
    client: httpx.Client,
    bounds_wgs84: tuple[float, float, float, float],
    year: int,
    destination: Path,
) -> str | Path:
    if destination.is_file():
        return destination
    archive = _cached_cdl_archive(year)
    if archive is not None:
        try:
            with zipfile.ZipFile(archive) as bundle:
                members = sorted(
                    name for name in bundle.namelist() if name.lower().endswith((".tif", ".tiff"))
                )
        except zipfile.BadZipFile as exc:
            raise DataValidationError(f"Invalid cached national CDL archive: {archive}") from exc
        if len(members) != 1:
            raise DataValidationError(
                f"Cached national CDL archive must contain one GeoTIFF, found {members}: {archive}"
            )
        # GDAL windows this local official archive directly, avoiding one
        # unreliable CropScape API request for every candidate site.
        # Keep this as a string. pathlib collapses the required double slash
        # between /vsizip and an absolute archive path.
        return f"/vsizip//{archive.resolve().as_posix().lstrip('/')}/{members[0]}"
    bounds = transform_bounds("EPSG:4326", "EPSG:5070", *bounds_wgs84, densify_pts=21)
    response = _retry_get(
        client,
        CDL_SERVICE,
        params={"year": year, "bbox": ",".join(str(value) for value in bounds)},
    )
    marker_start, marker_end = "<returnURL>", "</returnURL>"
    if marker_start not in response.text:
        raise DataValidationError(
            f"USDA CDL service returned no download URL: {response.text[:500]}"
        )
    url = response.text.split(marker_start, 1)[1].split(marker_end, 1)[0]
    image = _retry_get(client, url)
    if not image.content.startswith((b"II", b"MM")):
        raise DataValidationError(f"USDA CDL download was not a TIFF: {url}")
    _atomic_bytes(destination, image.content)
    return destination


def _cached_cdl_archive(year: int) -> Path | None:
    """Return the configured official CDL archive if it is locally available."""

    cache_dir = os.environ.get("CANYONBENCH_CDL_CACHE_DIR")
    if not cache_dir:
        return None
    archive = Path(cache_dir) / f"{year}_30m_cdls.zip"
    return archive if archive.is_file() else None


def _to_projected_geometry(geometry: dict[str, Any], crs: str) -> BaseGeometry:
    return shape(transform_geom("EPSG:4326", crs, geometry, antimeridian_cutting=False))


def _discovery_field_points(
    path: Path,
    *,
    half_extent_m: float,
    generator: np.random.Generator,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    with rasterio.open(path) as dataset:
        values = dataset.read(1)
        cultivated = values == 82
        positive_core = cv2.erode(
            cultivated.astype(np.uint8),
            np.ones((5, 5), np.uint8),
        ).astype(bool)
        radius = math.ceil(half_extent_m / max(abs(dataset.transform.a), 1))
        # Integral-image square sums make full-footprint negative screening O(n).
        integral = cv2.integral(cultivated.astype(np.uint8), sdepth=cv2.CV_64F)
        rows = np.arange(radius, max(radius, dataset.height - radius), dtype=int)
        columns = np.arange(radius, max(radius, dataset.width - radius), dtype=int)
        negative: list[tuple[float, float]] = []
        if len(rows) and len(columns):
            candidates = np.column_stack(
                (
                    generator.choice(rows, size=min(15000, len(rows) * len(columns))),
                    generator.choice(columns, size=min(15000, len(rows) * len(columns))),
                )
            )
            for row, column in candidates:
                y0, y1 = row - radius, row + radius + 1
                x0, x1 = column - radius, column + radius + 1
                total = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
                if total == 0:
                    east, north = xy(dataset.transform, int(row), int(column))
                    negative.append((float(east), float(north)))
        positive_indices = np.column_stack(np.where(positive_core))
        if len(positive_indices) > 15000:
            positive_indices = positive_indices[
                generator.choice(len(positive_indices), size=15000, replace=False)
            ]
        positive = [
            cast(tuple[float, float], xy(dataset.transform, int(row), int(column)))
            for row, column in positive_indices
        ]
        return positive, negative


def _convert_points(
    points: Iterable[tuple[float, float]],
    source_crs: str,
) -> list[tuple[float, float]]:
    materialized = list(points)
    if not materialized:
        return []
    xs, ys = zip(*materialized, strict=True)
    longitudes, latitudes = transform(source_crs, "EPSG:4326", list(xs), list(ys))
    return list(zip(longitudes, latitudes, strict=True))


def _spaced(
    candidates: Sequence[tuple[float, float, list[str], str, str]],
    count: int,
    *,
    minimum_m: float,
    generator: np.random.Generator,
) -> list[tuple[float, float, list[str], str, str]]:
    if not candidates:
        return []
    order = generator.permutation(len(candidates))
    selected: list[tuple[float, float, list[str], str, str]] = []
    for index in order:
        candidate = candidates[int(index)]
        longitude, latitude = candidate[:2]
        if all(
            _haversine_m(longitude, latitude, existing[0], existing[1]) >= minimum_m
            for existing in selected
        ):
            selected.append(candidate)
            if len(selected) == count:
                break
    return selected


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6_371_008.8
    first_latitude, second_latitude = math.radians(lat1), math.radians(lat2)
    latitude_delta = math.radians(lat2 - lat1)
    longitude_delta = math.radians(lon2 - lon1)
    value = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(first_latitude) * math.cos(second_latitude) * math.sin(longitude_delta / 2) ** 2
    )
    return 2 * radius * math.asin(min(1, math.sqrt(value)))


def _vector_candidates(
    features: Sequence[dict[str, Any]],
    region_bounds: tuple[float, float, float, float],
    *,
    crs: str,
    half_extent_m: float,
    prefix: str,
) -> tuple[
    list[tuple[float, float, list[str], str]],
    list[tuple[float, float, list[str], str]],
]:
    projected: list[tuple[BaseGeometry, str]] = []
    for feature in features:
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            continue
        candidate = _to_projected_geometry(geometry, crs)
        if candidate.is_empty:
            continue
        projected.append((candidate, _feature_id(feature, prefix=prefix)))
    positives: list[tuple[float, float, list[str], str]] = []
    for geometry, identifier in projected:
        point = (
            geometry.interpolate(0.5, normalized=True)
            if geometry.geom_type in {"LineString", "MultiLineString"}
            else geometry.representative_point()
        )
        longitudes, latitudes = transform(crs, "EPSG:4326", [point.x], [point.y])
        positives.append((longitudes[0], latitudes[0], [identifier], prefix))

    geometries = [geometry for geometry, _ in projected]
    tree = STRtree(geometries) if geometries else None
    projected_bounds = transform_bounds("EPSG:4326", crs, *region_bounds, densify_pts=21)
    step = max(half_extent_m * 0.4, 4000)
    negatives: list[tuple[float, float, list[str], str]] = []
    for north in np.arange(projected_bounds[1] + half_extent_m, projected_bounds[3], step):
        for east in np.arange(projected_bounds[0] + half_extent_m, projected_bounds[2], step):
            # Exact conservative envelope of the registered 24 km, 40° FOV,
            # 20° oblique camera. Coordinates follow the renderer's trapezoid:
            # far edge is wider/north, near edge is narrower/south. The supplied
            # half extent is 10.1 km, including the 30 m negative safety buffer.
            footprint = Polygon(
                [
                    (east - half_extent_m, north + half_extent_m * 0.552),
                    (east + half_extent_m, north + half_extent_m * 0.552),
                    (east + half_extent_m * 0.734, north - half_extent_m * 1.182),
                    (east - half_extent_m * 0.734, north - half_extent_m * 1.182),
                ]
            )
            if tree is not None and len(tree.query(footprint, predicate="intersects")):
                continue
            longitudes, latitudes = transform(crs, "EPSG:4326", [east], [north])
            negatives.append((longitudes[0], latitudes[0], [], prefix))
    return positives, negatives


def discover_candidates(
    config: SourceAcquisitionConfig,
    cache_dir: Path,
    *,
    client: httpx.Client | None = None,
) -> list[CandidateSeed]:
    """Build a deterministic, remotely pre-screened candidate pool larger than 120."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(config.seed)
    owns_client = client is None
    active = client or _http_client()
    pools: dict[tuple[str, str, str], list[tuple[float, float, list[str], str, str]]] = defaultdict(
        list
    )
    try:
        for region in config.regions:
            bounds = (region.west, region.south, region.east, region.north)
            centre_lon = (region.west + region.east) / 2
            zone = min(60, max(1, int((centre_lon + 180) // 6) + 1))
            crs = f"EPSG:{32600 + zone}"

            water_features = [
                feature
                for layer in WATER_LAYERS
                for feature in _cached_arcgis_features(
                    active,
                    NHD_SERVICE,
                    layer,
                    bounds,
                    cache_dir / f"{region.id}__nhd-layer-{layer}.json",
                )
            ]
            water_positive, water_negative = _vector_candidates(
                water_features,
                bounds,
                crs=crs,
                half_extent_m=config.negative_screen_half_extent_m,
                prefix="nhd",
            )
            road_features = [
                feature
                for layer in ROAD_LAYERS
                for feature in _cached_arcgis_features(
                    active,
                    TIGER_ROAD_SERVICE,
                    layer,
                    bounds,
                    cache_dir / f"{region.id}__tiger-roads-layer-{layer}.json",
                )
            ]
            road_positive, road_negative = _vector_candidates(
                road_features,
                bounds,
                crs=crs,
                half_extent_m=config.negative_screen_half_extent_m,
                prefix="tiger",
            )
            landcover = _fetch_nlcd(
                active,
                "Land-Cover-Native",
                bounds,
                config.discovery_landcover_year,
                cache_dir / f"{region.id}__nlcd-{config.discovery_landcover_year}.tif",
            )
            field_positive_projected, field_negative_projected = _discovery_field_points(
                landcover,
                # The maximum oblique camera footprint extends 1.182x the
                # nominal half extent in its long direction. Field negatives
                # must be clear across that full envelope, not only a nadir
                # square, or G2 will correctly reject them later.
                half_extent_m=config.negative_screen_half_extent_m * 1.182,
                generator=generator,
            )
            field_positive: list[tuple[float, float, list[str], str]] = [
                (longitude, latitude, [], "annual-nlcd")
                for longitude, latitude in _convert_points(field_positive_projected, "EPSG:5070")
            ]
            field_negative: list[tuple[float, float, list[str], str]] = [
                (longitude, latitude, [], "annual-nlcd")
                for longitude, latitude in _convert_points(field_negative_projected, "EPSG:5070")
            ]
            for feature, positive, negative in (
                ("water", water_positive, water_negative),
                ("road", road_positive, road_negative),
                ("field", field_positive, field_negative),
            ):
                pools[(region.group, feature, "positive")].extend(
                    (*candidate, region.id) for candidate in positive
                )
                pools[(region.group, feature, "negative")].extend(
                    (*candidate, region.id) for candidate in negative
                )
    finally:
        if owns_client:
            active.close()

    per_class = {"flight_corridor": 20, "regional_ood": 12, "cross_biome": 8}
    seeds: list[CandidateSeed] = []
    pending: list[tuple[str, str, str, tuple[float, float, list[str], str, str]]] = []
    for group, selected_per_class in per_class.items():
        minimum = math.ceil(selected_per_class / 2)
        requested = math.ceil(selected_per_class / 2 * config.candidate_multiplier)
        for feature in ("water", "road", "field"):
            for case_type in ("positive", "negative"):
                chosen = _spaced(
                    pools[(group, feature, case_type)],
                    requested,
                    minimum_m=max(1000, config.minimum_candidate_separation_m / 2),
                    generator=generator,
                )
                if len(chosen) < minimum:
                    raise DataValidationError(
                        f"Candidate discovery shortage for {(group, feature, case_type)}: "
                        f"needed final minimum {minimum}, found {len(chosen)}"
                    )
                pending.extend(
                    (group, feature, case_type, row)
                    for row in chosen[: min(requested, len(chosen))]
                )

    for identifier, (group, feature, case_type, row) in enumerate(
        sorted(
            pending,
            key=lambda value: (
                value[0],
                value[1],
                value[2],
                value[3][4],
                round(value[3][1], 7),
                round(value[3][0], 7),
            ),
        ),
        start=1,
    ):
        longitude, latitude, feature_ids, source, region_id = row
        seeds.append(
            CandidateSeed(
                candidate_id=f"candidate_{identifier:04d}",
                region_id=region_id,
                group=cast(Any, group),
                target_class=cast(Any, feature),
                case_type=cast(Any, case_type),
                longitude=longitude,
                latitude=latitude,
                discovery_source=source,
                discovery_feature_ids=feature_ids,
            )
        )
    return seeds


def write_candidate_manifest(path: Path, candidates: Sequence[CandidateSeed]) -> None:
    atomic_write_text(
        path,
        yaml.safe_dump(
            {
                "schema_version": "4.0.0",
                "candidate_count": len(candidates),
                "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
            },
            sort_keys=False,
        ),
    )


def _naip_items(
    client: httpx.Client,
    grid: _Grid,
    years: Sequence[int],
) -> tuple[int, list[dict[str, Any]]]:
    footprint = box(*grid.wgs84_bounds)
    for year in years:
        response = _retry_post(
            client,
            NAIP_STAC_SEARCH,
            json_value={
                "collections": [NAIP_COLLECTION],
                "bbox": list(grid.wgs84_bounds),
                "limit": 500,
                "query": {"naip:year": {"eq": year}},
            },
        )
        payload = response.json()
        items = payload.get("features", [])
        if not isinstance(items, list) or not items:
            continue
        coverage = unary_union(
            [shape(item["geometry"]) for item in items if isinstance(item.get("geometry"), dict)]
        )
        if coverage.buffer(1e-8).covers(footprint):
            return year, cast(list[dict[str, Any]], items)
    raise DataValidationError(
        f"No preferred NAIP year fully covers site bounds {grid.wgs84_bounds}"
    )


def _signed_planetary_computer_href(client: httpx.Client, href: str) -> str:
    """Sign a private Planetary Computer asset before GDAL reads its COG."""

    if not href.startswith(("https://", "http://")):
        return href
    # Signing is a shared public service; allow a longer bounded backoff for
    # transient 429s before treating the imagery fallback as unavailable.
    payload = _retry_get(
        client,
        PLANETARY_COMPUTER_SIGN,
        params={"href": href},
        attempts=6,
    ).json()
    signed = payload.get("href")
    if not isinstance(signed, str):
        raise DataValidationError("Planetary Computer did not return a signed NAIP asset URL")
    return signed


def _materialize_naip_cogs(
    items: Sequence[dict[str, Any]],
    grid: _Grid,
    destination: Path,
    *,
    client: httpx.Client,
) -> None:
    if _raster_matches(destination, grid, count=3, maximum_nodata_fraction=0.05):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    profile = grid.profile(dtype="uint8", count=3, nodata=0)
    profile.update({"compress": "jpeg", "jpeg_quality": 90, "photometric": "YCBCR"})
    profile.pop("predictor", None)
    environment = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    )
    with environment, rasterio.open(temporary, "w", **profile) as output:
        first = True
        for item in sorted(items, key=lambda row: str(row["id"])):
            asset = item.get("assets", {}).get("image", {})
            href = asset.get("href")
            if not isinstance(href, str):
                continue
            with rasterio.open(_signed_planetary_computer_href(client, href)) as source:
                count = min(3, source.count)
                reproject(
                    source=rasterio.band(source, list(range(1, count + 1))),
                    destination=rasterio.band(output, list(range(1, count + 1))),
                    src_transform=source.transform,
                    src_crs=source.crs,
                    dst_transform=grid.transform,
                    dst_crs=grid.crs,
                    src_nodata=0,
                    dst_nodata=0,
                    resampling=Resampling.bilinear,
                    init_dest_nodata=first,
                    num_threads=2,
                )
            first = False
    with rasterio.open(temporary) as dataset:
        sample = dataset.read(
            1,
            out_shape=(min(512, dataset.height), min(512, dataset.width)),
            resampling=Resampling.nearest,
        )
        if float(np.mean(sample == 0)) > 0.05:
            raise DataValidationError(f"NAIP mosaic has excessive nodata: {temporary}")
    os.replace(temporary, destination)


def _naip_export_chunk(
    client: httpx.Client,
    grid: _Grid,
    year: int,
    window: Window,
) -> tuple[Window, bytes]:
    left = grid.left + window.col_off * grid.resolution_m
    right = left + window.width * grid.resolution_m
    top = grid.top - window.row_off * grid.resolution_m
    bottom = top - window.height * grid.resolution_m
    payload = _retry_get(
        client,
        f"{NAIP_SERVICE}/exportImage",
        params={
            "f": "json",
            "bbox": f"{left},{bottom},{right},{top}",
            "bboxSR": str(grid.epsg),
            "imageSR": str(grid.epsg),
            "size": f"{int(window.width)},{int(window.height)}",
            "format": "tiff",
            "pixelType": "U8",
            "interpolation": "RSP_BilinearInterpolation",
            "mosaicRule": json.dumps(
                {
                    "mosaicMethod": "esriMosaicAttribute",
                    "where": f"Year = {year}",
                    "sortField": "acquisition_date",
                    "ascending": False,
                },
                separators=(",", ":"),
            ),
        },
    ).json()
    href = payload.get("href")
    if not isinstance(href, str):
        raise DataValidationError(f"NAIP export failed: {payload}")
    response = _retry_get(client, href)
    if not response.content.startswith((b"II", b"MM")):
        raise DataValidationError(f"NAIP export was not a TIFF: {href}")
    return window, response.content


def _materialize_naip(
    items: Sequence[dict[str, Any]],
    grid: _Grid,
    destination: Path,
    *,
    year: int | None = None,
    client: httpx.Client | None = None,
) -> None:
    """Materialize one NAIP mosaic, using bounded official exports in production."""

    if year is None:
        owns_client = client is None
        active = client or _http_client()
        try:
            _materialize_naip_cogs(items, grid, destination, client=active)
        finally:
            if owns_client:
                active.close()
        return
    if _raster_matches(destination, grid, count=3, maximum_nodata_fraction=0.05):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    profile = grid.profile(dtype="uint8", count=3, nodata=0)
    profile.update({"compress": "jpeg", "jpeg_quality": 90, "photometric": "YCBCR"})
    profile.pop("predictor", None)
    windows = [
        Window(column, row, min(4000, grid.width - column), min(4000, grid.height - row))
        for row in range(0, grid.height, 4000)
        for column in range(0, grid.width, 4000)
    ]
    owns_client = client is None
    active = client or _http_client()
    try:
        with rasterio.open(temporary, "w", **profile) as output:
            for offset in range(0, len(windows), 4):
                batch = windows[offset : offset + 4]
                with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                    chunks = list(
                        executor.map(
                            lambda window: _naip_export_chunk(
                                active,
                                grid,
                                year,
                                window,
                            ),
                            batch,
                        )
                    )
                for window, payload in chunks:
                    with MemoryFile(payload) as memory, memory.open() as source:
                        if source.count < 3:
                            raise DataValidationError("NAIP export has fewer than three bands")
                        data = source.read([1, 2, 3])
                    expected = (3, int(window.height), int(window.width))
                    if data.shape != expected:
                        raise DataValidationError(
                            f"NAIP export shape {data.shape} does not match {expected}"
                        )
                    output.write(data.astype(np.uint8), window=window)
        if not _raster_matches(
            temporary,
            grid,
            count=3,
            maximum_nodata_fraction=0.05,
        ):
            raise DataValidationError(f"NAIP export mosaic failed validation: {temporary}")
        os.replace(temporary, destination)
    finally:
        if owns_client:
            active.close()


def _raster_matches(
    path: Path,
    grid: _Grid,
    *,
    count: int,
    maximum_nodata_fraction: float | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        with rasterio.open(path) as source:
            if (
                source.width != grid.width
                or source.height != grid.height
                or source.count != count
                or source.crs is None
                or source.crs.to_string() != grid.crs
                or not source.transform.almost_equals(grid.transform)
            ):
                return False
            if maximum_nodata_fraction is not None:
                sample = source.read(
                    1,
                    out_shape=(min(512, source.height), min(512, source.width)),
                    resampling=Resampling.nearest,
                )
                if float(np.mean(sample == 0)) > maximum_nodata_fraction:
                    return False
    except (OSError, rasterio.errors.RasterioError):
        return False
    return True


def _query_osm_roads(
    client: httpx.Client,
    bounds: tuple[float, float, float, float],
    cache: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    if cache.is_file():
        payload = read_json(cache)
    else:
        west, south, east, north = bounds
        query = (
            "[out:json][timeout:180];("
            f'way["highway"~"^(motorway|trunk|primary|secondary)$"]'
            f"({south},{west},{north},{east});"
            ");out tags geom;"
        )
        payload = _retry_post(client, OSM_OVERPASS, data={"data": query}).json()
        write_json(cache, payload)
    geometries: list[dict[str, Any]] = []
    identifiers: list[str] = []
    for element in payload.get("elements", []):
        coordinates = [
            [float(point["lon"]), float(point["lat"])]
            for point in element.get("geometry", [])
            if "lon" in point and "lat" in point
        ]
        if len(coordinates) < 2:
            continue
        geometries.append({"type": "LineString", "coordinates": coordinates})
        identifiers.append(f"osm:way/{element['id']}")
    return geometries, identifiers


def _mask_profile(grid: _Grid) -> dict[str, Any]:
    return grid.profile(dtype="uint8", count=1, nodata=0)


def _write_vector_mask(
    geometries_wgs84: Sequence[dict[str, Any]],
    grid: _Grid,
    destination: Path,
    *,
    line_width_m: float | None = None,
) -> None:
    if _raster_matches(destination, grid, count=1):
        return
    projected = [
        transform_geom("EPSG:4326", grid.crs, geometry, antimeridian_cutting=False)
        for geometry in geometries_wgs84
    ]
    mask = rasterize(
        [(geometry, 1) for geometry in projected],
        out_shape=(grid.height, grid.width),
        transform=grid.transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    if line_width_m:
        radius = max(1, math.ceil(line_width_m / 2 / grid.resolution_m))
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (radius * 2 + 1, radius * 2 + 1),
            ),
        )
    temporary = destination.with_name(f".{destination.name}.partial")
    with rasterio.open(temporary, "w", **_mask_profile(grid)) as output:
        output.write((mask > 0).astype(np.uint8), 1)
    os.replace(temporary, destination)


def _write_class_mask(
    source_path: str | Path,
    grid: _Grid,
    destination: Path,
    values: set[int] | frozenset[int],
) -> None:
    if _raster_matches(destination, grid, count=1):
        return
    environment = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
    )
    with (
        environment,
        rasterio.open(source_path) as source,
        WarpedVRT(
            source,
            crs=grid.crs,
            transform=grid.transform,
            width=grid.width,
            height=grid.height,
            resampling=Resampling.nearest,
            nodata=0,
        ) as aligned,
    ):
        array = aligned.read(1)
    binary = np.isin(array, list(values)).astype(np.uint8)
    temporary = destination.with_name(f".{destination.name}.partial")
    with rasterio.open(temporary, "w", **_mask_profile(grid)) as output:
        output.write(binary, 1)
    os.replace(temporary, destination)


def _io_lulc_asset(
    client: httpx.Client,
    bounds: tuple[float, float, float, float],
    year: int,
) -> tuple[str, str, str]:
    response = _retry_post(
        client,
        NAIP_STAC_SEARCH,
        json_value={
            "collections": [IO_LULC_COLLECTION],
            "bbox": list(bounds),
            "datetime": f"{year}-01-01/{year + 1}-01-02",
            "limit": 20,
        },
    ).json()
    items = response.get("features", [])
    if not items:
        raise DataValidationError(f"No Impact Observatory LULC tile covers {bounds}")
    item = items[0]
    href = item.get("assets", {}).get("data", {}).get("href")
    if not isinstance(href, str):
        raise DataValidationError("Impact Observatory item has no data asset")
    signed = (
        _retry_get(
            client,
            PLANETARY_COMPUTER_SIGN,
            params={"href": href},
        )
        .json()
        .get("href")
    )
    if not isinstance(signed, str):
        raise DataValidationError("Planetary Computer did not sign the LULC asset")
    return str(item["id"]), href, signed


def _fetch_3dep(
    client: httpx.Client,
    grid: _Grid,
    raw: Path,
    destination: Path,
) -> None:
    if _raster_matches(destination, grid, count=1):
        return
    width = max(1, math.ceil((grid.right - grid.left) / 10))
    height = max(1, math.ceil((grid.top - grid.bottom) / 10))
    export = _retry_get(
        client,
        f"{THREE_DEP_SERVICE}/exportImage",
        attempts=6,
        params={
            "f": "json",
            "bbox": ",".join(str(value) for value in grid.bounds),
            "bboxSR": str(grid.epsg),
            "imageSR": str(grid.epsg),
            "size": f"{width},{height}",
            "format": "tiff",
            "pixelType": "F32",
            "interpolation": "RSP_BilinearInterpolation",
        },
    ).json()
    href = export.get("href")
    if not isinstance(href, str):
        raise DataValidationError(f"3DEP export failed: {export}")
    _atomic_bytes(raw, _retry_get(client, href, attempts=6).content)
    profile = grid.profile(dtype="float32", count=1, nodata=-9999)
    temporary = destination.with_name(f".{destination.name}.partial")
    with rasterio.open(raw) as source, rasterio.open(temporary, "w", **profile) as output:
        reproject(
            source=rasterio.band(source, 1),
            destination=rasterio.band(output, 1),
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=-9999,
            resampling=Resampling.bilinear,
            num_threads=2,
        )
    os.replace(temporary, destination)
    raw.unlink(missing_ok=True)


def _feature_rows(
    features: Sequence[dict[str, Any]],
    *,
    prefix: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    geometries: list[dict[str, Any]] = []
    identifiers: list[str] = []
    for feature in features:
        geometry = feature.get("geometry")
        if isinstance(geometry, dict):
            geometries.append(geometry)
            identifiers.append(_feature_id(feature, prefix=prefix))
    return geometries, identifiers


def _record(
    *,
    source_id: str,
    provider: str,
    product: str,
    version: str,
    acquisition_date: str,
    resolution_m: float,
    url: str,
    artifact: Path,
    license_name: str,
    terms_url: str,
    attribution: str,
    tile_ids: Sequence[str] = (),
    feature_ids: Sequence[str] = (),
) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        provider=provider,
        product=product,
        version=version,
        acquisition_date=acquisition_date,
        access_date=ACCESS_DATE,
        native_resolution_m=resolution_m,
        url=url,
        sha256=sha256_file(artifact),
        license=license_name,
        terms_url=terms_url,
        attribution=attribution,
        redistribution="allowed",
        tile_ids=sorted(set(tile_ids)),
        feature_ids=sorted(set(feature_ids)),
    )


def _site_from_yaml(path: Path) -> SiteSpec:
    return SiteSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def acquire_candidate(
    seed: CandidateSeed,
    config: SourceAcquisitionConfig,
    source_root: Path,
    *,
    flight_source_path: Path | None = None,
    client: httpx.Client | None = None,
) -> SiteSpec:
    """Materialize all aligned source layers and immutable provenance for one seed."""

    site_id = seed.candidate_id.replace("candidate", "site")
    directory = source_root / site_id
    site_path = directory / "site.yaml"
    if site_path.is_file() and (directory / "COMPLETE").is_file():
        return _site_from_yaml(site_path)
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    grid = _Grid(
        seed.longitude,
        seed.latitude,
        config.working_resolution_m,
        config.source_half_extent_m,
    )
    owns_client = client is None
    active = client or _http_client()
    try:
        imagery = directory / "imagery.tif"
        naip_year, naip_items = _naip_items(active, grid, config.preferred_naip_years)
        imagery_delivery_url = NAIP_SERVICE
        try:
            _materialize_naip(
                naip_items,
                grid,
                imagery,
                year=naip_year,
                client=active,
            )
        except DataValidationError as export_error:
            # ImageServer export tiles occasionally contain nodata holes even
            # when STAC reports full NAIP coverage. Reuse the same frozen NAIP
            # items via their signed Planetary Computer COG mirrors.
            try:
                _materialize_naip_cogs(naip_items, grid, imagery, client=active)
            except BaseException as cog_error:
                raise DataValidationError(
                    "NAIP export and signed COG fallback both failed: "
                    f"export={export_error}; cog={cog_error}"
                ) from cog_error
            imagery_delivery_url = NAIP_STAC_SEARCH
        naip_ids = [str(item["id"]) for item in naip_items]
        acquisition_dates = sorted(
            str(item.get("properties", {}).get("datetime", ""))[:10]
            for item in naip_items
            if item.get("properties", {}).get("datetime")
        )
        imagery_date = acquisition_dates[len(acquisition_dates) // 2]
        naip_resolution = min(
            float(item.get("properties", {}).get("gsd", config.working_resolution_m))
            for item in naip_items
        )

        water_primary_features = [
            feature
            for layer in WATER_LAYERS
            for feature in _arcgis_features(active, NHD_SERVICE, layer, grid.wgs84_bounds)
        ]
        water_primary_geometries, water_primary_ids = _feature_rows(
            water_primary_features, prefix="nhd"
        )
        water_secondary_features = _arcgis_features(
            active,
            TIGER_HYDRO_SERVICE,
            1,
            grid.wgs84_bounds,
        )
        water_secondary_geometries, water_secondary_ids = _feature_rows(
            water_secondary_features, prefix="tiger-hydro"
        )
        road_primary_features = [
            feature
            for layer in ROAD_LAYERS
            for feature in _arcgis_features(active, TIGER_ROAD_SERVICE, layer, grid.wgs84_bounds)
        ]
        road_primary_geometries, road_primary_ids = _feature_rows(
            road_primary_features, prefix="tiger-road"
        )
        road_secondary_geometries, road_secondary_ids = _query_osm_roads(
            active,
            grid.wgs84_bounds,
            raw / "osm_roads.json",
        )

        masks = {
            "water_primary": directory / "water_primary.tif",
            "water_secondary": directory / "water_secondary.tif",
            "road_primary": directory / "road_primary.tif",
            "road_secondary": directory / "road_secondary.tif",
            "field_primary": directory / "field_primary.tif",
            "field_secondary": directory / "field_secondary.tif",
        }
        _write_vector_mask(water_primary_geometries, grid, masks["water_primary"])
        _write_vector_mask(water_secondary_geometries, grid, masks["water_secondary"])
        _write_vector_mask(
            road_primary_geometries,
            grid,
            masks["road_primary"],
            line_width_m=16,
        )
        _write_vector_mask(
            road_secondary_geometries,
            grid,
            masks["road_secondary"],
            line_width_m=16,
        )

        cdl_year = naip_year
        cdl = _fetch_cdl(
            active,
            grid.wgs84_bounds,
            cdl_year,
            raw / f"cdl_{cdl_year}.tif",
        )
        nlcd_landcover = _fetch_nlcd(
            active,
            "Land-Cover-Native",
            grid.wgs84_bounds,
            naip_year,
            raw / f"annual_nlcd_landcover_{naip_year}.tif",
        )
        _write_class_mask(cdl, grid, masks["field_primary"], CULTIVATED_CDL_CODES)
        _write_class_mask(nlcd_landcover, grid, masks["field_secondary"], {82})

        io_item_id, io_href, io_read_href = _io_lulc_asset(
            active,
            grid.wgs84_bounds,
            config.detector_landcover_year,
        )
        detector_paths = {
            "water": directory / "water_detector_score.tif",
            "road": directory / "road_detector_score.tif",
            "field": directory / "field_detector_score.tif",
        }
        _write_class_mask(io_read_href, grid, detector_paths["water"], {1})
        _write_class_mask(io_read_href, grid, detector_paths["field"], {5})
        nlcd_impervious = _fetch_nlcd(
            active,
            "Impervious-Descriptor-Native",
            grid.wgs84_bounds,
            naip_year,
            raw / f"annual_nlcd_impervious_{naip_year}.tif",
        )
        # Annual NLCD Collection 1.2 uses 1=road; older services may expose
        # legacy road subclasses 20-23, so accept both documented encodings.
        _write_class_mask(nlcd_impervious, grid, detector_paths["road"], {1, 20, 21, 22, 23})

        dem = directory / "dem.tif"
        _fetch_3dep(active, grid, raw / "3dep_native.tif", dem)

        primary = {
            "water": masks["water_primary"],
            "road": masks["road_primary"],
            "field": masks["field_primary"],
        }
        secondary = {
            "water": masks["water_secondary"],
            "road": masks["road_secondary"],
            "field": masks["field_secondary"],
        }
        feature_ids = {
            "water": water_primary_ids + water_secondary_ids,
            "road": road_primary_ids + road_secondary_ids,
            "field": [
                f"cdl:{cdl_year}:{seed.longitude:.7f}:{seed.latitude:.7f}",
                f"annual-nlcd:{naip_year}:{seed.longitude:.7f}:{seed.latitude:.7f}",
            ],
        }
        imagery_record = _record(
            source_id=f"{site_id}-naip",
            provider="USDA Farm Service Agency; served by USGS and mirrored by Microsoft",
            product="National Agriculture Imagery Program",
            version=str(naip_year),
            acquisition_date=imagery_date,
            resolution_m=naip_resolution,
            url=imagery_delivery_url,
            artifact=imagery,
            license_name="U.S. federal public-domain data",
            terms_url=NAIP_TERMS,
            attribution="USDA Farm Service Agency National Agriculture Imagery Program",
            tile_ids=naip_ids,
        )
        feature_sources = {
            "water": [
                _record(
                    source_id=f"{site_id}-nhd-water",
                    provider="U.S. Geological Survey",
                    product="National Hydrography Dataset / 3DHP migration service",
                    version="service snapshot 2026-07",
                    acquisition_date="2026-07-01",
                    resolution_m=grid.resolution_m,
                    url=NHD_SERVICE,
                    artifact=primary["water"],
                    license_name="U.S. federal public-domain data",
                    terms_url=NHD_TERMS,
                    attribution="U.S. Geological Survey 3D Hydrography Program",
                    feature_ids=water_primary_ids,
                ),
                _record(
                    source_id=f"{site_id}-tiger-water",
                    provider="U.S. Census Bureau",
                    product="TIGERweb Areal Hydrography",
                    version="2025",
                    acquisition_date="2025-01-01",
                    resolution_m=grid.resolution_m,
                    url=TIGER_HYDRO_SERVICE,
                    artifact=secondary["water"],
                    license_name="U.S. federal public-domain data",
                    terms_url=TIGER_TERMS,
                    attribution="U.S. Census Bureau TIGER/Line",
                    feature_ids=water_secondary_ids,
                ),
            ],
            "road": [
                _record(
                    source_id=f"{site_id}-tiger-road",
                    provider="U.S. Census Bureau",
                    product="TIGERweb Primary and Secondary Roads",
                    version="2025",
                    acquisition_date="2025-01-01",
                    resolution_m=grid.resolution_m,
                    url=TIGER_ROAD_SERVICE,
                    artifact=primary["road"],
                    license_name="U.S. federal public-domain data",
                    terms_url=TIGER_TERMS,
                    attribution="U.S. Census Bureau TIGER/Line",
                    feature_ids=road_primary_ids,
                ),
                _record(
                    source_id=f"{site_id}-osm-road",
                    provider="OpenStreetMap contributors",
                    product="OpenStreetMap major-road extract",
                    version=ACCESS_DATE,
                    acquisition_date=ACCESS_DATE,
                    resolution_m=grid.resolution_m,
                    url=OSM_OVERPASS,
                    artifact=secondary["road"],
                    license_name="Open Data Commons Open Database License 1.0",
                    terms_url=OSM_TERMS,
                    attribution="© OpenStreetMap contributors",
                    feature_ids=road_secondary_ids,
                ),
            ],
            "field": [
                _record(
                    source_id=f"{site_id}-cdl-field",
                    provider="USDA National Agricultural Statistics Service",
                    product="Cropland Data Layer",
                    version=str(cdl_year),
                    acquisition_date=f"{cdl_year}-07-01",
                    resolution_m=10 if cdl_year >= 2025 else 30,
                    url=(CDL_RELEASE_URL if _cached_cdl_archive(cdl_year) else CDL_SERVICE),
                    artifact=primary["field"],
                    license_name="U.S. federal public-domain data",
                    terms_url=CDL_TERMS,
                    attribution="USDA NASS Cropland Data Layer",
                    feature_ids=feature_ids["field"][:1],
                ),
                _record(
                    source_id=f"{site_id}-nlcd-field",
                    provider="U.S. Geological Survey EROS",
                    product="Annual NLCD Land Cover",
                    version=f"Collection 1.2, {naip_year}",
                    acquisition_date=f"{naip_year}-01-01",
                    resolution_m=30,
                    url=_nlcd_url("Land-Cover-Native")[0],
                    artifact=secondary["field"],
                    license_name="CC0 1.0 / no restrictions",
                    terms_url=NLCD_TERMS,
                    attribution="U.S. Geological Survey Annual NLCD",
                    feature_ids=feature_ids["field"][1:],
                ),
            ],
        }
        detector_sources = {
            "water": _record(
                source_id=f"{site_id}-io-water-detector",
                provider="Impact Observatory, Microsoft, and Esri",
                product="10m Annual Land Use Land Cover 9-class V1",
                version=str(config.detector_landcover_year),
                acquisition_date=f"{config.detector_landcover_year}-01-01",
                resolution_m=10,
                url=io_href,
                artifact=detector_paths["water"],
                license_name="Creative Commons Attribution 4.0",
                terms_url=IO_LULC_TERMS,
                attribution="Impact Observatory, Microsoft, and Esri 10m Annual LULC",
                tile_ids=[io_item_id],
            ),
            "road": _record(
                source_id=f"{site_id}-nlcd-road-detector",
                provider="U.S. Geological Survey EROS",
                product="Annual NLCD Impervious Descriptor supervised classifier",
                version=f"Collection 1.2, {naip_year}",
                acquisition_date=f"{naip_year}-01-01",
                resolution_m=30,
                url=_nlcd_url("Impervious-Descriptor-Native")[0],
                artifact=detector_paths["road"],
                license_name="CC0 1.0 / no restrictions",
                terms_url=NLCD_IMPERVIOUS_TERMS,
                attribution="U.S. Geological Survey Annual NLCD",
            ),
            "field": _record(
                source_id=f"{site_id}-io-field-detector",
                provider="Impact Observatory, Microsoft, and Esri",
                product="10m Annual Land Use Land Cover 9-class V1",
                version=str(config.detector_landcover_year),
                acquisition_date=f"{config.detector_landcover_year}-01-01",
                resolution_m=10,
                url=io_href,
                artifact=detector_paths["field"],
                license_name="Creative Commons Attribution 4.0",
                terms_url=IO_LULC_TERMS,
                attribution="Impact Observatory, Microsoft, and Esri 10m Annual LULC",
                tile_ids=[io_item_id],
            ),
        }
        terrain = _record(
            source_id=f"{site_id}-3dep",
            provider="U.S. Geological Survey",
            product="3D Elevation Program seamless DEM",
            version=f"service snapshot {ACCESS_DATE}",
            acquisition_date=ACCESS_DATE,
            resolution_m=10,
            url=THREE_DEP_SERVICE,
            artifact=dem,
            license_name="U.S. federal public-domain data",
            terms_url=THREE_DEP_TERMS,
            attribution="U.S. Geological Survey 3D Elevation Program",
        )
        source_manifest = SourceManifest(
            site_id=site_id,
            imagery=imagery_record,
            feature_sources=cast(Any, feature_sources),
            detector_sources=cast(Any, detector_sources),
            terrain=terrain,
            flight_source_sha256=(
                sha256_file(flight_source_path)
                if flight_source_path is not None and flight_source_path.is_file()
                else None
            ),
        )
        source_manifest_path = directory / "source_manifest.json"
        write_json(source_manifest_path, source_manifest.model_dump(mode="json"))
        label_date = f"{cdl_year}-07-01" if seed.target_class == "field" else "2025-01-01"
        site = SiteSpec(
            site_id=site_id,
            group=seed.group,
            target_class=seed.target_class,
            case_type=seed.case_type,
            longitude=seed.longitude,
            latitude=seed.latitude,
            imagery_path=imagery,
            primary_mask_paths=cast(Any, primary),
            secondary_mask_paths=cast(Any, secondary),
            detector_score_paths=cast(Any, detector_paths),
            dem_path=dem,
            source_manifest_path=source_manifest_path,
            source_tile_ids=naip_ids,
            feature_ids=sorted(set(feature_ids[seed.target_class])),
            imagery_date=imagery_date,
            label_date=label_date,
            native_resolution_m=grid.resolution_m,
        )
        atomic_write_text(site_path, yaml.safe_dump(site.model_dump(mode="json"), sort_keys=False))
        write_json(
            directory / "acquisition.json",
            {
                "schema_version": "4.0.0",
                "candidate": seed.model_dump(mode="json"),
                "grid": {
                    "crs": grid.crs,
                    "bounds": grid.bounds,
                    "width": grid.width,
                    "height": grid.height,
                    "resolution_m": grid.resolution_m,
                },
                "naip_year": naip_year,
                "naip_items": naip_ids,
                "completed_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        atomic_write_text(directory / "COMPLETE", "CanyonBench-Trace source site complete\n")
        return site
    finally:
        if owns_client:
            active.close()


def acquire_candidates(
    candidates: Sequence[CandidateSeed],
    config: SourceAcquisitionConfig,
    source_root: Path,
    output_manifest: Path,
    report_path: Path,
    *,
    flight_source_path: Path | None = None,
    start: int = 0,
    limit: int | None = None,
    continue_on_error: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[SiteSpec]:
    """Resume materialization of a candidate pool and preserve every failure."""

    selected = list(candidates[start : None if limit is None else start + limit])
    sites: list[SiteSpec] = []
    failures: list[dict[str, str]] = []
    with _http_client() as client:
        for index, candidate in enumerate(selected, start=start + 1):
            if progress is not None:
                progress(
                    f"[{index}/{start + len(selected)}] acquiring "
                    f"{candidate.candidate_id} ({candidate.group}/"
                    f"{candidate.target_class}/{candidate.case_type})"
                )
            try:
                site = acquire_candidate(
                    candidate,
                    config,
                    source_root,
                    flight_source_path=flight_source_path,
                    client=client,
                )
                sites.append(site)
                if progress is not None:
                    progress(f"[{index}/{start + len(selected)}] complete: {site.site_id}")
            except BaseException as exc:
                failures.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                if progress is not None:
                    progress(
                        f"[{index}/{start + len(selected)}] failed: "
                        f"{candidate.candidate_id}: {type(exc).__name__}: {exc}"
                    )
                if not continue_on_error:
                    raise
            write_json(
                report_path,
                {
                    "schema_version": "4.0.0",
                    "requested": len(selected),
                    "processed": index - start,
                    "completed": len(sites),
                    "failed": len(failures),
                    "failures": failures,
                },
            )
    existing: list[SiteSpec] = []
    for path in sorted(source_root.glob("site_*/site.yaml")):
        try:
            existing.append(_site_from_yaml(path))
        except ValueError:
            continue
    atomic_write_text(
        output_manifest,
        yaml.safe_dump(
            {"sites": [site.model_dump(mode="json") for site in existing]},
            sort_keys=False,
        ),
    )
    return sites
