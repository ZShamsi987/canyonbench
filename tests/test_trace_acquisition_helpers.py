"""Network, provenance, and rasterization contracts of source acquisition.

Every request path is exercised against a mock transport: the acquisition step is
the one place where a silent retry, a duplicated page, or a half-written artifact
would corrupt a frozen source manifest.
"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from canyonbench.exceptions import DataValidationError
from canyonbench.trace.acquisition import (
    _arcgis_features,
    _atomic_bytes,
    _cached_arcgis_features,
    _convert_points,
    _discovery_field_points,
    _feature_rows,
    _fetch_cdl,
    _fetch_nlcd,
    _Grid,
    _haversine_m,
    _io_lulc_asset,
    _mask_profile,
    _nlcd_url,
    _query_osm_roads,
    _raster_matches,
    _record,
    _retry_get,
    _retry_post,
    _site_from_yaml,
    _to_projected_geometry,
    _write_vector_mask,
)


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://example.invalid")


def _grid() -> _Grid:
    return _Grid(-111.5, 36.7, resolution_m=10.0, half_extent_m=200.0)


def test_site_grid_is_square_even_and_utm_anchored() -> None:
    grid = _grid()
    assert grid.width == grid.height
    assert grid.width % 2 == 0
    # Northern hemisphere UTM zone 12 covers the flight corridor.
    assert grid.crs == "EPSG:32612"
    assert grid.transform == from_origin(grid.left, grid.top, 10.0, 10.0)
    assert grid.bounds == (grid.left, grid.bottom, grid.right, grid.top)
    assert grid.right - grid.left == pytest.approx(grid.width * 10.0)
    west, south, east, north = grid.wgs84_bounds
    assert west < -111.5 < east
    assert south < 36.7 < north
    profile = grid.profile(dtype="uint8")
    assert profile["crs"] == grid.crs
    assert profile["predictor"] == 2
    assert grid.profile(dtype="float32")["predictor"] == 3
    southern = _Grid(-111.5, -36.7, resolution_m=10.0, half_extent_m=200.0)
    assert southern.crs == "EPSG:32712"


def test_retry_get_recovers_then_aborts_with_provenance(monkeypatch) -> None:
    monkeypatch.setattr("canyonbench.trace.acquisition.time.sleep", lambda _seconds: None)
    attempts = {"count": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": True})

    with _client(flaky) as client:
        assert _retry_get(client, "https://example.invalid/x").json() == {"ok": True}
    assert attempts["count"] == 3

    with (
        _client(lambda request: httpx.Response(500)) as client,
        pytest.raises(DataValidationError, match="failed after 2 attempts"),
    ):
        _retry_get(client, "https://example.invalid/x", attempts=2)

    with (
        _client(lambda request: httpx.Response(500)) as client,
        pytest.raises(DataValidationError, match="failed after 1 attempts"),
    ):
        _retry_post(client, "https://example.invalid/x", data={"a": "b"}, attempts=1)


def test_fetch_cdl_uses_a_local_official_archive_when_available(tmp_path, monkeypatch) -> None:
    archive_dir = tmp_path / "cdl"
    archive_dir.mkdir()
    archive = archive_dir / "2024_30m_cdls.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("2024_30m_cdls.tif", b"not-read-by-fetch")
    monkeypatch.setenv("CANYONBENCH_CDL_CACHE_DIR", str(archive_dir))

    def unexpected_request(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"cached CDL fetch made a network request: {request.url}")

    with _client(unexpected_request) as client:
        result = _fetch_cdl(client, (-112.0, 36.0, -111.0, 37.0), 2024, tmp_path / "cdl.tif")

    assert result == f"/vsizip//{archive.resolve().as_posix().lstrip('/')}/2024_30m_cdls.tif"


def test_arcgis_features_chunk_ids_without_loss_or_duplication() -> None:
    identifiers = list(range(1, 601))
    seen: list[list[int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if params.get("returnIdsOnly") == "true":
            return httpx.Response(200, json={"objectIds": list(reversed(identifiers))})
        chunk = [int(value) for value in params["objectIds"].split(",")]
        seen.append(chunk)
        return httpx.Response(
            200,
            json={
                "features": [
                    {
                        "properties": {"OBJECTID": value},
                        "geometry": {"type": "Point", "coordinates": [-111.0, 36.5]},
                    }
                    for value in chunk
                ]
            },
        )

    with _client(handler) as client:
        features = _arcgis_features(
            client, "https://example.invalid/svc", 0, (-112.0, 36.0, -111.0, 37.0)
        )
    assert [len(chunk) for chunk in seen] == [250, 250, 100]
    flat = [value for chunk in seen for value in chunk]
    assert flat == identifiers, "object IDs must be requested in sorted order exactly once"
    assert len(features) == 600

    geometries, feature_ids = _feature_rows(features, prefix="tiger")
    assert len(geometries) == 600
    assert feature_ids[0] == "tiger:1"
    # Features without geometry are dropped rather than silently identified.
    partial, partial_ids = _feature_rows([{"properties": {"OBJECTID": 9}}], prefix="tiger")
    assert partial == [] and partial_ids == []


def test_arcgis_errors_and_malformed_payloads_fail_closed() -> None:
    def id_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"code": 400}})

    with (
        _client(id_error) as client,
        pytest.raises(DataValidationError, match="ID query failed"),
    ):
        _arcgis_features(client, "https://example.invalid/svc", 0, (-112.0, 36.0, -111.0, 37.0))

    def feature_error(request: httpx.Request) -> httpx.Response:
        if dict(request.url.params).get("returnIdsOnly") == "true":
            return httpx.Response(200, json={"objectIds": [1]})
        return httpx.Response(200, json={"error": {"code": 500}})

    with (
        _client(feature_error) as client,
        pytest.raises(DataValidationError, match="feature query failed"),
    ):
        _arcgis_features(client, "https://example.invalid/svc", 0, (-112.0, 36.0, -111.0, 37.0))

    def not_a_list(request: httpx.Request) -> httpx.Response:
        if dict(request.url.params).get("returnIdsOnly") == "true":
            return httpx.Response(200, json={"objectIds": [1]})
        return httpx.Response(200, json={"features": {"unexpected": True}})

    with (
        _client(not_a_list) as client,
        pytest.raises(DataValidationError, match="no feature list"),
    ):
        _arcgis_features(client, "https://example.invalid/svc", 0, (-112.0, 36.0, -111.0, 37.0))


def test_cached_arcgis_query_is_frozen_after_the_first_call(tmp_path) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if params.get("returnIdsOnly") == "true":
            calls["count"] += 1
            return httpx.Response(200, json={"objectIds": [7]})
        return httpx.Response(
            200,
            json={
                "features": [
                    {
                        "properties": {"PERMANENT_IDENTIFIER": "abc"},
                        "geometry": {"type": "Point", "coordinates": [-111.0, 36.5]},
                    }
                ]
            },
        )

    cache = tmp_path / "cache" / "nhd.json"
    with _client(handler) as client:
        first = _cached_arcgis_features(
            client, "https://example.invalid/svc", 0, (-112.0, 36.0, -111.0, 37.0), cache
        )
        second = _cached_arcgis_features(
            client, "https://example.invalid/svc", 0, (-112.0, 36.0, -111.0, 37.0), cache
        )
    assert first == second
    assert calls["count"] == 1, "a rerun must reuse the frozen query, not re-hit the service"
    _, identifiers = _feature_rows(second, prefix="nhd")
    assert identifiers == ["nhd:abc"]

    cache.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with (
        _client(handler) as client,
        pytest.raises(DataValidationError, match="not a feature list"),
    ):
        _cached_arcgis_features(
            client, "https://example.invalid/svc", 0, (-112.0, 36.0, -111.0, 37.0), cache
        )


def test_osm_road_query_caches_and_skips_degenerate_ways(tmp_path) -> None:
    payload = {
        "elements": [
            {
                "id": 1,
                "geometry": [
                    {"lon": -111.0, "lat": 36.5},
                    {"lon": -110.99, "lat": 36.51},
                ],
            },
            {"id": 2, "geometry": [{"lon": -111.0, "lat": 36.5}]},
            {"id": 3, "geometry": []},
        ]
    }
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        assert request.method == "POST"
        return httpx.Response(200, json=payload)

    cache = tmp_path / "osm.json"
    with _client(handler) as client:
        geometries, identifiers = _query_osm_roads(client, (-112.0, 36.0, -111.0, 37.0), cache)
        _query_osm_roads(client, (-112.0, 36.0, -111.0, 37.0), cache)
    assert calls["count"] == 1
    assert identifiers == ["osm:way/1"]
    assert geometries[0]["type"] == "LineString"


def test_atomic_bytes_leaves_no_partial_file(tmp_path) -> None:
    target = tmp_path / "nested" / "artifact.bin"
    _atomic_bytes(target, b"payload")
    assert target.read_bytes() == b"payload"
    _atomic_bytes(target, b"replacement")
    assert target.read_bytes() == b"replacement"
    assert not list(tmp_path.joinpath("nested").glob(".artifact.bin.*"))


def test_vector_mask_rasterizes_lines_with_a_width_and_is_idempotent(tmp_path) -> None:
    grid = _grid()
    west, south, east, north = grid.wgs84_bounds
    line = {
        "type": "LineString",
        "coordinates": [
            [(west + east) / 2, south + (north - south) * 0.1],
            [(west + east) / 2, north - (north - south) * 0.1],
        ],
    }
    destination = tmp_path / "road_mask.tif"
    _write_vector_mask([line], grid, destination, line_width_m=60.0)
    with rasterio.open(destination) as source:
        mask = source.read(1)
        assert source.crs.to_string() == grid.crs
        assert (source.width, source.height) == (grid.width, grid.height)
    assert set(np.unique(mask).tolist()) <= {0, 1}
    assert mask.any()
    # A 60 m line on a 10 m grid must be several pixels wide, not hairline;
    # the elliptical kernel legitimately tapers at the two endpoints.
    widths = mask.sum(axis=1)
    assert float(np.median(widths[widths > 0])) >= 6
    assert _raster_matches(destination, grid, count=1)

    before = destination.stat().st_mtime_ns
    _write_vector_mask([line], grid, destination, line_width_m=60.0)
    assert destination.stat().st_mtime_ns == before, "a matching mask must not be rewritten"
    assert not list(tmp_path.glob(".road_mask.tif.partial"))


def test_raster_matches_rejects_every_mismatch(tmp_path) -> None:
    grid = _grid()
    assert not _raster_matches(tmp_path / "absent.tif", grid, count=1)

    def write(path: Path, **overrides: Any) -> Path:
        profile = _mask_profile(grid)
        profile.update(overrides)
        with rasterio.open(path, "w", **profile) as output:
            output.write(np.ones((profile["height"], profile["width"]), np.uint8), 1)
        return path

    assert _raster_matches(write(tmp_path / "ok.tif"), grid, count=1)
    assert not _raster_matches(write(tmp_path / "wide.tif", width=grid.width + 2), grid, count=1)
    assert not _raster_matches(write(tmp_path / "crs.tif", crs="EPSG:4326"), grid, count=1)
    assert not _raster_matches(
        write(tmp_path / "shift.tif", transform=from_origin(0, 0, 10, 10)), grid, count=1
    )
    assert not _raster_matches(write(tmp_path / "ok.tif"), grid, count=2)
    (tmp_path / "corrupt.tif").write_bytes(b"not a raster")
    assert not _raster_matches(tmp_path / "corrupt.tif", grid, count=1)

    empty = tmp_path / "empty.tif"
    with rasterio.open(empty, "w", **_mask_profile(grid)) as output:
        output.write(np.zeros((grid.height, grid.width), np.uint8), 1)
    assert _raster_matches(empty, grid, count=1)
    assert not _raster_matches(empty, grid, count=1, maximum_nodata_fraction=0.5)


def test_projection_helpers_round_trip_and_measure_distance() -> None:
    grid = _grid()
    projected = _to_projected_geometry(
        {"type": "Point", "coordinates": [-111.5, 36.7]},
        grid.crs,
    )
    assert grid.left < projected.x < grid.right
    assert grid.bottom < projected.y < grid.top

    back = _convert_points([(projected.x, projected.y)], grid.crs)
    assert back[0][0] == pytest.approx(-111.5, abs=1e-6)
    assert back[0][1] == pytest.approx(36.7, abs=1e-6)
    assert _convert_points([], grid.crs) == []

    assert _haversine_m(-111.5, 36.7, -111.5, 36.7) == 0.0
    one_degree_north = _haversine_m(-111.5, 36.0, -111.5, 37.0)
    assert one_degree_north == pytest.approx(111_195, rel=0.01)
    assert _haversine_m(-111.5, 36.7, -110.5, 36.7) < one_degree_north


def test_nlcd_workspace_and_coverage_are_paired() -> None:
    base, coverage = _nlcd_url("nlcd_lndcov")
    assert base.endswith("/mrlc_nlcd_lndcov_conus_year_data/wcs")
    assert coverage == "mrlc_nlcd_lndcov_conus_year_data:nlcd_lndcov_conus_year_data"


def test_source_record_hashes_the_artifact_and_dedupes_identifiers(tmp_path) -> None:
    artifact = tmp_path / "layer.tif"
    artifact.write_bytes(b"content")
    record = _record(
        source_id="water_primary",
        provider="USGS",
        product="NHD",
        version="1",
        acquisition_date="2025-06-01",
        resolution_m=1.0,
        url="https://example.invalid/nhd",
        artifact=artifact,
        license_name="public domain",
        terms_url="https://example.invalid/terms",
        attribution="USGS",
        tile_ids=["b", "a", "a"],
        feature_ids=["f2", "f1", "f2"],
    )
    assert record.tile_ids == ["a", "b"]
    assert record.feature_ids == ["f1", "f2"]
    assert record.redistribution == "allowed"
    assert len(record.sha256) == 64
    artifact.write_bytes(b"changed")
    assert (
        _record(
            source_id="water_primary",
            provider="USGS",
            product="NHD",
            version="1",
            acquisition_date="2025-06-01",
            resolution_m=1.0,
            url="https://example.invalid/nhd",
            artifact=artifact,
            license_name="public domain",
            terms_url="https://example.invalid/terms",
            attribution="USGS",
        ).sha256
        != record.sha256
    )


def test_site_yaml_loader_validates_the_strict_contract(tmp_path) -> None:
    path = tmp_path / "site.yaml"
    path.write_text("site_id: not-a-site-id\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _site_from_yaml(path)


def _tiff_bytes(values: np.ndarray, crs: str = "EPSG:5070") -> bytes:
    from io import BytesIO

    profile = {
        "driver": "GTiff",
        "width": values.shape[1],
        "height": values.shape[0],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": from_origin(-1_200_000, 1_900_000, 30, 30),
    }
    buffer = BytesIO()
    with rasterio.io.MemoryFile() as memory:
        with memory.open(**profile) as output:
            output.write(values, 1)
        buffer.write(memory.read())
    return buffer.getvalue()


def test_land_cover_fetchers_reject_non_raster_responses_and_cache(tmp_path) -> None:
    payload = _tiff_bytes(np.full((8, 8), 82, np.uint8))
    bounds = (-112.0, 36.0, -111.9, 36.1)
    calls = {"nlcd": 0, "cdl": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "geoserver" in url:
            calls["nlcd"] += 1
            return httpx.Response(200, content=payload, headers={"content-type": "image/tiff"})
        if url.endswith(".tif"):
            return httpx.Response(200, content=payload, headers={"content-type": "image/tiff"})
        calls["cdl"] += 1
        return httpx.Response(
            200,
            text="<returnURL>https://example.invalid/cdl.tif</returnURL>",
            headers={"content-type": "text/xml"},
        )

    nlcd = tmp_path / "nlcd.tif"
    with _client(handler) as client:
        _fetch_nlcd(client, "nlcd_lndcov", bounds, 2024, nlcd)
        _fetch_nlcd(client, "nlcd_lndcov", bounds, 2024, nlcd)
    assert nlcd.read_bytes() == payload
    assert calls["nlcd"] == 1, "an existing artifact must not be re-downloaded"

    cdl = tmp_path / "cdl.tif"
    with _client(handler) as client:
        _fetch_cdl(client, bounds, 2024, cdl)
    assert cdl.read_bytes() == payload

    def html_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text="<html>outage</html>", headers={"content-type": "text/html"}
        )

    with (
        _client(html_error) as client,
        pytest.raises(DataValidationError, match="Annual NLCD returned"),
    ):
        _fetch_nlcd(client, "nlcd_lndcov", bounds, 2024, tmp_path / "bad_nlcd.tif")
    with (
        _client(html_error) as client,
        pytest.raises(DataValidationError, match="no download URL"),
    ):
        _fetch_cdl(client, bounds, 2024, tmp_path / "bad_cdl.tif")

    def not_a_tiff(request: httpx.Request) -> httpx.Response:
        if "cdl.tif" in str(request.url):
            return httpx.Response(200, content=b"<html>nope</html>")
        return httpx.Response(200, text="<returnURL>https://example.invalid/cdl.tif</returnURL>")

    with (
        _client(not_a_tiff) as client,
        pytest.raises(DataValidationError, match="not a TIFF"),
    ):
        _fetch_cdl(client, bounds, 2024, tmp_path / "worse_cdl.tif")


def test_lulc_asset_signing_fails_closed_at_each_step() -> None:
    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"features": []})

    with (
        _client(empty) as client,
        pytest.raises(DataValidationError, match="No Impact Observatory"),
    ):
        _io_lulc_asset(client, (-112.0, 36.0, -111.0, 37.0), 2024)

    def no_asset(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"features": [{"id": "tile", "assets": {}}]})

    with (
        _client(no_asset) as client,
        pytest.raises(DataValidationError, match="no data asset"),
    ):
        _io_lulc_asset(client, (-112.0, 36.0, -111.0, 37.0), 2024)

    def unsigned(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "features": [
                        {
                            "id": "tile",
                            "assets": {"data": {"href": "https://example.invalid/lulc.tif"}},
                        }
                    ]
                },
            )
        return httpx.Response(200, json={})

    with (
        _client(unsigned) as client,
        pytest.raises(DataValidationError, match="did not sign"),
    ):
        _io_lulc_asset(client, (-112.0, 36.0, -111.0, 37.0), 2024)

    def signed(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "features": [
                        {
                            "id": "tile-1",
                            "assets": {"data": {"href": "https://example.invalid/lulc.tif"}},
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"href": "https://example.invalid/lulc.tif?sig=abc"})

    with _client(signed) as client:
        identifier, href, signed_href = _io_lulc_asset(client, (-112.0, 36.0, -111.0, 37.0), 2024)
    assert identifier == "tile-1"
    assert href == "https://example.invalid/lulc.tif"
    assert "sig=abc" in signed_href


def test_field_discovery_separates_cultivated_cores_from_clear_footprints(tmp_path) -> None:
    values = np.zeros((120, 120), np.uint8)
    values[10:60, 10:60] = 82  # cultivated block
    path = tmp_path / "cdl.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=120,
        height=120,
        count=1,
        dtype="uint8",
        crs="EPSG:5070",
        transform=from_origin(-1_200_000, 1_900_000, 30, 30),
    ) as output:
        output.write(values, 1)

    generator = np.random.default_rng(11)
    positive, negative = _discovery_field_points(path, half_extent_m=300.0, generator=generator)
    assert positive, "cultivated cores must yield positive candidates"
    assert negative, "clear footprints must yield negative candidates"

    with rasterio.open(path) as dataset:
        for east, north in positive[:50]:
            row, column = dataset.index(east, north)
            assert values[row, column] == 82
        radius = math.ceil(300.0 / 30)
        for east, north in negative[:50]:
            row, column = dataset.index(east, north)
            window = values[row - radius : row + radius + 1, column - radius : column + radius + 1]
            assert not (window == 82).any(), "a negative footprint must be entirely clear"

    # Discovery is seeded, so the same generator state reproduces the same set.
    repeat_positive, repeat_negative = _discovery_field_points(
        path, half_extent_m=300.0, generator=np.random.default_rng(11)
    )
    assert repeat_positive == positive
    assert repeat_negative == negative

    empty = tmp_path / "empty.tif"
    with rasterio.open(
        empty,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=1,
        dtype="uint8",
        crs="EPSG:5070",
        transform=from_origin(-1_200_000, 1_900_000, 30, 30),
    ) as output:
        output.write(np.zeros((8, 8), np.uint8), 1)
    none_positive, _ = _discovery_field_points(
        empty, half_extent_m=3000.0, generator=np.random.default_rng(3)
    )
    assert none_positive == []


def test_grid_extent_covers_the_requested_half_extent() -> None:
    for half_extent in (150.0, 1000.0, 12345.0):
        grid = _Grid(-111.5, 36.7, resolution_m=30.0, half_extent_m=half_extent)
        assert (grid.right - grid.left) / 2 >= half_extent
        assert grid.width == math.ceil(2 * half_extent / 30.0) + (
            0 if math.ceil(2 * half_extent / 30.0) % 2 == 0 else 1
        )
