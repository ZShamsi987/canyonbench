from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio  # type: ignore[import-untyped]
import yaml
from rasterio.transform import from_origin  # type: ignore[import-untyped]

from canyonbench.trace.acquisition import (
    _feature_id,
    _Grid,
    _materialize_naip,
    _spaced,
    _write_class_mask,
    write_candidate_manifest,
)
from canyonbench.trace.config import load_candidate_seeds
from canyonbench.trace.schemas import CandidateSeed


def test_feature_fallback_identifier_is_process_stable() -> None:
    feature = {
        "properties": {},
        "geometry": {
            "type": "LineString",
            "coordinates": [[-111.0, 36.5], [-110.9, 36.6]],
        },
    }
    first = _feature_id(feature, prefix="test")
    second = _feature_id(json.loads(json.dumps(feature)), prefix="test")
    assert first == second
    assert first.startswith("test:sha256:")


def test_candidate_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "candidates.yaml"
    candidate = CandidateSeed(
        candidate_id="candidate_0001",
        region_id="test_region",
        group="flight_corridor",
        target_class="water",
        case_type="negative",
        longitude=-111.0,
        latitude=36.7,
        discovery_source="nhd",
    )
    write_candidate_manifest(path, [candidate])
    assert load_candidate_seeds(path) == [candidate]
    assert yaml.safe_load(path.read_text())["candidate_count"] == 1


def test_spaced_selection_is_seeded_and_preserves_region() -> None:
    candidates = [
        (-111.0 + index * 0.1, 36.7, [f"id-{index}"], "source", f"region_{index}")
        for index in range(6)
    ]
    first = _spaced(
        candidates,
        3,
        minimum_m=1000,
        generator=np.random.default_rng(2026),
    )
    second = _spaced(
        candidates,
        3,
        minimum_m=1000,
        generator=np.random.default_rng(2026),
    )
    assert first == second
    assert len(first) == 3
    assert all(row[4].startswith("region_") for row in first)


def test_local_naip_mosaic_and_class_alignment(tmp_path: Path) -> None:
    grid = _Grid(-111.0, 36.7, resolution_m=2.0, half_extent_m=20.0)
    source_transform = from_origin(grid.left, grid.top, 2.0, 2.0)
    rgb_path = tmp_path / "source.tif"
    class_path = tmp_path / "classes.tif"
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "crs": grid.crs,
        "transform": source_transform,
        "tiled": True,
        "blockxsize": 16,
        "blockysize": 16,
    }
    rgb = np.stack(
        [
            np.full((grid.height, grid.width), 40, dtype=np.uint8),
            np.full((grid.height, grid.width), 80, dtype=np.uint8),
            np.full((grid.height, grid.width), 120, dtype=np.uint8),
        ]
    )
    with rasterio.open(rgb_path, "w", count=3, dtype="uint8", **profile) as output:
        output.write(rgb)
    classes = np.zeros((grid.height, grid.width), dtype=np.uint8)
    classes[:, grid.width // 2 :] = 82
    with rasterio.open(class_path, "w", count=1, dtype="uint8", **profile) as output:
        output.write(classes, 1)

    mosaic = tmp_path / "mosaic.tif"
    _materialize_naip(
        [{"id": "local", "assets": {"image": {"href": str(rgb_path)}}}],
        grid,
        mosaic,
    )
    with rasterio.open(mosaic) as source:
        assert source.count == 3
        assert source.shape == (grid.height, grid.width)
        assert np.allclose(
            source.read()[:, 2:-2, 2:-2].mean(axis=(1, 2)),
            [40, 80, 120],
            atol=2,
        )

    mask = tmp_path / "field.tif"
    _write_class_mask(class_path, grid, mask, {82})
    with rasterio.open(mask) as source:
        values = source.read(1)
    assert not values[:, : grid.width // 2].any()
    assert values[:, grid.width // 2 :].all()
