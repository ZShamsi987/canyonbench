"""Sim-to-real fidelity accounting, extinction reporting, and cost preflight."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from canyonbench.eval.adapters import FixtureAdapter
from canyonbench.exceptions import DataValidationError
from canyonbench.schemas import AdapterConfig, BudgetConfig, ModelConfig
from canyonbench.trace.audit import extinction_band_validation, load_audit
from canyonbench.trace.fidelity import (
    dataset_relief_report,
    relief_displacement_px,
    relief_map,
    terrain_relief_m,
    view_relief_displacement,
)
from canyonbench.trace.geometry import apply_relief_displacement
from canyonbench.trace.pilot import run_price_pilot
from canyonbench.trace.planning import (
    REGISTERED_INPUT_TOKENS_PER_CALL,
    REGISTERED_OUTPUT_TOKENS_PER_CALL,
    project_cost,
)
from canyonbench.trace.reporting import build_extinction_figure, extinction_ladder
from canyonbench.trace.schemas import (
    AuditRecord,
    CameraSpec,
    TraceProtocolConfig,
    TraceRunConfig,
)

ROOT = Path(__file__).parents[1]


def _camera(**overrides: float) -> CameraSpec:
    values: dict[str, float] = {
        "longitude": -111.5,
        "latitude": 36.7,
        "altitude_agl_m": 24000.0,
        "width_px": 1024,
        "height_px": 1024,
    }
    values.update(overrides)
    return CameraSpec.model_validate(values)


def test_relief_displacement_reproduces_the_registered_worked_example() -> None:
    # Section 4.2: 1.5 km of canyon relief at 24 km displaces an edge point by
    # about 6 percent of its radial distance, roughly 31 px in a 1024 px image.
    displacement = relief_displacement_px(radial_px=511.5, relief_m=1500, height_agl_m=24000)
    assert displacement == pytest.approx(32.0, abs=1.0)

    camera = _camera()
    depth = np.zeros((1024, 1024), np.float32)
    depth[:512] = 1000.0
    depth[512:] = 2500.0
    quantified = view_relief_displacement(camera, depth)
    assert quantified.relief_m == pytest.approx(1500, abs=1)
    assert quantified.displacement_ratio == pytest.approx(0.0625, abs=1e-3)
    assert quantified.edge_displacement_px == pytest.approx(32.0, abs=1.0)
    # The corner is further from the principal point, so it always shifts more.
    assert quantified.corner_displacement_px > quantified.edge_displacement_px
    assert quantified.corner_displacement_px == pytest.approx(
        quantified.edge_displacement_px * math.sqrt(2), rel=1e-3
    )
    assert quantified.dem_available and not quantified.injected


def test_relief_displacement_scales_inversely_with_flying_height() -> None:
    depth = np.concatenate(
        [np.zeros((512, 1024), np.float32), np.full((512, 1024), 1500, np.float32)]
    )
    low = view_relief_displacement(_camera(altitude_agl_m=3000.0), depth)
    high = view_relief_displacement(_camera(altitude_agl_m=24000.0), depth)
    assert low.edge_displacement_px == pytest.approx(high.edge_displacement_px * 8, rel=1e-6)
    assert terrain_relief_m(None) == 0.0
    assert view_relief_displacement(_camera(), None).relief_m == 0.0
    with pytest.raises(ValueError, match="positive"):
        relief_displacement_px(radial_px=10, relief_m=100, height_agl_m=0)


def test_relief_injection_keeps_masks_binary_and_aligned() -> None:
    camera = _camera(width_px=128, height_px=128, altitude_agl_m=3000.0)
    generator = np.random.default_rng(7)
    rgb = generator.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    mask = np.zeros((128, 128), np.uint8)
    mask[40:90, 60:66] = 1
    rows = np.arange(128, dtype=np.float32).reshape(-1, 1)
    depth = np.tile(rows * 6.0, (1, 128)).astype(np.float32)
    detector = (mask.astype(np.float32) * 0.9).astype(np.float32)

    warped_rgb, warped_masks, warped_continuous = apply_relief_displacement(
        rgb, {"road": mask}, {"road": detector}, depth, camera
    )
    assert warped_rgb.shape == rgb.shape
    assert set(np.unique(warped_masks["road"]).tolist()) <= {0, 1}
    assert warped_masks["road"].any()
    assert warped_continuous["road"].shape == detector.shape
    # The displaced mask still coincides with the displaced detector signal,
    # because one sampling grid drives every raster.
    assert warped_continuous["road"][warped_masks["road"] > 0].mean() > 0.5

    flat = np.full((128, 128), 1800.0, np.float32)
    identity_x, identity_y = relief_map(camera, flat)
    assert identity_x == pytest.approx(np.tile(np.arange(128, dtype=np.float32), (128, 1)))
    assert identity_y == pytest.approx(
        np.tile(np.arange(128, dtype=np.float32).reshape(-1, 1), (1, 128))
    )
    with pytest.raises(ValueError, match="match the rendered view shape"):
        relief_map(camera, np.zeros((64, 64), np.float32))


def test_extinction_ladder_matches_the_registered_geometry_table() -> None:
    ladder = extinction_ladder(altitudes_agl_m=[3000.0, 8000.0, 16000.0, 21800.0, 24000.0])
    by_altitude = ladder.set_index("altitude_agl_m")
    assert by_altitude.loc[3000.0, "ground_width_m"] == pytest.approx(2184, abs=5)
    assert by_altitude.loc[3000.0, "gsd_m_per_px"] == pytest.approx(2.13, abs=0.01)
    assert by_altitude.loc[3000.0, "road_apparent_width_px"] == pytest.approx(5.6, abs=0.1)
    assert by_altitude.loc[8000.0, "road_apparent_width_px"] == pytest.approx(2.1, abs=0.1)
    assert by_altitude.loc[16000.0, "road_apparent_width_px"] == pytest.approx(1.1, abs=0.1)
    assert by_altitude.loc[21800.0, "road_apparent_width_px"] == pytest.approx(0.8, abs=0.1)
    assert by_altitude.loc[24000.0, "gsd_m_per_px"] == pytest.approx(17.1, abs=0.1)
    assert by_altitude.loc[24000.0, "field_apparent_width_px"] == pytest.approx(47, abs=1)
    # Roads extinguish; water and fields must not, which controls for a model
    # simply becoming more conservative with altitude.
    assert by_altitude.loc[24000.0, "road_sub_pixel"]
    assert not by_altitude.loc[24000.0, "water_sub_pixel"]
    assert not by_altitude.loc[24000.0, "field_sub_pixel"]
    assert not by_altitude.loc[3000.0, "road_sub_pixel"]


def _audit_rows(cases: dict[tuple[str, str], bool]) -> list[AuditRecord]:
    records: list[AuditRecord] = []
    for (site, view), resolvable in cases.items():
        for auditor in ("AUD-A", "AUD-B"):
            records.append(
                AuditRecord(
                    site=site,
                    view=view,
                    auditor=auditor,
                    overlay_aligned=True,
                    feature_resolvable=resolvable,
                    obvious_edit_artifact=False,
                    source_mismatch=False,
                )
            )
    return records


def test_extinction_band_human_validation_closes_the_fourth_criterion(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    index = [
        {
            "site_id": "site_0001",
            "view_id": "view_d24km_nadir",
            "variant": "clean",
            "group": "flight_corridor",
            "target_class": "road",
            "case_type": "extinction",
        },
        {
            "site_id": "site_0002",
            "view_id": "view_d24km_nadir",
            "variant": "clean",
            "group": "flight_corridor",
            "target_class": "road",
            "case_type": "extinction",
        },
        {
            "site_id": "site_0003",
            "view_id": "view_a3km_nadir",
            "variant": "clean",
            "group": "flight_corridor",
            "target_class": "water",
            "case_type": "positive",
        },
    ]
    (dataset / "index.json").write_text(json.dumps(index), encoding="utf-8")
    records = _audit_rows(
        {
            ("site_0001", "view_d24km_nadir"): False,
            ("site_0002", "view_d24km_nadir"): True,
            ("site_0003", "view_a3km_nadir"): True,
            ("site_9999", "view_a3km_nadir"): True,
        }
    )
    report = extinction_band_validation(records, dataset)
    band = report["extinction_band"]
    assert band["audited_views"] == 2
    assert band["human_confirmed_no_trace"] == 1
    assert band["human_confirmation_rate"] == pytest.approx(0.5)
    assert band["human_contradicted_views"] == ["site_0002/view_d24km_nadir"]
    assert band["by_class"]["road"]["views"] == 2
    assert report["resolvable_positive_control"]["human_visible_rate"] == pytest.approx(1.0)
    assert report["unmatched_audit_views"] == ["site_9999/view_a3km_nadir"]
    assert report["by_machine_case_type"]["extinction"]["auditor_agreement_rate"] == 1.0


def _run_config(dataset: Path, tmp_path: Path, *, prices: bool) -> TraceRunConfig:
    return TraceRunConfig(
        dataset_dir=dataset,
        output_dir=tmp_path / "run",
        prompt_file=ROOT / "configs" / "trace_prompts.yaml",
        models=[
            ModelConfig(
                id="paid",
                adapter=AdapterConfig(kind="fixture"),
                metered=True,
                input_per_million_usd=5.0 if prices else None,
                output_per_million_usd=30.0 if prices else None,
            ),
            ModelConfig(id="free", adapter=AdapterConfig(kind="fixture"), metered=False),
        ],
        budget=BudgetConfig(max_requests=1000, max_cost_usd=230.0),
        protocol=TraceProtocolConfig(
            screening_views=8,
            causal_core_views=1,
            prompt_cave_views=1,
            robustness_views=1,
        ),
        tiers=["A"],
        enforce_model_roster=False,
    )


def test_cost_projection_prices_only_metered_models(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "index.json").write_text("[]", encoding="utf-8")
    config = _run_config(dataset, tmp_path, prices=True)
    projection = project_cost(config, {"paid": 10_000, "free": 10_000})
    assert set(projection["by_model"]) == {"paid"}
    expected_per_call = (
        REGISTERED_INPUT_TOKENS_PER_CALL * 5.0 + REGISTERED_OUTPUT_TOKENS_PER_CALL * 30.0
    ) / 1_000_000
    assert projection["by_model"]["paid"]["usd_per_call"] == pytest.approx(expected_per_call)
    assert projection["nominal_usd"] == pytest.approx(expected_per_call * 10_000)
    assert projection["by_model"]["paid"]["token_source"] == "registered_estimate"
    assert projection["configured_cost_cap_usd"] == 230.0

    assert projection["nominal_fits_cost_cap"]

    # Provider image-token accounting is the registered budget risk: a pilot that
    # measures twice the assumed tokens must double the projection.
    measured = project_cost(
        config,
        {"paid": 10_000},
        observed_tokens={"paid": {"input": 2800.0, "output": 160.0}},
    )
    assert measured["by_model"]["paid"]["token_source"] == "price_pilot"
    assert measured["nominal_usd"] == pytest.approx(projection["nominal_usd"] * 2)
    assert measured["worst_case_usd_with_parse_retries"] == pytest.approx(
        measured["nominal_usd"] * (config.protocol.parse_retries + 1)
    )
    # A projection above the registered cap must not silently authorize a run.
    over_cap = project_cost(config, {"paid": 50_000})
    assert over_cap["nominal_usd"] > config.budget.max_cost_usd
    assert not over_cap["nominal_fits_cost_cap"]
    assert not over_cap["worst_case_fits_cost_cap"]


def test_price_pilot_measures_tokens_and_reprices_the_plan(tmp_path) -> None:
    from test_trace_integration import make_site

    from canyonbench.trace.config import load_prompts
    from canyonbench.trace.render import build_dataset

    site, project = make_site(tmp_path)
    dataset = build_dataset(project, [site], enforce_quota=False)
    responses = {
        prompt.user.format(target_class="road", grid_size=6, cell_budget=6): json.dumps(
            {
                "answer": "yes",
                "confidence": 80,
                "evidence_cells": ["2,2"],
                "cell_ranking": ["2,2"],
            }
        )
        for prompt in load_prompts(ROOT / "configs" / "trace_prompts.yaml")
    }
    config = _run_config(dataset, tmp_path, prices=True)
    adapters = {"paid": FixtureAdapter(responses), "free": FixtureAdapter(responses)}
    report = run_price_pilot(
        config,
        calls_per_model=4,
        output_dir=tmp_path / "pilot",
        adapters=adapters,
    )
    assert report["calls_per_model"] == 4
    assert report["by_model"]["paid"]["calls"] == 4
    assert report["by_model"]["paid"]["format_failures"] == 0
    assert set(report["by_model"]) == {"paid"}
    assert (tmp_path / "pilot" / "price_pilot.json").exists()
    assert len((tmp_path / "pilot" / "pilot_predictions.jsonl").read_text().splitlines()) == 4
    assert report["registered_projection_usd"]["by_model"]["paid"]["calls"] > 0
    assert "nominal_fits_cost_cap" in report["measured_projection_usd"]

    with pytest.raises(ValueError, match="at least one call"):
        run_price_pilot(config, calls_per_model=0, adapters=adapters)
    unpriced = config.model_copy(
        update={"models": [model for model in config.models if not model.metered]}
    )
    with pytest.raises(DataValidationError, match="No metered models"):
        run_price_pilot(unpriced, calls_per_model=1, adapters=adapters)


def test_generated_bundles_carry_and_aggregate_the_relief_gap(tmp_path) -> None:
    from test_trace_integration import make_site, write_raster

    from canyonbench.io import read_json, sha256_file
    from canyonbench.trace.render import build_dataset

    site, project = make_site(tmp_path)
    with __import__("rasterio").open(site.imagery_path) as source:
        transform = source.transform
        shape = (source.height, source.width)
    rows = np.arange(shape[0], dtype=np.float32).reshape(-1, 1)
    dem_path = tmp_path / "dem.tif"
    write_raster(dem_path, np.tile(rows * 8.0, (1, shape[1])), transform)
    manifest = read_json(site.source_manifest_path)
    manifest["terrain"] = {
        "source_id": "terrain",
        "provider": "synthetic-terrain",
        "product": "aligned-terrain",
        "version": "1",
        "acquisition_date": "2025-06-01",
        "access_date": "2026-07-29",
        "native_resolution_m": 100,
        "url": "https://example.invalid/source",
        "sha256": sha256_file(dem_path),
        "license": "test-only",
        "terms_url": "https://example.invalid/terms",
        "attribution": "Synthetic test fixture",
        "redistribution": "allowed",
        "tile_ids": ["tile-a"],
        "feature_ids": ["terrain"],
    }
    site.source_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    dataset = build_dataset(
        project,
        [site.model_copy(update={"dem_path": dem_path})],
        enforce_quota=False,
    )

    view_manifest = read_json(dataset / "site_0001/view_a3km_nadir/view_manifest.json")
    relief = view_manifest["relief_displacement"]
    assert relief["dem_available"] and not relief["injected"]
    assert relief["relief_m"] > 0
    assert relief["edge_displacement_px"] == pytest.approx(
        relief["relief_m"] * (project.dataset.width_px - 1) / 2 / 3000.0, rel=1e-6
    )

    report = dataset_relief_report(dataset, tmp_path / "relief.json")
    assert report["views_with_dem"] == 8
    assert report["views_with_injection"] == 0
    assert report["overall"]["maximum_edge_displacement_px"] > 0
    by_altitude = report["by_altitude_agl_m"]
    assert set(by_altitude) == {"3000", "8000", "16000", "24000"}
    # Under a constant terrain gradient the relief inside the footprint grows with
    # the footprint, so d = r*relief/H is scale invariant: a higher camera sees
    # more relief but divides by a proportionally larger flying height. The gap is
    # therefore reported per altitude rather than assumed to shrink with height.
    assert by_altitude["3000"]["median_edge_displacement_px"] == pytest.approx(
        by_altitude["24000"]["median_edge_displacement_px"], rel=0.15
    )
    assert by_altitude["24000"]["median_relief_m"] > by_altitude["3000"]["median_relief_m"]
    assert json.loads((tmp_path / "relief.json").read_text())["formula"].startswith("d = r *")

    figures = build_extinction_figure(dataset, tmp_path / "figures")
    assert all(path.exists() for path in figures)
    ladder_path = tmp_path / "figures" / "extinction_ladder.csv"
    with ladder_path.open(encoding="utf-8", newline="") as handle:
        ladder_rows = list(csv.DictReader(handle))
    assert [row["altitude_agl_m"] for row in ladder_rows] == [
        "3000.0",
        "8000.0",
        "16000.0",
        "24000.0",
    ]
    assert (tmp_path / "figures" / "gsd_extinction.pdf").exists()


def test_written_view_manifests_reach_the_per_view_validators(tmp_path) -> None:
    """A frozen bundle must re-validate, or every per-view check is skipped."""

    from test_trace_integration import make_site

    from canyonbench.io import read_json
    from canyonbench.trace.render import build_dataset
    from canyonbench.trace.schemas import ViewManifest
    from canyonbench.trace.validation import validate_dataset

    site, project = make_site(tmp_path)
    dataset = build_dataset(project, [site], enforce_quota=False)
    manifest_path = dataset / "site_0001/view_a3km_nadir/view_manifest.json"
    raw = read_json(manifest_path)
    # camera.json/view_manifest.json serialize the derived footprint and GSD.
    assert {"ground_width_m", "ground_height_m", "gsd_m_per_px"} <= set(raw["camera"])
    manifest = ViewManifest.model_validate(raw)
    assert manifest.camera.gsd_m_per_px == pytest.approx(raw["camera"]["gsd_m_per_px"])

    codes = {issue.code for issue in validate_dataset(dataset, project=project)}
    assert "SCHEMA_ERROR" not in codes
    assert "RGB_HASH_MISMATCH" not in codes
    assert "UNRESOLVABLE_UNGATED_VIEW" not in codes

    # With the schema check restored, tampering is actually detected.
    image_path = dataset / "site_0001/view_a3km_nadir/rgb.png"
    image_path.write_bytes(image_path.read_bytes() + b"\x00")
    tampered = {issue.code for issue in validate_dataset(dataset, project=project)}
    assert "RGB_HASH_MISMATCH" in tampered

    # A camera whose derived geometry disagrees with its parameters is rejected.
    raw["camera"]["gsd_m_per_px"] = raw["camera"]["gsd_m_per_px"] * 2
    with pytest.raises(ValueError, match="derived fields"):
        ViewManifest.model_validate(raw)


def test_through_going_features_stay_resolvable(tmp_path) -> None:
    """Resolvability uses interior reach, not closest approach to the border."""

    from canyonbench.trace.derived import boundary_distance, derive_feature, interior_distance

    image = np.full((256, 256, 3), 40, np.uint8)
    crossing = np.zeros((256, 256), np.uint8)
    crossing[:, 120:126] = 1
    image[crossing > 0] = 220
    detector = np.where(crossing > 0, 0.9, 0.0).astype(float)
    assert boundary_distance(crossing) == 0.0
    assert interior_distance(crossing) > 100
    derived = derive_feature(
        image,
        crossing,
        case_type="positive",
        minimum_resolvable_width_px=2.0,
        detector_score=detector,
    )
    assert derived.resolvable
    assert not derived.extinction
    assert derived.boundary_distance_px == 0.0
    assert derived.interior_distance_px > 100

    corner = np.zeros((256, 256), np.uint8)
    corner[0:1, 0:200] = 1
    assert interior_distance(corner) == 0.0
    clipped = derive_feature(
        np.where(corner[..., None] > 0, 220, image),
        corner,
        case_type="positive",
        minimum_resolvable_width_px=2.0,
        detector_score=np.where(corner > 0, 0.9, 0.0).astype(float),
    )
    assert not clipped.resolvable


def test_relief_injection_requires_a_dem(tmp_path) -> None:
    from test_trace_integration import make_site

    from canyonbench.trace.render import build_dataset

    site, project = make_site(tmp_path)
    injected = project.model_copy(
        update={"dataset": project.dataset.model_copy(update={"inject_relief_displacement": True})}
    )
    with pytest.raises(DataValidationError, match="requires a DEM"):
        build_dataset(injected, [site], enforce_quota=False)


def test_audit_csv_round_trip_supports_the_band_validation(tmp_path) -> None:
    path = tmp_path / "audit.csv"
    fields = [
        "site",
        "view",
        "auditor",
        "overlay_aligned",
        "feature_resolvable",
        "obvious_edit_artifact",
        "source_mismatch",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for auditor in ("AUD-KUNSH", "AUD-ATHARVA"):
            writer.writerow(
                {
                    "site": "site_0001",
                    "view": "view_d24km_nadir",
                    "auditor": auditor,
                    "overlay_aligned": "yes",
                    "feature_resolvable": "no",
                    "obvious_edit_artifact": "no",
                    "source_mismatch": "no",
                    "notes": "",
                }
            )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "index.json").write_text(
        json.dumps(
            [
                {
                    "site_id": "site_0001",
                    "view_id": "view_d24km_nadir",
                    "variant": "clean",
                    "group": "flight_corridor",
                    "target_class": "road",
                    "case_type": "extinction",
                }
            ]
        ),
        encoding="utf-8",
    )
    report = extinction_band_validation(load_audit(path), dataset)
    assert report["extinction_band"]["human_confirmation_rate"] == pytest.approx(1.0)
