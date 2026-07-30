from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as transform_coordinates

from canyonbench.eval.adapters import FixtureAdapter
from canyonbench.schemas import AdapterConfig, BudgetConfig, ModelConfig
from canyonbench.trace.audit import create_audit_sample
from canyonbench.trace.cave_pipeline import (
    apply_from_run,
    component_ablations_from_run,
    tune_from_run,
)
from canyonbench.trace.config import load_prompts
from canyonbench.trace.metrics import score_trace
from canyonbench.trace.release import build_release
from canyonbench.trace.render import build_dataset
from canyonbench.trace.reporting import build_report
from canyonbench.trace.runner import run_trace
from canyonbench.trace.schemas import (
    DatasetConfig,
    InterventionConfig,
    ProjectConfig,
    SiteSpec,
    TraceProtocolConfig,
    TraceRunConfig,
)
from canyonbench.trace.validation import validate_dataset


def write_raster(path: Path, array: np.ndarray, transform: object) -> None:
    if array.ndim == 2:
        array = array[None, ...]
    profile = {
        "driver": "GTiff",
        "height": array.shape[1],
        "width": array.shape[2],
        "count": array.shape[0],
        "dtype": str(array.dtype),
        "crs": "EPSG:32612",
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(array)


def source_record(
    identifier: str,
    sha256: str,
    *,
    provider: str | None = None,
) -> dict[str, object]:
    return {
        "source_id": identifier,
        "provider": provider or f"synthetic-{identifier}",
        "product": f"aligned-{identifier}",
        "version": "1",
        "acquisition_date": "2025-06-01",
        "access_date": "2026-07-29",
        "native_resolution_m": 100,
        "url": "https://example.invalid/source",
        "sha256": sha256,
        "license": "test-only",
        "terms_url": "https://example.invalid/terms",
        "attribution": "Synthetic test fixture",
        "redistribution": "allowed",
        "tile_ids": ["tile-a"],
        "feature_ids": [identifier],
    }


def make_site(tmp_path: Path) -> tuple[SiteSpec, ProjectConfig]:
    from canyonbench.io import sha256_file

    size = 256
    transform = from_origin(400000, 4500000, 100, 100)
    image = np.full((3, size, size), 35, np.uint8)
    masks: dict[str, np.ndarray] = {}
    road = np.zeros((size, size), np.uint8)
    road[35:220, 124:131] = 1
    water = np.zeros_like(road)
    water[80:115, 55:95] = 1
    field = np.zeros_like(road)
    field[145:195, 165:220] = 1
    masks.update(water=water, road=road, field=field)
    colours = {"water": (30, 100, 220), "road": (230, 230, 230), "field": (190, 150, 30)}
    for feature, mask in masks.items():
        for band, value in enumerate(colours[feature]):
            image[band][mask > 0] = value
    imagery = tmp_path / "imagery.tif"
    write_raster(imagery, image, transform)
    primary: dict[str, Path] = {}
    secondary: dict[str, Path] = {}
    for feature, mask in masks.items():
        primary[feature] = tmp_path / f"{feature}_primary.tif"
        secondary[feature] = tmp_path / f"{feature}_secondary.tif"
        write_raster(primary[feature], mask, transform)
        write_raster(secondary[feature], mask, transform)
    centre_x = 400000 + size * 100 / 2
    centre_y = 4500000 - size * 100 / 2
    longitude, latitude = transform_coordinates("EPSG:32612", "EPSG:4326", [centre_x], [centre_y])
    manifest_path = tmp_path / "source_manifest.json"
    imagery_record = source_record("imagery", sha256_file(imagery))
    feature_sources = {
        feature: [
            source_record(f"{feature}-primary", sha256_file(primary[feature])),
            source_record(f"{feature}-secondary", sha256_file(secondary[feature])),
        ]
        for feature in masks
    }
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "4.0.0",
                "site_id": "site_0001",
                "imagery": imagery_record,
                "feature_sources": feature_sources,
            }
        ),
        encoding="utf-8",
    )
    site = SiteSpec(
        site_id="site_0001",
        group="flight_corridor",
        target_class="road",
        case_type="positive",
        longitude=longitude[0],
        latitude=latitude[0],
        imagery_path=imagery,
        primary_mask_paths=primary,
        secondary_mask_paths=secondary,
        source_manifest_path=manifest_path,
        source_tile_ids=["tile-a"],
        feature_ids=["road-a"],
        imagery_date="2025-06-01",
        label_date="2025-06-01",
        native_resolution_m=100,
        split="development",
    )
    project = ProjectConfig(
        dataset=DatasetConfig(
            source_root=tmp_path,
            output_root=tmp_path / "generated",
            site_manifest=tmp_path / "sites.yaml",
            width_px=128,
            height_px=128,
            degraded_view_count=0,
        ),
        interventions=InterventionConfig(
            random_candidates=32,
            maximum_match_smd=5,
            feather_px=2,
        ),
        protocol=TraceProtocolConfig(
            screening_views=8,
            causal_core_views=1,
            prompt_cave_views=1,
            robustness_views=1,
        ),
    )
    return site, project


def test_synthetic_build_run_and_score(tmp_path) -> None:
    site, project = make_site(tmp_path)
    dataset = build_dataset(project, [site], enforce_quota=False)
    index = json.loads((dataset / "index.json").read_text())
    assert len(index) == 8
    assert all(row["variant"] == "clean" for row in index)
    assert (dataset / "site_0001/view_a3km_nadir/interventions/manifest.json").exists()
    issue_codes = {issue.code for issue in validate_dataset(dataset, require_interventions=True)}
    assert "MISSING_ARTIFACT" not in issue_codes
    assert "PIXEL_MISALIGNMENT" not in issue_codes
    assert "INCOMPLETE_INTERVENTIONS" not in issue_codes
    audit_path = create_audit_sample(dataset, tmp_path / "audit.csv", fraction=0.1)
    audit_manifest = json.loads(audit_path.with_suffix(".manifest.json").read_text())
    review_sheet = audit_manifest["review_assets"][0]["review_sheet"]
    assert review_sheet is not None
    assert (audit_path.parent / review_sheet).exists()

    prompts_path = Path(__file__).parents[1] / "configs" / "trace_prompts.yaml"
    fixture_content = json.dumps(
        {
            "answer": "yes",
            "confidence": 90,
            "evidence_cells": ["2,2"],
            "cell_ranking": ["2,2"],
        }
    )
    responses = {}
    for prompt in load_prompts(prompts_path):
        responses[prompt.user.format(target_class="road", grid_size=6, cell_budget=6)] = (
            fixture_content
        )
    model = ModelConfig(id="fixture", adapter=AdapterConfig(kind="fixture"), metered=False)
    run_config = TraceRunConfig(
        dataset_dir=dataset,
        output_dir=tmp_path / "run",
        prompt_file=prompts_path,
        models=[model],
        budget=BudgetConfig(max_requests=100, max_cost_usd=0),
        protocol=TraceProtocolConfig(
            screening_views=8,
            causal_core_views=1,
            prompt_cave_views=1,
            robustness_views=1,
        ),
        tiers=["A"],
        enforce_model_roster=False,
    )
    predictions = run_trace(
        run_config,
        adapters={"fixture": FixtureAdapter(responses)},
    )
    assert len(predictions.read_text().splitlines()) == 8
    result = score_trace(
        dataset,
        predictions,
        tmp_path / "metrics.json",
        bootstrap_iterations=0,
    )
    assert result["query_count"] == 8
    assert result["models"]["fixture"]["performance"]["false_negative_rate"] == 0
    assert "mask_weighted_iou" in result["models"]["fixture"]["localization"]

    full_config = run_config.model_copy(
        update={"tiers": ["A", "B", "C"], "analyses": ["main", "sensitivity"]}
    )
    predictions = run_trace(
        full_config,
        adapters={"fixture": FixtureAdapter(responses)},
    )
    assert len(predictions.read_text().splitlines()) > 50
    thresholds = tune_from_run(dataset, predictions, tmp_path / "thresholds.json")
    frontier_path = tmp_path / "thresholds.frontier.json"
    frontier = json.loads(frontier_path.read_text())
    assert frontier["points"]
    decisions_path = tmp_path / "cave_decisions.jsonl"
    decisions = apply_from_run(
        dataset,
        predictions,
        thresholds,
        decisions_path,
    )
    assert decisions
    ablations_path = tmp_path / "cave_ablations.jsonl"
    ablations = component_ablations_from_run(
        dataset,
        predictions,
        thresholds,
        ablations_path,
    )
    assert {record.variant for record in ablations} == {
        "full",
        "necessity_only",
        "sufficiency_only",
        "nuisance_only",
    }
    result = score_trace(
        dataset,
        predictions,
        tmp_path / "full_metrics.json",
        bootstrap_iterations=0,
        cave_decisions=decisions_path,
        cave_ablations=ablations_path,
        cave_frontier_path=frontier_path,
    )
    assert result["cave"]["fixture"]["calls_per_initial"] == 4
    assert len(result["cave"]["fixture"]["by_prompt"]) == 2
    assert "false_premise_compliance_delta" in result["cave"]["fixture"]
    assert result["cave_frontier"]["points"]
    assert "full" in result["cave_component_ablations"]["fixture"]
    report_paths = build_report(
        tmp_path / "full_metrics.json",
        tmp_path / "full_metrics.rows.csv",
        tmp_path / "report",
    )
    assert all(path.exists() for path in report_paths)
    public, escrow = build_release(dataset, tmp_path / "release")
    release_manifest = json.loads((public / "release_manifest.json").read_text())
    assert release_manifest["files"]
    assert release_manifest["input_index_sha256"]
    assert (escrow / "held_out_hashes.json").exists()
