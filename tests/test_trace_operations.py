from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from PIL import Image
from rasterio.transform import from_origin

from canyonbench.eval.adapters import FixtureAdapter
from canyonbench.exceptions import DataValidationError
from canyonbench.schemas import AdapterConfig, BudgetConfig, ModelConfig
from canyonbench.trace.ablations import gate_ablation_report, write_gate_ablation_report
from canyonbench.trace.audit import create_audit_sample, load_audit, summarize_audit
from canyonbench.trace.config import (
    load_preregistration,
    load_project_config,
    load_prompts,
    load_sites,
    load_source_manifest,
    load_trace_run_config,
)
from canyonbench.trace.degradation import (
    apply_degradation,
    calibrate_from_frames,
    measure_quality,
    sample_degradations,
)
from canyonbench.trace.instruments import (
    build_synthetic_width_series,
    edit_detectability_audit,
    operator_agreement,
    suppression_efficacy,
    synthetic_road_insert,
)
from canyonbench.trace.metrics import causal_metrics
from canyonbench.trace.positive_control import (
    run_positive_control,
    score_positive_control,
)
from canyonbench.trace.schemas import (
    DatasetConfig,
    GateResult,
    QualityParameters,
    SiteSpec,
    TraceProtocolConfig,
    TraceRunConfig,
)
from canyonbench.trace.selection import select_sites, write_selection
from canyonbench.trace.sources import (
    align_raster_to_reference,
    assert_binary_mask,
    rasterize_geojson,
)
from canyonbench.trace.splits import (
    distance_m,
    independence_issues,
    leakage_issues,
)
from canyonbench.trace.statistics import (
    benjamini_hochberg,
    fit_mixed_effects,
    hierarchical_bootstrap,
    paired_site_comparison,
)

ROOT = Path(__file__).parents[1]


def _site(
    identifier: int,
    group: str = "flight_corridor",
    feature: str = "road",
    case_type: str = "positive",
) -> SiteSpec:
    row, column = divmod(identifier - 1, 12)
    dummy = Path(f"/tmp/site_{identifier:04d}.tif")
    masks = {"water": dummy, "road": dummy, "field": dummy}
    return SiteSpec(
        site_id=f"site_{identifier:04d}",
        group=group,
        target_class=feature,
        case_type=case_type,
        longitude=-120 + column * 0.5,
        latitude=30 + row * 0.5,
        imagery_path=dummy,
        primary_mask_paths=masks,
        secondary_mask_paths=masks,
        source_manifest_path=dummy.with_suffix(".json"),
        source_tile_ids=[f"tile-{identifier}"],
        feature_ids=[f"feature-{identifier}"],
        imagery_date="2025-01-01",
        label_date="2025-01-01",
        native_resolution_m=1,
    )


def _gate(site: SiteSpec, feature: str) -> GateResult:
    return GateResult(
        site_id=site.site_id,
        feature=feature,
        case_type=site.case_type if feature == site.target_class else "negative",
        g1_time_alignment=True,
        date_gap_days=0,
        maximum_date_gap_days=100,
        g2_consensus=True,
        primary_secondary_iou=1,
        consensus_within_tolerance=1,
        negative_buffer_clear=True,
        g3_resolvable_or_extinction=True,
        minimum_width_px=3,
        median_width_px=4,
        component_count=1,
        boundary_distance_px=4,
        local_contrast=0.2,
        occlusion_fraction=0,
        extinction=False,
        g4_detector_pass=True,
        accepted=True,
    )


def test_configuration_loaders(tmp_path) -> None:
    project = load_project_config(ROOT / "configs" / "trace.yaml")
    assert project.dataset.site_count == 120
    assert len(load_prompts(ROOT / "configs" / "trace_prompts.yaml")) == 5
    assert load_preregistration(ROOT / "configs" / "preregistration.yaml").alpha == 0.05
    run = load_trace_run_config(ROOT / "configs" / "trace_run.example.yaml")
    assert len(run.models) == 8

    site = _site(1)
    site_path = tmp_path / "sites.yaml"
    site_path.write_text(
        yaml.safe_dump({"sites": [site.model_dump(mode="json")]}),
        encoding="utf-8",
    )
    loaded = load_sites(site_path)
    assert loaded[0].imagery_path.is_absolute()
    source_path = tmp_path / "source.json"
    record = {
        "source_id": "x",
        "provider": "x",
        "product": "x",
        "version": "x",
        "acquisition_date": "2025-01-01",
        "access_date": "2026-07-29",
        "native_resolution_m": 1,
        "url": "https://example.invalid",
        "sha256": "0" * 64,
        "license": "test",
        "terms_url": "https://example.invalid/terms",
        "attribution": "Test fixture",
        "redistribution": "allowed",
    }
    source_path.write_text(
        json.dumps(
            {
                "site_id": "site_0001",
                "imagery": record,
                "feature_sources": {
                    feature: [
                        record
                        | {
                            "source_id": f"{feature}-1",
                            "provider": f"{feature}-provider-1",
                            "product": f"{feature}-product-1",
                        },
                        record
                        | {
                            "source_id": f"{feature}-2",
                            "provider": f"{feature}-provider-2",
                            "product": f"{feature}-product-2",
                        },
                    ]
                    for feature in ("water", "road", "field")
                },
            }
        ),
        encoding="utf-8",
    )
    assert load_source_manifest(source_path).site_id == "site_0001"
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(DataValidationError, match="empty"):
        load_sites(empty)


def test_quality_calibration_and_all_degradations(tmp_path) -> None:
    paths = []
    for index in range(6):
        gradient = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
        image = np.stack(
            [gradient, np.roll(gradient, index * 3, axis=1), 255 - gradient],
            axis=2,
        )
        path = tmp_path / f"frame-{index}.png"
        Image.fromarray(image).save(path)
        paths.append(path)
    calibration = calibrate_from_frames(paths)
    assert calibration.frame_count == 6
    image = np.asarray(Image.open(paths[0]).convert("RGB"))
    assert measure_quality(image).contrast_std > 0
    qualities = [
        QualityParameters(),
        QualityParameters(degradation="blur", blur_sigma=1),
        QualityParameters(degradation="haze", haze_strength=0.2),
        QualityParameters(degradation="exposure", exposure_ev=-0.5),
        QualityParameters(degradation="saturation", saturation_scale=0.5),
        QualityParameters(degradation="contrast", contrast_scale=0.5),
        QualityParameters(degradation="jpeg", jpeg_quality=40),
    ]
    for quality in qualities:
        assert apply_degradation(image, quality).shape == image.shape
    sampled = sample_degradations(
        [f"view-{index}" for index in range(12)],
        count=6,
        seed=2,
        calibration=calibration,
        calibration_source_sha256="1" * 64,
    )
    assert sum(value.degradation != "none" for value in sampled.values()) == 6
    with pytest.raises(ValueError, match="exceed"):
        sample_degradations(["one"], count=2, seed=1)


def _write_raster(path: Path, array: np.ndarray, transform: Any) -> None:
    if array.ndim == 2:
        array = array[None, ...]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=array.shape[0],
        dtype=str(array.dtype),
        crs="EPSG:32612",
        transform=transform,
    ) as destination:
        destination.write(array)


def test_source_alignment_and_rasterization(tmp_path) -> None:
    reference = tmp_path / "reference.tif"
    source = tmp_path / "source.tif"
    _write_raster(reference, np.zeros((32, 32), np.uint8), from_origin(0, 320, 10, 10))
    _write_raster(source, np.ones((16, 16), np.uint8), from_origin(0, 320, 20, 20))
    aligned = align_raster_to_reference(source, reference, tmp_path / "aligned.tif")
    with rasterio.open(aligned) as dataset:
        assert dataset.shape == (32, 32)
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[20, 20], [20, 100], [100, 100], [100, 20], [20, 20]]],
                },
            }
        ],
    }
    mask = rasterize_geojson(geojson, reference, tmp_path / "mask.tif")
    assert_binary_mask(mask)
    invalid = tmp_path / "invalid.tif"
    _write_raster(invalid, np.full((32, 32), 3, np.uint8), from_origin(0, 320, 10, 10))
    with pytest.raises(DataValidationError, match="not binary"):
        assert_binary_mask(invalid)


def test_audit_sampling_loading_and_summary(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    index = [
        {
            "site_id": f"site_{index:04d}",
            "view_id": "view_a3km_nadir",
            "variant": "clean",
            "group": "flight_corridor",
            "target_class": "road",
            "case_type": "positive",
        }
        for index in range(1, 11)
    ]
    (dataset / "index.json").write_text(json.dumps(index), encoding="utf-8")
    path = create_audit_sample(dataset, tmp_path / "audit.csv", fraction=0.1)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(
            overlay_aligned="yes",
            feature_resolvable="yes",
            obvious_edit_artifact="no",
            source_mismatch="no",
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    records = load_audit(path)
    summary = summarize_audit(records)
    assert summary["agreement"]["overlay_aligned"] == 1
    duplicate_auditors = [
        records[0],
        records[1].model_copy(update={"auditor": records[0].auditor}),
    ]
    with pytest.raises(DataValidationError, match="distinct auditor"):
        summarize_audit(duplicate_auditors)
    rows[0]["overlay_aligned"] = "maybe"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(DataValidationError, match="yes/no"):
        load_audit(path)


def test_instrument_validation_suite(tmp_path) -> None:
    base = np.full((64, 64, 3), 80, np.uint8)
    inserted, mask = synthetic_road_insert(base, apparent_width_px=2)
    assert inserted.shape == base.shape and mask.any()
    source = tmp_path / "source.png"
    Image.fromarray(base).save(source)
    assert len(build_synthetic_width_series(source, tmp_path / "widths", [1, 2])) == 2

    rows = []
    groups = []
    for index in range(12):
        label = ("untouched", "target", "distractor")[index % 3]
        value = 40 if label == "untouched" else 140 if label == "target" else 220
        path = tmp_path / f"audit-{index}.png"
        Image.fromarray(np.full((32, 32, 3), value, np.uint8)).save(path)
        rows.append({"image_path": str(path), "label": label})
        groups.append(f"site-{index // 2}")
    audit = edit_detectability_audit(rows, groups=groups)
    assert audit["n_images"] == 12
    operator_frame = pd.DataFrame(
        [
            {"model": model, "operator": operator, "metric": value + offset}
            for value, model in enumerate(("a", "b", "c", "d"))
            for offset, operator in enumerate(("blur", "texture", "frequency"))
        ]
    )
    assert len(operator_agreement(operator_frame, metric="metric")) == 3
    efficacy = suppression_efficacy(
        np.array([0.1, 0.8, 0.2, 0.9]),
        np.array([0.1, 0.4, 0.1, 0.5]),
        np.array([0, 1, 0, 1]),
    )
    assert efficacy["mean_signal_removed"] > 0


def test_v1_positive_control_runner_is_strict_and_resumable(tmp_path) -> None:
    source = tmp_path / "negative.png"
    Image.fromarray(np.full((64, 64, 3), 80, np.uint8)).save(source)
    prompts = ROOT / "configs" / "trace_prompts.yaml"
    prompt = load_prompts(prompts)[0].user.format(
        target_class="road",
        grid_size=6,
        cell_budget=6,
    )
    response = json.dumps(
        {
            "answer": "yes",
            "confidence": 80,
            "evidence_cells": ["2,2"],
            "cell_ranking": ["2,2"],
        }
    )
    model = ModelConfig(
        id="fixture",
        adapter=AdapterConfig(kind="fixture"),
        metered=False,
    )
    config = TraceRunConfig(
        dataset_dir=tmp_path,
        output_dir=tmp_path / "unused",
        prompt_file=prompts,
        models=[model],
        budget=BudgetConfig(max_requests=10, max_cost_usd=0),
        protocol=TraceProtocolConfig(
            screening_views=1,
            causal_core_views=1,
            prompt_cave_views=1,
            robustness_views=1,
        ),
        enforce_model_roster=False,
    )
    predictions = run_positive_control(
        config,
        source,
        tmp_path / "v1",
        widths=[0.5, 2],
        adapters={"fixture": FixtureAdapter({prompt: response})},
    )
    assert len(predictions.read_text().splitlines()) == 2
    run_positive_control(
        config,
        source,
        tmp_path / "v1",
        widths=[0.5, 2],
        adapters={"fixture": FixtureAdapter({prompt: response})},
    )
    assert len(predictions.read_text().splitlines()) == 2
    metrics = score_positive_control(predictions, tmp_path / "v1" / "metrics.json")
    assert metrics["models"]["fixture"]["n_widths"] == 2


def test_site_selection_splits_and_gate_ablations(tmp_path, monkeypatch) -> None:
    candidates = []
    identifier = 1
    quotas = {
        "flight_corridor": 20,
        "regional_ood": 12,
        "cross_biome": 8,
    }
    for group, per_class in quotas.items():
        for feature in ("water", "road", "field"):
            for case_type in ("positive", "negative"):
                for _ in range(per_class // 2):
                    candidates.append(_site(identifier, group, feature, case_type))
                    identifier += 1

    def fake_evaluate(site: SiteSpec, _config: DatasetConfig) -> list[GateResult]:
        return [_gate(site, feature) for feature in ("water", "road", "field")]

    monkeypatch.setattr("canyonbench.trace.selection.evaluate_site", fake_evaluate)
    monkeypatch.setattr("canyonbench.trace.ablations.evaluate_site", fake_evaluate)
    config = DatasetConfig(
        source_root=tmp_path,
        output_root=tmp_path,
        site_manifest=tmp_path / "sites.yaml",
        degraded_view_count=0,
    )
    selected, gates = select_sites(candidates, config, attempts=2)
    assert len(selected) == 120 and len(gates) == 360
    assert not independence_issues(selected, minimum_site_separation_m=30000)
    assert not leakage_issues(selected)
    assert distance_m(selected[0], selected[1]) > 30000
    write_selection(tmp_path / "selected.yaml", selected, gates)
    assert (tmp_path / "selected.gates.json").exists()
    report = gate_ablation_report(selected[:2], config)
    assert set(report) == {
        "registered_consensus_strict_road2_detector",
        "single_source",
        "relaxed_dates",
        "road_3px",
        "without_exclusion_detector",
    }
    write_gate_ablation_report(selected[:2], config, tmp_path / "ablation.json")


def test_site_aware_statistics() -> None:
    rows = []
    for site in range(12):
        for condition in ("first", "second"):
            rows.append(
                {
                    "site_id": f"s{site}",
                    "group": f"g{site % 3}",
                    "target_class": ("water", "road", "field")[site % 3],
                    "condition": condition,
                    "value": site / 12 + ((0.08 + site * 0.001) if condition == "first" else 0),
                    "outcome": int((site + (condition == "first")) % 2 == 0),
                    "log_gsd": np.log(1 + site),
                    "apparent_width_px": 1 + site,
                    "feature_area_fraction": 0.01 * (site + 1),
                    "oblique": int(condition == "second"),
                }
            )
    frame = pd.DataFrame(rows)
    interval = hierarchical_bootstrap(
        frame,
        lambda values: float(values["value"].mean()),
        iterations=30,
        seed=1,
    )
    assert interval["independent_sites"] == 12
    comparison = paired_site_comparison(
        frame,
        value="value",
        condition="condition",
        first="first",
        second="second",
    )
    assert comparison["mean_difference"] > 0.08
    adjusted = benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.2})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    mixed = fit_mixed_effects(frame, outcome="outcome")
    assert mixed["n_observations"] == len(frame)


def test_causal_metrics_anchor_to_clean_view_and_normalize_sufficiency() -> None:
    rows = [
        {
            "tier": "A",
            "sequence": "screening",
            "variant": "clean",
            "site_id": "site_0001",
            "view_id": "view_a3km_nadir",
            "target_class": "road",
            "fraction": 0.0,
            "yes_probability": 1.0,
        }
    ]
    curves = {
        "oracle_deletion": [(0.5, 0.4), (1.0, 0.0)],
        "distractor_deletion": [(0.5, 0.9), (1.0, 0.8)],
        "self_deletion": [(0.5, 0.5), (1.0, 0.2)],
        "self_sufficiency": [(0.5, 0.9), (1.0, 1.0)],
        "random_control": [(0.5, 0.9), (1.0, 0.7)],
        "texture_control": [(0.5, 0.8), (1.0, 0.6)],
    }
    for sequence, values in curves.items():
        for fraction, probability in values:
            rows.append(
                {
                    "tier": "B",
                    "sequence": sequence,
                    "variant": "clean",
                    "site_id": "site_0001",
                    "view_id": "view_a3km_nadir",
                    "target_class": "road",
                    "fraction": fraction,
                    "yes_probability": probability,
                }
            )
    metrics = causal_metrics(pd.DataFrame(rows))
    assert metrics["auc_oracle_deletion"] == pytest.approx(0.45)
    assert metrics["ocrs"] == pytest.approx(0.45)
    assert metrics["sen"] == pytest.approx(0.325)
    assert metrics["ses"] == pytest.approx(0.95)
