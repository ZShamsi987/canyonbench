from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from canyonbench.eval.adapters import FixtureAdapter
from canyonbench.io import write_jsonl
from canyonbench.schemas import AdapterConfig, BudgetConfig, ModelConfig
from canyonbench.trace.cave import cave_frontier, cave_scores, decide, tune_thresholds
from canyonbench.trace.derived import (
    derive_feature,
    grid_occupancy,
    grid_target_pixel_counts,
)
from canyonbench.trace.gates import evaluate_gates, mask_iou
from canyonbench.trace.interventions import (
    apply_operator,
    match_distractor,
    standardized_differences,
)
from canyonbench.trace.protocol import cell_mask, materialize_self_sequences
from canyonbench.trace.runner import _existing, run_trace
from canyonbench.trace.schemas import (
    CameraSpec,
    CaveThresholds,
    DatasetConfig,
    GateConfig,
    QualityParameters,
    TracePrediction,
    TraceProtocolConfig,
    TraceRequest,
    TraceResponse,
    TraceRunConfig,
)


def response(answer: str, confidence: int, cells: list[str] | None = None) -> TraceResponse:
    selected = cells or []
    return TraceResponse(
        answer=answer,
        confidence=confidence,
        evidence_cells=selected,
        cell_ranking=selected,
    )


def test_camera_geometry_and_frozen_dataset_contract(tmp_path) -> None:
    camera = CameraSpec(longitude=-111, latitude=40, altitude_agl_m=3000)
    assert camera.gsd_m_per_px == pytest.approx(camera.ground_width_m / 1024)
    config = DatasetConfig(
        source_root=tmp_path,
        output_root=tmp_path / "out",
        site_manifest=tmp_path / "sites.yaml",
    )
    assert config.site_count == 120
    assert config.clean_view_count == 960
    with pytest.raises(ValueError, match="lattice"):
        DatasetConfig(
            source_root=tmp_path,
            output_root=tmp_path / "out",
            site_manifest=tmp_path / "sites.yaml",
            altitudes_agl_m=[3000, 8000],
        )


def test_quality_allows_exactly_one_factor() -> None:
    assert QualityParameters(degradation="blur", blur_sigma=1.5).blur_sigma == 1.5
    with pytest.raises(ValueError, match="exactly the named"):
        QualityParameters(degradation="blur", blur_sigma=1.5, jpeg_quality=50)
    with pytest.raises(ValueError, match="degradation is none"):
        QualityParameters(blur_sigma=1.5)


def test_derived_geometry_and_grid() -> None:
    image = np.full((64, 64, 3), 30, np.uint8)
    mask = np.zeros((64, 64), np.uint8)
    mask[16:48, 28:36] = 1
    image[mask > 0] = 220
    derived = derive_feature(image, mask, case_type="positive")
    assert derived.present
    assert derived.area_px == 256
    assert derived.median_width_px > 1
    assert derived.local_contrast > 0.5
    assert grid_occupancy(mask, 4) == ["1,1", "1,2", "2,1", "2,2"]
    assert sum(grid_target_pixel_counts(mask, 4).values()) == derived.area_px
    assert sum(derived.grid_target_pixel_counts["6x6"].values()) == derived.area_px
    assert mask_iou(mask, mask) == 1
    intended_extinction = derive_feature(image, mask, case_type="extinction")
    assert intended_extinction.case_type == "positive"
    narrow = np.zeros_like(mask)
    narrow[12:52, 32] = 1
    empirical_extinction = derive_feature(
        np.full_like(image, 30),
        narrow,
        case_type="extinction",
        minimum_resolvable_width_px=3,
        extinction_width_px=3,
        minimum_local_contrast=0.1,
        detector_score=np.zeros(narrow.shape),
    )
    assert empirical_extinction.extinction
    assert empirical_extinction.case_type == "extinction"


def test_cdl_can_supply_authoritative_field_consensus() -> None:
    """The crop-specific federal source is allowed by the registered G2 rule."""

    image = np.full((64, 64, 3), 20, np.uint8)
    primary = np.zeros((64, 64), np.uint8)
    primary[16:48, 20:44] = 1
    image[primary > 0] = 220
    site = SimpleNamespace(
        site_id="site_0001",
        case_type="positive",
        imagery_date="2023-01-01",
        label_date="2023-07-01",
    )

    field = evaluate_gates(
        site,
        "field",
        primary,
        np.zeros_like(primary),
        image,
        config=GateConfig(),
        native_resolution_m=2.0,
        detector_score=np.ones_like(primary),
    )
    road = evaluate_gates(
        site,
        "road",
        primary,
        np.zeros_like(primary),
        image,
        config=GateConfig(),
        native_resolution_m=2.0,
        detector_score=np.ones_like(primary),
    )
    assert field.g2_consensus and field.accepted
    assert not road.g2_consensus and "G2_SOURCE_DISAGREEMENT" in road.reasons


def test_matched_distractor_and_all_operators() -> None:
    generator = np.random.default_rng(4)
    image = generator.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    target = np.zeros((96, 96), np.uint8)
    target[20:70, 42:47] = 1
    distractor, control, differences = match_distractor(
        image, target, depth=None, candidates=100, seed=3
    )
    assert distractor.sum() == target.sum()
    assert not np.logical_and(distractor, target).any()
    assert standardized_differences(control, control) == pytest.approx(
        {name: 0 for name in differences}
    )
    for operator in ("blur", "texture", "frequency", "inpaint"):
        edited = apply_operator(image, target, operator, feather_px=3)
        assert edited.shape == image.shape
        assert edited.dtype == np.uint8


def test_dynamic_self_sequences(tmp_path) -> None:
    from PIL import Image

    image = np.tile(np.arange(96, dtype=np.uint8), (96, 1))
    rgb = np.repeat(image[..., None], 3, axis=2)
    path = tmp_path / "rgb.png"
    Image.fromarray(rgb).save(path)
    rows = materialize_self_sequences(
        path,
        response("yes", 90, ["1,1", "2,2", "3,3"]),
        tmp_path / "dynamic",
        grid_size=6,
        cell_budget=3,
    )
    assert len(rows) == 12
    assert {row["sequence"] for row in rows} == {
        "self_deletion",
        "self_sufficiency",
        "random_control",
        "texture_control",
    }
    assert cell_mask((60, 60), 6, ["1,2"]).sum() == 100


def test_cave_scores_decision_and_dev_tuning() -> None:
    initial = response("yes", 90, ["1,1"])
    necessity = response("no", 80)
    sufficiency = response("yes", 85, ["1,1"])
    nuisance = response("yes", 88, ["2,2"])
    scores = cave_scores(initial, necessity, sufficiency, nuisance)
    assert scores[0] > 0.5
    thresholds = CaveThresholds(
        necessity_min=0.2,
        sufficiency_min=0.8,
        nuisance_max=0.1,
        confidence_min=70,
    )
    decision = decide("a" * 64, initial, necessity, sufficiency, nuisance, thresholds)
    assert decision.accepted
    assert decision.calls_used == 4
    development = [
        (True, initial, necessity, sufficiency, nuisance),
        (
            False,
            response("yes", 80, ["1,1"]),
            response("yes", 80, ["1,1"]),
            response("no", 80),
            response("yes", 20, ["2,2"]),
        ),
        (False, response("no", 95), None, None, None),
    ]
    frontier = cave_frontier(development)
    assert frontier
    assert all(1 <= point.mean_calls_per_initial <= 4 for point in frontier)
    tuned = tune_thresholds(development)
    assert tuned.tuned_split == "development"


def test_resumed_screenings_are_keyed_by_site_and_view(tmp_path) -> None:
    rows = []
    for index in (1, 2):
        request = TraceRequest(
            request_id=f"{index:064x}",
            tier="A",
            sequence="screening",
            model="fixture",
            site_id=f"site_{index:04d}",
            view_id="view_a3km_nadir",
            target_class="road",
            prompt_id="neutral-v4",
            image_path=tmp_path / f"{index}.png",
        )
        prediction = TracePrediction(
            request=request,
            response=response("yes", 80, ["1,1"]),
            raw_response="{}",
            format_failure=False,
            attempts=1,
            latency_s=0,
        )
        rows.append(prediction.model_dump(mode="json"))
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, rows)
    _, _, screenings = _existing(path)
    assert len(screenings) == 2


def test_trace_content_cache_preserves_logical_rows(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    image_path = dataset / "shared.png"
    Image.fromarray(np.full((32, 32, 3), 80, np.uint8)).save(image_path)
    (dataset / "index.json").write_text(
        json.dumps(
            [
                {
                    "site_id": f"site_{index:04d}",
                    "view_id": "view_a3km_nadir",
                    "variant": "clean",
                    "group": "flight_corridor",
                    "target_class": "road",
                    "case_type": "negative",
                    "image_path": "shared.png",
                }
                for index in (1, 2)
            ]
        ),
        encoding="utf-8",
    )
    prompt_path = Path(__file__).parents[1] / "configs" / "trace_prompts.yaml"
    from canyonbench.trace.config import load_prompts

    neutral = next(prompt for prompt in load_prompts(prompt_path) if prompt.variant == "neutral")
    rendered_prompt = neutral.user.format(
        target_class="road",
        grid_size=6,
        cell_budget=6,
    )
    content = json.dumps(
        {
            "answer": "no",
            "confidence": 90,
            "evidence_cells": [],
            "cell_ranking": [],
        }
    )
    model = ModelConfig(
        id="fixture",
        adapter=AdapterConfig(kind="fixture"),
        metered=False,
    )
    config = TraceRunConfig(
        dataset_dir=dataset,
        output_dir=tmp_path / "run",
        prompt_file=prompt_path,
        models=[model],
        budget=BudgetConfig(max_requests=10, max_cost_usd=0),
        protocol=TraceProtocolConfig(
            screening_views=2,
            causal_core_views=0,
            prompt_cave_views=0,
            robustness_views=0,
        ),
        tiers=["A"],
        enforce_model_roster=False,
    )
    predictions_path = run_trace(
        config,
        adapters={"fixture": FixtureAdapter({rendered_prompt: content})},
    )
    predictions = [
        TracePrediction.model_validate(row)
        for row in (
            json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()
        )
    ]
    assert len(predictions) == 2
    assert sum(prediction.cache_hit for prediction in predictions) == 1
    assert sum(prediction.attempts for prediction in predictions) == 1
