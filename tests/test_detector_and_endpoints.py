"""Non-language detector reference, endpoint overrides, and the credit budget."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from canyonbench.compute import (
    INSTANCES,
    LAMBDA_GPU_ALLOCATION_USD,
    LAMBDA_STORAGE_RESERVE_USD,
    PROJECTED_GPU_HOURS,
    gpu_budget,
)
from canyonbench.detector import (
    DetectorQuery,
    parse_detector_prompt,
    resolve_label_ids,
    response_from_segmentation,
)
from canyonbench.trace.config import (
    apply_endpoint_overrides,
    endpoint_env_var,
    load_run_config_for_host,
    load_trace_run_config,
)
from canyonbench.trace.schemas import TraceResponse

ROOT = Path(__file__).parents[1]

# A stand-in for the ADE20K label set, with the semicolon-joined synonyms the
# real checkpoint uses.
ADE20K_SAMPLE = {
    0: "wall",
    6: "road;route",
    9: "grass",
    13: "earth;ground",
    21: "water",
    26: "sea",
    29: "field",
    52: "path",
    60: "river",
    128: "lake",
}


def test_prompt_parsing_recovers_class_grid_and_budget() -> None:
    prompts = [prompt for prompt in _rendered_prompts() if prompt]
    assert prompts, "the frozen prompt file must render"
    for prompt in prompts:
        query = parse_detector_prompt(prompt)
        assert query.target_class == "road"
        assert query.grid_size == 6
        assert query.cell_budget == 6

    other = parse_detector_prompt("Is water visibly present? Use a 8x8 grid, at most 3 cells.")
    assert other == DetectorQuery(target_class="water", grid_size=8, cell_budget=3)

    with pytest.raises(ValueError, match="exactly one"):
        parse_detector_prompt("Is anything present?")
    with pytest.raises(ValueError, match="exactly one"):
        parse_detector_prompt("Is water or road present?")


def _rendered_prompts() -> list[str]:
    from canyonbench.trace.config import load_prompts

    return [
        prompt.user.format(target_class="road", grid_size=6, cell_budget=6)
        for prompt in load_prompts(ROOT / "configs" / "trace_prompts.yaml")
        if prompt.variant != "no_image"
    ]


def test_label_resolution_uses_the_checkpoint_not_hardcoded_indices() -> None:
    assert resolve_label_ids(ADE20K_SAMPLE, "water") == (21, 26, 60, 128)
    assert resolve_label_ids(ADE20K_SAMPLE, "road") == (6, 52)
    assert resolve_label_ids(ADE20K_SAMPLE, "field") == (9, 29)
    # Bare ground must not count as a cultivated field, or every desert view
    # would trigger the reference.
    assert 13 not in resolve_label_ids(ADE20K_SAMPLE, "field")

    with pytest.raises(KeyError):
        resolve_label_ids(ADE20K_SAMPLE, "building")
    with pytest.raises(ValueError, match="no label"):
        resolve_label_ids({0: "wall", 1: "sky"}, "water")


def test_segmentation_becomes_a_schema_valid_answer() -> None:
    query = DetectorQuery(target_class="road", grid_size=6, cell_budget=6)
    segmentation = np.zeros((120, 120), np.int32)
    segmentation[:, 58:64] = 6  # a road crossing the view

    answer = response_from_segmentation(segmentation, (6, 52), query)
    parsed = TraceResponse.model_validate(answer)
    parsed.validate_protocol(grid_size=6, cell_budget=6)
    assert parsed.answer == "yes"
    assert parsed.evidence_cells, "a detected feature must carry evidence cells"
    # The road runs down the middle column band, so every named cell is in it.
    assert {cell.split(",")[1] for cell in parsed.evidence_cells} <= {"2", "3"}
    assert parsed.cell_ranking == parsed.evidence_cells

    empty = response_from_segmentation(np.zeros((120, 120), np.int32), (6, 52), query)
    parsed_empty = TraceResponse.model_validate(empty)
    parsed_empty.validate_protocol(grid_size=6, cell_budget=6)
    assert parsed_empty.answer == "no"
    assert parsed_empty.evidence_cells == []
    assert parsed_empty.confidence == 100

    # A trace of noise below the presence floor stays a negative.
    speck = np.zeros((120, 120), np.int32)
    speck[0, 0] = 6
    assert response_from_segmentation(speck, (6, 52), query)["answer"] == "no"

    with pytest.raises(ValueError, match="2-D"):
        response_from_segmentation(np.zeros((4, 4, 3), np.int32), (6,), query)


def test_detector_respects_the_cell_budget_and_grid() -> None:
    segmentation = np.full((80, 80), 21, np.int32)  # water everywhere
    for grid_size, budget in ((4, 3), (6, 6), (8, 10)):
        query = DetectorQuery(target_class="water", grid_size=grid_size, cell_budget=budget)
        answer = response_from_segmentation(segmentation, (21,), query)
        parsed = TraceResponse.model_validate(answer)
        parsed.validate_protocol(grid_size=grid_size, cell_budget=budget)
        assert len(parsed.evidence_cells) == min(budget, grid_size * grid_size)
        assert parsed.answer == "yes"


def test_endpoint_overrides_redirect_only_the_named_model(monkeypatch) -> None:
    config = load_trace_run_config(ROOT / "configs" / "trace_run.frozen.yaml")
    detector = "canyonbench-independent-detector-v1"
    variable = endpoint_env_var(detector)
    assert variable == "CANYONBENCH_ENDPOINT__CANYONBENCH_INDEPENDENT_DETECTOR_V1"

    monkeypatch.setenv(variable, "http://10.0.0.4:9100")
    updated, applied = apply_endpoint_overrides(config)
    assert applied == {detector: "http://10.0.0.4:9100"}
    by_id = {model.id: model for model in updated.models}
    assert by_id[detector].adapter.base_url == "http://10.0.0.4:9100"
    # Every other model keeps the frozen address.
    for model in config.models:
        if model.id != detector:
            assert by_id[model.id].adapter.base_url == model.adapter.base_url

    monkeypatch.delenv(variable)
    unchanged, applied = apply_endpoint_overrides(config)
    assert applied == {}
    assert unchanged is config


def test_host_loader_applies_overrides_and_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(
        endpoint_env_var("akshaydudhane/EarthDial_4B_RGB"), "http://127.0.0.1:9001/v1"
    )
    config = load_run_config_for_host(
        ROOT / "configs" / "trace_run.frozen.yaml",
        dataset_dir=tmp_path / "dataset",
        output_dir=tmp_path / "results",
    )
    by_id = {model.id: model for model in config.models}
    assert by_id["akshaydudhane/EarthDial_4B_RGB"].adapter.base_url == "http://127.0.0.1:9001/v1"
    assert config.dataset_dir == (tmp_path / "dataset").resolve()
    assert config.output_dir == (tmp_path / "results").resolve()


def test_gpu_credit_budget_covers_the_projection_with_headroom() -> None:
    assert LAMBDA_STORAGE_RESERVE_USD + LAMBDA_GPU_ALLOCATION_USD <= 400.0

    rates = {
        instance.name: instance.usd_per_hour
        for instance in INSTANCES
        if instance.usd_per_hour is not None
    }
    assert rates == {"1x H100 80 GB PCIe": 3.29, "1x H100 80 GB SXM5": 4.29}

    pcie = gpu_budget(usd_per_hour=3.29)
    assert pcie["fits"]
    assert pcie["affordable_hours"] == pytest.approx(340 / 3.29, rel=1e-6)
    assert pcie["projected_cost_usd"] == pytest.approx(15 * 3.29, rel=1e-6)
    # Even the pricier card leaves several times the projected requirement.
    sxm5 = gpu_budget(usd_per_hour=4.29)
    assert sxm5["fits"]
    assert sxm5["headroom_multiple"] > 5
    assert sxm5["affordable_hours"] < pcie["affordable_hours"]
    assert PROJECTED_GPU_HOURS == 15.0

    # A run long enough to exhaust the allocation is reported as not fitting.
    exhausted = gpu_budget(usd_per_hour=3.29, projected_hours=200)
    assert not exhausted["fits"]
    with pytest.raises(ValueError, match="positive"):
        gpu_budget(usd_per_hour=0)


def test_h100_is_the_registered_primary_and_serves_every_planned_class() -> None:
    from canyonbench.compute import admissible_instances

    planned = ["vlm_7_8b", "vlm_12_14b", "vlm_26_34b", "detector_or_segmenter"]
    rows = {row["instance"]: row for row in admissible_instances(planned)}
    assert rows["1x H100 80 GB PCIe"]["verdict"] == "primary"
    assert rows["1x H100 80 GB PCIe"]["unserved"] == []
    assert rows["1x H100 80 GB SXM5"]["verdict"] == "secondary"
    # The 40 GB A100 is now a fallback precisely because it cannot hold the 32B.
    assert rows["1x A100 40 GB SXM4"]["verdict"] == "fallback"
    assert "vlm_26_34b" in rows["1x A100 40 GB SXM4"]["unserved"]
