"""Registered compute contract, serving-profile selection, and run merging."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from canyonbench.compute import (
    INSTANCES,
    LAMBDA_TOTAL_ESTIMATE_GB,
    MODEL_CLASSES,
    REQUIRED_DTYPE,
    StorageLayout,
    admissible_instances,
    compute_check,
    fits,
    model_class,
    serving_profile,
)
from canyonbench.exceptions import DataValidationError
from canyonbench.trace.merge import merge_predictions

ROOT = Path(__file__).parents[1]


def test_serving_profile_matches_the_registered_vram_bands() -> None:
    small = serving_profile(24)
    medium = serving_profile(40)
    large = serving_profile(80)
    assert (small.max_model_len, small.max_num_seqs) == (8192, 16)
    assert (medium.max_model_len, medium.max_num_seqs) == (16384, 64)
    assert (large.max_model_len, large.max_num_seqs) == (32768, 128)
    # Band edges: 30 and 60 GB are the registered thresholds.
    assert serving_profile(29.9).max_num_seqs == 16
    assert serving_profile(30).max_num_seqs == 64
    assert serving_profile(59.9).max_num_seqs == 64
    assert serving_profile(60).max_num_seqs == 128
    with pytest.raises(ValueError, match="positive"):
        serving_profile(0)


def test_every_profile_pins_bfloat16_and_single_image_prompts() -> None:
    for vram in (24, 40, 80):
        profile = serving_profile(vram)
        kwargs = profile.as_vllm_kwargs(download_dir="/lambda/canyonbench/hf")
        assert kwargs["dtype"] == REQUIRED_DTYPE == "bfloat16"
        assert kwargs["limit_mm_per_prompt"] == {"image": 1}
        assert kwargs["gpu_memory_utilization"] == 0.90
        assert kwargs["download_dir"] == "/lambda/canyonbench/hf"
        args = profile.as_server_args()
        assert "--dtype" in args and args[args.index("--dtype") + 1] == "bfloat16"
        assert "--max-num-seqs" in args
        assert str(profile.max_model_len) in args


def test_model_class_requirements_and_single_card_fit() -> None:
    assert model_class("vlm_7_8b").minimum_vram_gb == 24
    assert model_class("vlm_26_34b").minimum_vram_gb == 80
    assert not model_class("vlm_70b_plus").required
    with pytest.raises(KeyError):
        model_class("vlm_missing")

    # A 26-34B model in bfloat16 does not fit a 40 GB card.
    assert fits("vlm_7_8b", 24)
    assert fits("vlm_7_8b", 40)
    assert not fits("vlm_26_34b", 40)
    assert fits("vlm_26_34b", 80)
    # Extra devices only help a model that exceeds one card.
    assert not fits("vlm_70b_plus", 80, gpus=1)
    assert fits("vlm_70b_plus", 80, gpus=2)


def test_instance_assessment_matches_the_registered_selection() -> None:
    planned = ["vlm_7_8b", "vlm_12_14b", "vlm_26_34b", "detector_or_segmenter"]
    rows = {row["instance"]: row for row in admissible_instances(planned)}

    a100 = rows["1x A100 40 GB SXM4"]
    assert a100["verdict"] == "primary"
    assert "vlm_26_34b" in a100["unserved"], "40 GB cannot serve a 26-34B model in bf16"
    assert "vlm_12_14b" in a100["serves"]

    h100 = rows["1x H100 80 GB PCIe"]
    assert h100["verdict"] == "secondary"
    assert h100["unserved"] == []

    a10 = rows["1x A10 24 GB PCIe"]
    assert a10["serves"] == ["vlm_7_8b", "detector_or_segmenter"]

    volta = rows["8x Tesla V100 16 GB"]
    assert volta["verdict"] == "excluded"
    assert not volta["bfloat16_capable"]
    assert volta["serves"] == [], "the precision constraint excludes Volta outright"

    # Nothing in the offered set is silently unassessed.
    assert {instance.name for instance in INSTANCES} == set(rows)
    assert {entry.name for entry in MODEL_CLASSES} >= set(planned)


def test_storage_layout_is_the_registered_lambda_tree(tmp_path) -> None:
    layout = StorageLayout(tmp_path / "canyonbench")
    layout.create()
    assert layout.hf_home.is_dir()
    assert layout.dataset.is_dir()
    assert layout.results.is_dir()
    assert layout.logs.is_dir()
    environment = layout.environment()
    assert environment["HF_HOME"] == str(layout.hf_home)
    assert environment["HF_HUB_ENABLE_HF_TRANSFER"] == "1"
    assert LAMBDA_TOTAL_ESTIMATE_GB == 200


def test_compute_check_reports_blocking_failures_per_role(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    report = compute_check(role="openrouter", required_env=("OPENROUTER_API_KEY",))
    assert not report["ready"]
    assert "env:OPENROUTER_API_KEY" in report["blocking_failures"]

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    report = compute_check(role="openrouter", required_env=("OPENROUTER_API_KEY",))
    assert report["ready"]

    # Optional tooling is reported but never blocks.
    assert any(
        check["name"].startswith("tool:") and not check["blocking"] for check in report["checks"]
    )

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    missing = compute_check(role="adroit", dataset_dir=dataset)
    assert not missing["ready"]
    assert "dataset:index" in missing["blocking_failures"]
    (dataset / "index.json").write_text("[]", encoding="utf-8")
    assert compute_check(role="adroit", dataset_dir=dataset)["ready"]

    with pytest.raises(ValueError, match="role must be"):
        compute_check(role="laptop")


def test_lambda_check_creates_storage_and_grades_the_device(tmp_path) -> None:
    report = compute_check(role="lambda", storage_root=tmp_path / "lambda")
    names = {check["name"] for check in report["checks"]}
    assert "storage:persistent_filesystem" in names
    assert "gpu:cuda_devices" in names
    assert "gpu:vllm" in names
    assert (tmp_path / "lambda" / "hf").is_dir()
    storage = next(
        check for check in report["checks"] if check["name"] == "storage:persistent_filesystem"
    )
    assert storage["ok"]
    assert storage["data"]["environment"]["HF_HUB_ENABLE_HF_TRANSFER"] == "1"


def _prediction(seed: str, model: str) -> dict[str, Any]:
    request_id = hashlib.sha256(seed.encode()).hexdigest()
    return {
        "request": {
            "request_id": request_id,
            "tier": "A",
            "sequence": "screening",
            "model": model,
            "site_id": "site_0001",
            "view_id": "view_a3km_nadir",
            "target_class": "road",
            "prompt_id": "neutral_presence_v4",
            "image_path": "site_0001/view_a3km_nadir/rgb.png",
            "image_sha256": "a" * 64,
            "grid_size": 6,
            "cell_budget": 6,
            "intervention_fraction": 0.0,
        },
        "response": {
            "answer": "yes",
            "confidence": 90,
            "evidence_cells": ["2,2"],
            "cell_ranking": ["2,2"],
        },
        "raw_response": '{"answer": "yes"}',
        "format_failure": False,
        "attempts": 1,
        "latency_s": 1.0,
    }


def test_merge_runs_joins_hosts_and_rejects_conflicts(tmp_path) -> None:
    lambda_log = tmp_path / "lambda.jsonl"
    api_log = tmp_path / "openrouter.jsonl"
    lambda_log.write_text(
        "\n".join(
            json.dumps(_prediction(f"lam{index}", "qwen/qwen3-vl-8b-instruct"))
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    api_log.write_text(
        "\n".join(
            json.dumps(_prediction(f"api{index}", "openai/gpt-5.6-sol")) for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    merged = tmp_path / "merged.jsonl"
    report = merge_predictions([lambda_log, api_log], merged)
    assert report["merged_rows"] == 5
    assert report["rows_by_model"] == {
        "openai/gpt-5.6-sol": 2,
        "qwen/qwen3-vl-8b-instruct": 3,
    }
    assert report["identical_duplicates_dropped"] == 0
    assert merged.with_suffix(".merge.json").is_file()

    # An identical row present in both logs is dropped, not duplicated.
    both = tmp_path / "both.jsonl"
    both.write_text(json.dumps(_prediction("lam0", "qwen/qwen3-vl-8b-instruct")) + "\n", "utf-8")
    again = merge_predictions([lambda_log, both], tmp_path / "again.jsonl")
    assert again["merged_rows"] == 3
    assert again["identical_duplicates_dropped"] == 1

    # The same request ID with a different answer is a protocol error.
    conflicting = _prediction("lam0", "qwen/qwen3-vl-8b-instruct")
    conflicting["response"]["answer"] = "no"
    clash = tmp_path / "clash.jsonl"
    clash.write_text(json.dumps(conflicting) + "\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="two recorded answers"):
        merge_predictions([lambda_log, clash], tmp_path / "conflict.jsonl")

    with pytest.raises(DataValidationError, match="missing prediction logs"):
        merge_predictions([tmp_path / "absent.jsonl"], tmp_path / "no.jsonl")
    with pytest.raises(DataValidationError, match="at least one"):
        merge_predictions([], tmp_path / "no.jsonl")


def _manifest(models: list[dict[str, Any]], index_sha: str = "b" * 64) -> dict[str, Any]:
    return {
        "schema_version": "4.0.0",
        "created_at": "2026-08-09T00:00:00+00:00",
        "canyonbench_version": "0.1.0",
        "dataset_index_sha256": index_sha,
        "models": models,
        "tiers": ["A"],
    }


def test_merge_carries_benchmark_roles_across_hosts(tmp_path) -> None:
    """Scoring reads roles from a manifest beside the log; the merge must keep it."""

    lambda_dir = tmp_path / "lambda"
    api_dir = tmp_path / "openrouter"
    for directory in (lambda_dir, api_dir):
        directory.mkdir()
    (lambda_dir / "predictions.jsonl").write_text(
        json.dumps(_prediction("l0", "qwen/qwen3-vl-8b-instruct")) + "\n", encoding="utf-8"
    )
    (api_dir / "predictions.jsonl").write_text(
        json.dumps(_prediction("a0", "openai/gpt-5.6-sol")) + "\n", encoding="utf-8"
    )
    (lambda_dir / "run_manifest.json").write_text(
        json.dumps(
            _manifest([{"id": "qwen/qwen3-vl-8b-instruct", "benchmark_role": "open_weight"}])
        ),
        encoding="utf-8",
    )
    (api_dir / "run_manifest.json").write_text(
        json.dumps(_manifest([{"id": "openai/gpt-5.6-sol", "benchmark_role": "proprietary"}])),
        encoding="utf-8",
    )

    merged = tmp_path / "final" / "predictions.jsonl"
    report = merge_predictions(
        [lambda_dir / "predictions.jsonl", api_dir / "predictions.jsonl"], merged
    )
    assert report["roles"] == {
        "openai/gpt-5.6-sol": "proprietary",
        "qwen/qwen3-vl-8b-instruct": "open_weight",
    }
    assert report["models_without_a_run_manifest"] == []
    carried = json.loads((merged.parent / "run_manifest.json").read_text())
    assert {model["id"] for model in carried["models"]} == set(report["rows_by_model"])
    assert carried["dataset_index_sha256"] == "b" * 64

    # Hosts that ran against different frozen bundles must not be merged.
    (api_dir / "run_manifest.json").write_text(
        json.dumps(
            _manifest(
                [{"id": "openai/gpt-5.6-sol", "benchmark_role": "proprietary"}],
                index_sha="c" * 64,
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="different dataset indexes"):
        merge_predictions(
            [lambda_dir / "predictions.jsonl", api_dir / "predictions.jsonl"],
            tmp_path / "bad" / "predictions.jsonl",
        )

    # The same model declared differently by two hosts is a roster conflict.
    (api_dir / "run_manifest.json").write_text(
        json.dumps(
            _manifest(
                [{"id": "qwen/qwen3-vl-8b-instruct", "benchmark_role": "proprietary"}],
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="declared differently"):
        merge_predictions(
            [lambda_dir / "predictions.jsonl", api_dir / "predictions.jsonl"],
            tmp_path / "clash" / "predictions.jsonl",
        )


def test_frozen_roster_splits_cleanly_across_the_two_hosts() -> None:
    from canyonbench.trace.config import load_trace_run_config

    config = load_trace_run_config(ROOT / "configs" / "trace_run.frozen.yaml")
    served = [model for model in config.models if not model.metered]
    paid = [model for model in config.models if model.metered]

    assert len(config.models) == 8
    # Everything self-served points at a local port and names its weights.
    for model in served:
        base_url = model.adapter.base_url or ""
        assert "127.0.0.1" in base_url, f"{model.id} is unmetered but not local"
        if model.benchmark_role != "detector":
            assert model.served_model_id, f"{model.id} must name its weights repository"
    # Everything paid goes through OpenRouter with explicit non-zero pricing.
    for model in paid:
        assert model.adapter.base_url == "https://openrouter.ai/api/v1"
        assert model.adapter.api_key_env == "OPENROUTER_API_KEY"
        assert (model.input_per_million_usd or 0) > 0
    assert config.budget.max_cost_usd == 220.0
    assert {model.benchmark_role for model in served} == {
        "open_weight",
        "remote_sensing",
        "detector",
    }


def test_lambda_driver_selects_exactly_the_self_served_models() -> None:
    """The driver's selection rule must match the frozen roster's local slice."""

    import urllib.parse

    config = yaml.safe_load(
        (ROOT / "configs" / "trace_run.frozen.yaml").read_text(encoding="utf-8")
    )
    selected = []
    for model in config["models"]:
        if model.get("metered", True) or model.get("benchmark_role") == "detector":
            continue
        url = urllib.parse.urlparse(model["adapter"]["base_url"])
        if url.hostname in {"127.0.0.1", "localhost"}:
            selected.append((model["id"], model.get("served_model_id"), url.port or 8000))

    assert [row[0] for row in selected] == [
        "qwen/qwen3-vl-8b-instruct",
        "qwen/qwen3-vl-32b-instruct",
        "akshaydudhane/EarthDial_4B_RGB",
    ]
    assert all(weights for _, weights, _ in selected)
    # EarthDial needs its own wrapper, so it must not collide with the vLLM port.
    ports = {identifier: port for identifier, _, port in selected}
    assert ports["akshaydudhane/EarthDial_4B_RGB"] != ports["qwen/qwen3-vl-8b-instruct"]
