from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from canyonbench.eval.adapters import OpenAICompatibleAdapter, encode_image, make_adapter
from canyonbench.eval.budget import BudgetTracker
from canyonbench.eval.probes import load_probes
from canyonbench.eval.statistics import controlled_association, stratified_trends
from canyonbench.exceptions import BudgetExceededError, DataValidationError
from canyonbench.reporting.figures import plot_ascent_association, plot_registration_residuals
from canyonbench.reporting.tables import flatten_metrics
from canyonbench.schemas import AdapterConfig, BudgetConfig, ModelConfig


def test_openai_compatible_adapter_with_mock_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (32, 16), "brown").save(image)
    assert encode_image(image, 16)
    monkeypatch.setenv("TEST_API_KEY", "secret")
    model = ModelConfig(
        id="provider/model",
        adapter=AdapterConfig(
            kind="openai_compatible",
            base_url="https://example.test/v1",
            api_key_env="TEST_API_KEY",
            max_retries=0,
        ),
    )
    adapter = OpenAICompatibleAdapter(model)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["response_format"]["type"] == "json_schema"
        return httpx.Response(
            200,
            headers={"x-request-id": "request-1"},
            json={
                "id": "fallback-id",
                "choices": [{"message": {"content": '{"percent":12}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )

    adapter.client.close()
    adapter.client = httpx.Client(
        base_url="https://example.test/v1", transport=httpx.MockTransport(handler)
    )
    response = adapter.complete(
        image_path=image,
        system="system",
        prompt="prompt",
        json_schema={"type": "object"},
        model=model,
    )
    assert response.provider_request_id == "request-1"
    assert response.output_tokens == 2
    assert make_adapter(ModelConfig(id="f", adapter=AdapterConfig(kind="fixture")))


def test_adapter_requires_named_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_TEST_KEY", raising=False)
    model = ModelConfig(
        id="m",
        adapter=AdapterConfig(
            kind="openai_compatible",
            base_url="https://example.test",
            api_key_env="MISSING_TEST_KEY",
        ),
    )
    with pytest.raises(DataValidationError, match="unset"):
        OpenAICompatibleAdapter(model)


def test_budget_caps_are_hard() -> None:
    tracker = BudgetTracker(
        BudgetConfig(
            max_requests=1,
            max_cost_usd=0.001,
            input_per_million_usd=100,
            output_per_million_usd=100,
        )
    )
    tracker.reserve_request()
    with pytest.raises(BudgetExceededError, match="Request cap"):
        tracker.reserve_request()
    with pytest.raises(BudgetExceededError, match="Cost cap"):
        tracker.record_tokens(100, 100)


def test_probe_contract_errors(tmp_path: Path) -> None:
    path = tmp_path / "probes.yaml"
    path.write_text("schema_version: 1\nprobes: []\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="non-empty"):
        load_probes(path)
    path.write_text(
        "schema_version: 1\nprobes:\n  - {name: unknown, variant: neutral, system: x, prompt: y}\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="Unknown"):
        load_probes(path)


def test_controlled_and_stratified_ascent_associations() -> None:
    rng = np.random.default_rng(2)
    count = 60
    frame = pd.DataFrame(
        {
            "alt_m": np.linspace(1000, 24000, count),
            "contrast_std": rng.uniform(0.1, 0.4, count),
            "sharpness_edge_var": rng.uniform(0.01, 0.1, count),
            "clipped_high_fraction": rng.uniform(0, 0.05, count),
            "clipped_low_fraction": rng.uniform(0, 0.05, count),
            "feature_prevalence": rng.uniform(0, 1, count),
            "phase": ["Launching"] * 30 + ["Floating"] * 30,
            "segment_id": [f"s{index // 10}" for index in range(count)],
        }
    )
    frame["error"] = frame["alt_m"] * 0.0001 + rng.normal(0, 0.1, count)
    result = controlled_association(frame, outcome="error")
    assert result["interpretation"] == "association_not_causation"
    assert result["effective_segments"] == 6
    trends = stratified_trends(frame, outcome="error")
    assert trends


def test_flatten_metrics() -> None:
    frame = flatten_metrics(
        {"models": {"m": {"vegetation:neutral": {"mae": 2.0, "ci": {"lower": 1}}}}}
    )
    assert frame.to_dict(orient="records") == [
        {"model": "m", "probe": "vegetation", "variant": "neutral", "metric": "mae", "value": 2.0}
    ]


def test_figure_builders_write_outputs(tmp_path: Path) -> None:
    ascent = pd.DataFrame(
        {
            "alt_m": np.arange(24, dtype=float) * 1000,
            "error": np.linspace(0, 1, 24),
            "segment_id": [f"s{index // 4}" for index in range(24)],
        }
    )
    assert plot_ascent_association(
        ascent, tmp_path / "ascent.png", outcome="error", bins=6
    ).exists()
    residuals = pd.DataFrame(
        {
            "holdout_rmse_m": [10, 20],
            "threshold_m": [15, 15],
            "reliable": [True, False],
        }
    )
    assert plot_registration_residuals(residuals, tmp_path / "registration.png").exists()
