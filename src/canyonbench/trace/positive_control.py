"""V1 synthetic-insert execution through the same strict model adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr  # type: ignore[import-untyped]

from canyonbench.eval.adapters import Adapter, make_adapter
from canyonbench.eval.budget import BudgetTracker
from canyonbench.io import iter_jsonl, sha256_file, write_json, write_jsonl
from canyonbench.trace.cave import yes_probability
from canyonbench.trace.config import load_prompts
from canyonbench.trace.instruments import build_synthetic_width_series
from canyonbench.trace.protocol import request_id
from canyonbench.trace.runner import _load_fixture, _query
from canyonbench.trace.schemas import (
    TracePrediction,
    TraceRequest,
    TraceRunConfig,
)


def _view_id(width: float) -> str:
    return f"view_v1_w{str(width).replace('.', 'p')}"


def run_positive_control(
    config: TraceRunConfig,
    negative_image: Path,
    output_dir: Path,
    *,
    widths: list[float] | None = None,
    model_ids: list[str] | None = None,
    adapters: dict[str, Adapter] | None = None,
) -> Path:
    """Generate V1 images and run selected frozen-roster models, resumably."""

    widths = widths or [0.5, 0.75, 1, 1.5, 2, 3, 4, 6]
    output_dir.mkdir(parents=True, exist_ok=True)
    controls = build_synthetic_width_series(
        negative_image,
        output_dir / "images",
        widths,
    )
    write_json(output_dir / "controls.json", controls)
    selected = [model for model in config.models if model_ids is None or model.id in model_ids]
    if not selected:
        raise ValueError("no configured models matched the requested V1 model IDs")
    if model_ids is not None:
        missing = sorted(set(model_ids) - {model.id for model in selected})
        if missing:
            raise ValueError(f"unknown V1 model IDs: {missing}")
    predictions_path = output_dir / "predictions.jsonl"
    raw_rows = list(iter_jsonl(predictions_path)) if predictions_path.exists() else []
    existing = {
        prediction.request.request_id: prediction
        for prediction in (TracePrediction.model_validate(row) for row in raw_rows)
    }
    metered = {model.id for model in selected if model.metered}
    budget = BudgetTracker(
        config.budget,
        requests=sum(
            prediction.attempts
            for prediction in existing.values()
            if prediction.request.model in metered
        ),
        cost_usd=sum(prediction.cost_usd for prediction in existing.values()),
    )
    prompt = next(
        template for template in load_prompts(config.prompt_file) if template.variant == "neutral"
    )
    write_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": "4.0.0",
            "instrument": "V1_synthetic_insert_positive_control",
            "negative_image_sha256": sha256_file(negative_image),
            "prompt_sha256": sha256_file(config.prompt_file),
            "widths_px": widths,
            "models": [model.model_dump(mode="json") for model in selected],
        },
    )
    fixture = _load_fixture(config.fixture_responses)
    model_adapters = adapters or {model.id: make_adapter(model, fixture) for model in selected}
    for model in selected:
        for control in controls:
            width = float(control["apparent_width_px"])
            image_path = Path(str(control["image_path"]))
            image_sha256 = sha256_file(image_path)
            identifier = request_id(
                "V1",
                model.id,
                image_sha256,
                width,
                prompt.id,
            )
            if identifier in existing:
                continue
            request = TraceRequest(
                request_id=identifier,
                tier="A",
                sequence="synthetic_positive_control",
                model=model.id,
                site_id="site_0000",
                view_id=_view_id(width),
                target_class="road",
                prompt_id=prompt.id,
                image_path=image_path,
                image_sha256=image_sha256,
                grid_size=6,
                cell_budget=6,
                synthetic_apparent_width_px=width,
            )
            prediction = _query(
                request=request,
                template=prompt,
                adapter=model_adapters[model.id],
                model=model,
                parse_retries=config.protocol.parse_retries,
                budget=budget,
            )
            raw_rows.append(prediction.model_dump(mode="json"))
            existing[identifier] = prediction
            write_jsonl(predictions_path, raw_rows)
    return predictions_path


def score_positive_control(predictions_path: Path, output: Path) -> dict[str, Any]:
    """Report response-vs-width monotonicity and the preregistered V1 pass check."""

    rows = [TracePrediction.model_validate(row) for row in iter_jsonl(predictions_path)]
    result: dict[str, Any] = {}
    for model in sorted({row.request.model for row in rows}):
        model_rows = [
            row
            for row in rows
            if row.request.model == model and row.request.synthetic_apparent_width_px is not None
        ]
        widths = np.asarray(
            [row.request.synthetic_apparent_width_px for row in model_rows],
            dtype=float,
        )
        probabilities = np.asarray(
            [yes_probability(row.response) for row in model_rows],
            dtype=float,
        )
        high = probabilities[widths >= 2]
        subpixel = probabilities[widths < 1]
        correlation = (
            float(spearmanr(widths, probabilities).statistic)
            if len(np.unique(probabilities)) > 1
            else 0.0
        )
        high_mean = float(np.mean(high)) if len(high) else float("nan")
        subpixel_mean = float(np.mean(subpixel)) if len(subpixel) else float("nan")
        result[model] = {
            "n_widths": len(widths),
            "spearman_width_vs_yes_probability": correlation,
            "mean_yes_probability_width_ge_2px": high_mean,
            "mean_yes_probability_width_lt_1px": subpixel_mean,
            "positive_control_pass": bool(
                np.isfinite(high_mean)
                and np.isfinite(subpixel_mean)
                and high_mean >= 0.5
                and high_mean > subpixel_mean
            ),
            "curve": [
                {
                    "apparent_width_px": float(width),
                    "yes_probability": float(probability),
                }
                for width, probability in sorted(zip(widths, probabilities, strict=True))
            ],
        }
    payload = {
        "schema_version": "4.0.0",
        "instrument": "V1_synthetic_insert_positive_control",
        "models": result,
    }
    write_json(output, payload)
    return payload
