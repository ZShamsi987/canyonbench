"""D1 price pilot: measure real per-call token cost before authorizing a run.

Section 18 makes provider image-token accounting the single most likely source of
a budget surprise, so the registered token model is never trusted for
authorization. This runs a small, capped batch of real Tier-A screening calls,
records the token counts the provider actually returns, and re-prices the full
plan from the measurement.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from canyonbench.eval.adapters import Adapter, make_adapter
from canyonbench.eval.budget import BudgetTracker
from canyonbench.exceptions import DataValidationError
from canyonbench.io import read_json, write_json, write_jsonl
from canyonbench.schemas import BudgetConfig
from canyonbench.trace.config import load_prompts
from canyonbench.trace.planning import estimate_call_plan
from canyonbench.trace.protocol import stratified_select
from canyonbench.trace.runner import _load_fixture, _query, _request
from canyonbench.trace.schemas import TraceRunConfig
from canyonbench.version import __version__


def run_price_pilot(
    config: TraceRunConfig,
    *,
    calls_per_model: int = 50,
    include_unmetered: bool = False,
    output_dir: Path | None = None,
    adapters: dict[str, Adapter] | None = None,
) -> dict[str, Any]:
    """Measure observed per-call tokens/cost and re-project the full-run budget."""

    if calls_per_model < 1:
        raise ValueError("the price pilot requires at least one call per model")
    index = read_json(config.dataset_dir / "index.json")
    clean = [row for row in index if row.get("variant") == "clean"]
    if not clean:
        raise DataValidationError(f"No clean views in {config.dataset_dir / 'index.json'}")
    views = stratified_select(
        clean,
        min(calls_per_model, len(clean)),
        seed=config.protocol.seed,
    )
    prompts = {prompt.variant: prompt for prompt in load_prompts(config.prompt_file)}
    template = prompts["neutral"]
    destination = output_dir or config.output_dir / "price_pilot"
    destination.mkdir(parents=True, exist_ok=True)

    models = [model for model in config.models if model.metered or include_unmetered]
    if not models:
        raise DataValidationError(
            "No metered models to price; pass include_unmetered to smoke-test"
        )
    fixture = _load_fixture(config.fixture_responses)
    model_adapters = adapters or {model.id: make_adapter(model, fixture) for model in models}

    # A pilot must never be able to spend the production budget.
    pilot_budget = BudgetConfig(
        max_requests=max(1, calls_per_model * len(models) * (config.protocol.parse_retries + 1)),
        max_cost_usd=max(1.0, config.budget.max_cost_usd * 0.05),
        input_per_million_usd=config.budget.input_per_million_usd,
        output_per_million_usd=config.budget.output_per_million_usd,
    )
    budget = BudgetTracker(pilot_budget)

    rows: list[dict[str, Any]] = []
    observed: dict[str, dict[str, float]] = {}
    by_model: dict[str, dict[str, Any]] = {}
    for model in models:
        adapter = model_adapters[model.id]
        input_tokens: list[int] = []
        output_tokens: list[int] = []
        costs: list[float] = []
        latencies: list[float] = []
        failures = 0
        for row in views:
            request = _request(
                tier="A",
                sequence="screening",
                model_id=model.id,
                row=row,
                prompt=template,
                image_path=config.dataset_dir / str(row["image_path"]),
                grid_size=6,
                cell_budget=6,
            )
            prediction = _query(
                request=request,
                template=template,
                adapter=adapter,
                model=model,
                parse_retries=config.protocol.parse_retries,
                budget=budget,
            )
            rows.append(prediction.model_dump(mode="json"))
            input_tokens.append(prediction.input_tokens)
            output_tokens.append(prediction.output_tokens)
            costs.append(prediction.cost_usd)
            latencies.append(prediction.latency_s)
            failures += int(prediction.format_failure)
        measured_input = float(np.mean(input_tokens)) if input_tokens else 0.0
        measured_output = float(np.mean(output_tokens)) if output_tokens else 0.0
        if model.metered and measured_input > 0:
            observed[model.id] = {"input": measured_input, "output": measured_output}
        by_model[model.id] = {
            "metered": bool(model.metered),
            "calls": len(input_tokens),
            "format_failures": failures,
            "format_failure_rate": (failures / len(input_tokens) if input_tokens else float("nan")),
            "mean_input_tokens": measured_input,
            "mean_output_tokens": measured_output,
            "max_input_tokens": int(max(input_tokens, default=0)),
            "observed_usd_per_call": (float(np.mean(costs)) if costs else 0.0),
            "observed_usd_total": float(sum(costs)),
            "median_latency_s": (float(np.median(latencies)) if latencies else float("nan")),
            "tokens_reported_by_provider": measured_input > 0,
        }

    plan_registered = estimate_call_plan(config)
    plan_measured = estimate_call_plan(config, observed_tokens=observed)
    report = {
        "schema_version": "4.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "canyonbench_version": __version__,
        "calls_per_model": len(views),
        "pilot_request_cap": pilot_budget.max_requests,
        "pilot_cost_cap_usd": pilot_budget.max_cost_usd,
        "pilot_spend_usd": budget.cost_usd,
        "by_model": by_model,
        "observed_tokens_per_call": observed,
        "registered_projection_usd": plan_registered["cost_projection_usd"],
        "measured_projection_usd": plan_measured["cost_projection_usd"],
        "authorized": bool(plan_measured["cost_projection_usd"]["nominal_fits_cost_cap"]),
        "decision_rule": (
            "Authorize the full run only when the measured projection fits the "
            "configured cost cap; otherwise descope per the registered ladder before "
            "raising the cap."
        ),
    }
    write_jsonl(destination / "pilot_predictions.jsonl", rows)
    write_json(destination / "price_pilot.json", report)
    return report
