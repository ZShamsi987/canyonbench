"""Preflight request and dollar accounting for the tiered protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from canyonbench.io import read_json, write_json
from canyonbench.trace.protocol import stratified_select
from canyonbench.trace.schemas import TraceRunConfig

# Registered v4 token model for a 768 px long-edge view plus the structured answer.
# `trace price-pilot` replaces these with provider-reported counts before the run.
REGISTERED_INPUT_TOKENS_PER_CALL = 1400
REGISTERED_OUTPUT_TOKENS_PER_CALL = 80


def estimate_call_plan(
    config: TraceRunConfig,
    *,
    observed_tokens: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    index = read_json(config.dataset_dir / "index.json")
    clean = [row for row in index if row.get("variant") == "clean"]
    degraded = [row for row in index if row.get("variant") == "degraded"]
    tier_b = stratified_select(
        clean,
        min(config.protocol.causal_core_views, len(clean)),
        seed=config.protocol.seed + 1,
    )
    # Tier B costs differ by who pays: metered models take the primary operator
    # only, self-served models take all three. The plan therefore has to be
    # built per model rather than shared.
    metered_operators = set(config.protocol.metered_causal_operators)
    credited_operators = set(config.protocol.causal_operators)

    common: dict[str, int] = {}
    if "A" in config.tiers:
        common["tier_a_clean"] = min(config.protocol.screening_views, len(clean))
        common["tier_a_degraded_robustness"] = min(config.protocol.robustness_views, len(degraded))

    static_by_operator: dict[str, int] = {}
    if "B" in config.tiers:
        for row in tier_b:
            manifest = (
                (config.dataset_dir / str(row["image_path"])).parent
                / "interventions"
                / "manifest.json"
            )
            if not manifest.exists():
                continue
            for record in read_json(manifest):
                if not record.get("accepted"):
                    continue
                operator = str(record["operator"])
                if operator == "inpaint" and "inpainting" not in config.analyses:
                    continue
                static_by_operator[operator] = static_by_operator.get(operator, 0) + 1

    tier_c_count = min(config.protocol.prompt_cave_views, len(clean)) if "C" in config.tiers else 0
    robustness_count = min(config.protocol.robustness_views, len(clean))
    sensitivity_calls_per_view = sum(
        1 + 4 * cell_budget * len(config.protocol.causal_operators)
        for _grid_size in config.protocol.grid_sizes
        for cell_budget in config.protocol.cell_budgets
    )

    breakdown_by_model: dict[str, dict[str, int]] = {}
    per_model: dict[str, int] = {}
    for model in config.models:
        breakdown = dict(common)
        operators = metered_operators if model.metered else credited_operators
        if "B" in config.tiers:
            breakdown["tier_b_oracle_and_distractor"] = sum(
                count
                for operator, count in static_by_operator.items()
                if operator in operators or operator == "inpaint"
            )
        if model.benchmark_role != "detector":
            if "B" in config.tiers:
                # Worst case: every query names the full cell budget.
                breakdown["tier_b_self_and_controls_max"] = len(tier_b) * 4 * 6 * len(operators)
            if "C" in config.tiers:
                breakdown["tier_c_prompts_and_image_controls"] = tier_c_count * 8
                breakdown["tier_c_cave_stages_max"] = tier_c_count * 6
            if "sensitivity" in config.analyses and (
                config.protocol.sensitivity_on_metered_models or not model.metered
            ):
                breakdown["v4_grid_k_sensitivity_max"] = (
                    robustness_count * sensitivity_calls_per_view
                )
        breakdown_by_model[model.id] = breakdown
        per_model[model.id] = sum(breakdown.values())
    metered_models = [model.id for model in config.models if model.metered]
    nominal_metered = sum(per_model[model] for model in metered_models)
    cost = project_cost(config, per_model, observed_tokens=observed_tokens)
    return {
        "breakdown_by_model": breakdown_by_model,
        "nominal_calls_per_model": per_model,
        "maximum_nominal_calls_per_model": max(per_model.values(), default=0),
        "calls_by_model": per_model,
        "metered_models": metered_models,
        "nominal_metered_requests": nominal_metered,
        "worst_case_metered_requests_with_parse_retries": nominal_metered
        * (config.protocol.parse_retries + 1),
        "configured_request_cap": config.budget.max_requests,
        "nominal_fits_request_cap": nominal_metered <= config.budget.max_requests,
        "sensitivity_is_separate_expensive_run": "sensitivity" in config.analyses,
        "cost_projection_usd": cost,
    }


def project_cost(
    config: TraceRunConfig,
    calls_by_model: dict[str, int],
    *,
    observed_tokens: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Price the nominal plan per metered model against the configured cost cap.

    ``observed_tokens`` carries measured per-call token means from the D1 price
    pilot (``{model_id: {"input": float, "output": float}}``); the registered
    token model is used for any model the pilot has not measured.
    """

    observed_tokens = observed_tokens or {}
    by_model: dict[str, dict[str, Any]] = {}
    for model in config.models:
        if not model.metered:
            continue
        measured = observed_tokens.get(model.id, {})
        input_tokens = float(measured.get("input", REGISTERED_INPUT_TOKENS_PER_CALL))
        output_tokens = float(measured.get("output", REGISTERED_OUTPUT_TOKENS_PER_CALL))
        input_price = (
            model.input_per_million_usd
            if model.input_per_million_usd is not None
            else config.budget.input_per_million_usd
        )
        output_price = (
            model.output_per_million_usd
            if model.output_per_million_usd is not None
            else config.budget.output_per_million_usd
        )
        per_call = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
        calls = calls_by_model.get(model.id, 0)
        by_model[model.id] = {
            "calls": float(calls),
            "input_tokens_per_call": input_tokens,
            "output_tokens_per_call": output_tokens,
            "input_per_million_usd": float(input_price),
            "output_per_million_usd": float(output_price),
            "usd_per_call": per_call,
            "nominal_usd": per_call * calls,
            "token_source": "price_pilot" if measured else "registered_estimate",
        }
    nominal = sum(values["nominal_usd"] for values in by_model.values())
    worst_case = nominal * (config.protocol.parse_retries + 1)
    return {
        "by_model": by_model,
        "nominal_usd": nominal,
        "worst_case_usd_with_parse_retries": worst_case,
        "configured_cost_cap_usd": config.budget.max_cost_usd,
        "nominal_fits_cost_cap": nominal <= config.budget.max_cost_usd,
        "worst_case_fits_cost_cap": worst_case <= config.budget.max_cost_usd,
        "note": (
            "Tier B/C self-evidence counts are worst case: every named cell costs a "
            "call. The runtime BudgetTracker aborts on the first request that would "
            "breach the cap, so this projection is the authorization check, not the "
            "enforcement mechanism."
        ),
    }


def write_call_plan(
    config: TraceRunConfig,
    output: Path,
    *,
    observed_tokens: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    plan = estimate_call_plan(config, observed_tokens=observed_tokens)
    write_json(output, plan)
    return plan
