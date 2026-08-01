"""Resumable Tier A/B/C black-box execution with strict parsing and retries."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from canyonbench.eval.adapters import Adapter, make_adapter
from canyonbench.eval.budget import BudgetTracker
from canyonbench.exceptions import BudgetExceededError, DataValidationError
from canyonbench.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl
from canyonbench.trace.baselines import materialize_image_controls
from canyonbench.trace.config import load_prompts
from canyonbench.trace.planning import write_call_plan
from canyonbench.trace.protocol import materialize_self_sequences, request_id, stratified_select
from canyonbench.trace.schemas import (
    PromptTemplate,
    TracePrediction,
    TraceRequest,
    TraceResponse,
    TraceRunConfig,
)
from canyonbench.version import __version__

PRIMARY_OPERATORS = ("blur", "texture", "frequency")


def _schema() -> dict[str, Any]:
    return TraceResponse.model_json_schema()


def _call_key(
    request: TraceRequest,
    template: PromptTemplate,
    model: Any,
) -> str:
    image_hash = (
        "NO_IMAGE"
        if template.variant == "no_image"
        else request.image_sha256 or sha256_file(request.image_path)
    )
    return request_id(
        model.id,
        image_hash,
        template.system,
        _prompt(
            template,
            target_class=request.target_class,
            grid_size=request.grid_size,
            cell_budget=request.cell_budget,
        ),
        model.temperature,
        model.max_tokens,
        model.image_max_side,
        _schema(),
    )


def _prompt(
    template: PromptTemplate,
    *,
    target_class: str,
    grid_size: int,
    cell_budget: int,
) -> str:
    return template.user.format(
        target_class=target_class,
        grid_size=grid_size,
        cell_budget=cell_budget,
    )


def _load_fixture(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    value = read_json(path)
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(content, str) for key, content in value.items()
    ):
        raise DataValidationError("fixture responses must be a JSON string-to-string object")
    return value


def _query(
    *,
    request: TraceRequest,
    template: PromptTemplate,
    adapter: Adapter,
    model: Any,
    parse_retries: int,
    budget: BudgetTracker,
) -> TracePrediction:
    raw: str | None = None
    errors: list[str] = []
    started = time.monotonic()
    input_tokens = output_tokens = 0
    cost = 0.0
    provider_request_id = finish_reason = None
    attempts = 0
    prompt = _prompt(
        template,
        target_class=request.target_class,
        grid_size=request.grid_size,
        cell_budget=request.cell_budget,
    )
    for attempts in range(1, parse_retries + 2):
        if model.metered:
            budget.reserve_request()
        try:
            result = adapter.complete(
                image_path=None if template.variant == "no_image" else request.image_path,
                system=template.system,
                prompt=prompt,
                json_schema=_schema(),
                model=model,
            )
            raw = result.content
            input_tokens += result.input_tokens
            output_tokens += result.output_tokens
            if not model.metered:
                cost += budget.record_cost(0.0)
            elif (
                model.input_per_million_usd is not None and model.output_per_million_usd is not None
            ):
                request_cost = (
                    result.input_tokens * model.input_per_million_usd
                    + result.output_tokens * model.output_per_million_usd
                ) / 1_000_000
                cost += budget.record_cost(request_cost)
            else:
                cost += budget.record_tokens(result.input_tokens, result.output_tokens)
            provider_request_id = result.provider_request_id
            finish_reason = result.raw_finish_reason
            parsed = TraceResponse.model_validate_json(raw)
            parsed.validate_protocol(grid_size=request.grid_size, cell_budget=request.cell_budget)
            return TracePrediction(
                request=request,
                response=parsed,
                raw_response=raw,
                format_failure=False,
                attempts=attempts,
                latency_s=time.monotonic() - started,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                provider_request_id=provider_request_id,
                finish_reason=finish_reason,
            )
        except BudgetExceededError:
            raise
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return TracePrediction(
        request=request,
        response=None,
        raw_response=raw,
        format_failure=True,
        attempts=attempts,
        error=" | ".join(errors),
        latency_s=time.monotonic() - started,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        provider_request_id=provider_request_id,
        finish_reason=finish_reason,
    )


def _request(
    *,
    tier: str,
    sequence: str,
    model_id: str,
    row: dict[str, Any],
    prompt: PromptTemplate,
    image_path: Path,
    grid_size: int,
    cell_budget: int,
    fraction: float = 0,
    operator: str | None = None,
    cave_stage: str | None = None,
    baseline_kind: str | None = None,
) -> TraceRequest:
    image_sha256 = sha256_file(image_path)
    key = request_id(
        tier,
        sequence,
        model_id,
        row["site_id"],
        row["view_id"],
        prompt.id,
        fraction,
        operator,
        image_sha256,
        grid_size,
        cell_budget,
        cave_stage,
        baseline_kind,
    )
    return TraceRequest(
        request_id=key,
        tier=tier,  # type: ignore[arg-type]
        sequence=sequence,  # type: ignore[arg-type]
        model=model_id,
        site_id=str(row["site_id"]),
        view_id=str(row["view_id"]),
        target_class=str(row["target_class"]),  # type: ignore[arg-type]
        prompt_id=prompt.id,
        image_path=image_path,
        image_sha256=image_sha256,
        grid_size=grid_size,  # type: ignore[arg-type]
        cell_budget=cell_budget,  # type: ignore[arg-type]
        intervention_fraction=fraction,
        intervention_operator=operator,  # type: ignore[arg-type]
        cave_stage=cave_stage,  # type: ignore[arg-type]
        baseline_kind=baseline_kind,  # type: ignore[arg-type]
    )


def _existing(
    path: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, TracePrediction],
    dict[tuple[str, str, str], TracePrediction],
]:
    if not path.exists():
        return [], {}, {}
    rows = list(iter_jsonl(path))
    predictions: dict[str, TracePrediction] = {}
    screenings: dict[tuple[str, str, str], TracePrediction] = {}
    for row in rows:
        prediction = TracePrediction.model_validate(row)
        predictions[prediction.request.request_id] = prediction
        if prediction.request.sequence == "screening":
            screenings[
                (
                    prediction.request.model,
                    prediction.request.site_id,
                    prediction.request.view_id,
                )
            ] = prediction
    return rows, predictions, screenings


def _run_one(
    request: TraceRequest,
    template: PromptTemplate,
    *,
    adapter: Adapter,
    model: Any,
    config: TraceRunConfig,
    budget: BudgetTracker,
    rows: list[dict[str, Any]],
    predictions: dict[str, TracePrediction],
    call_cache: dict[str, TracePrediction],
    output: Path,
) -> TracePrediction:
    if request.request_id in predictions:
        return predictions[request.request_id]
    call_key = _call_key(request, template, model)
    cached = call_cache.get(call_key)
    if cached is not None and cached.response is not None:
        prediction = TracePrediction(
            request=request,
            response=cached.response,
            raw_response=cached.raw_response,
            format_failure=False,
            attempts=0,
            cache_hit=True,
            latency_s=0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0,
            provider_request_id=f"cache:{cached.request.request_id}",
            finish_reason=cached.finish_reason,
        )
        rows.append(prediction.model_dump(mode="json"))
        predictions[request.request_id] = prediction
        write_jsonl(output, rows)
        return prediction
    prediction = _query(
        request=request,
        template=template,
        adapter=adapter,
        model=model,
        parse_retries=config.protocol.parse_retries,
        budget=budget,
    )
    rows.append(prediction.model_dump(mode="json"))
    predictions[request.request_id] = prediction
    if prediction.response is not None and not prediction.format_failure:
        call_cache[call_key] = prediction
    write_jsonl(output, rows)
    return prediction


def run_trace(
    config: TraceRunConfig,
    *,
    adapters: dict[str, Adapter] | None = None,
    only_models: list[str] | None = None,
) -> Path:
    """Run all requested tiers; dynamic traces use each model's own Tier A cells.

    ``only_models`` restricts this invocation to part of the frozen roster. The
    Lambda driver serves one model at a time, so it traces one model per vLLM
    session and appends into the same resumable log.
    """

    if only_models:
        known = {model.id for model in config.models}
        unknown = sorted(set(only_models) - known)
        if unknown:
            raise DataValidationError(f"--only-model names models outside the roster: {unknown}")
        config = config.model_copy(
            update={"models": [model for model in config.models if model.id in set(only_models)]}
        )
    index_path = config.dataset_dir / "index.json"
    raw_index = read_json(index_path)
    if not isinstance(raw_index, list):
        raise DataValidationError(f"Dataset index must be a list: {index_path}")
    clean = [row for row in raw_index if row.get("variant") == "clean"]
    degraded = [row for row in raw_index if row.get("variant") == "degraded"]
    if len(clean) < config.protocol.screening_views:
        raise DataValidationError(
            f"Tier A requests {config.protocol.screening_views} clean views; found {len(clean)}"
        )
    prompts = {prompt.variant: prompt for prompt in load_prompts(config.prompt_file)}
    config.output_dir.mkdir(parents=True, exist_ok=True)
    call_plan = write_call_plan(config, config.output_dir / "call_plan.json")
    if not call_plan["nominal_fits_request_cap"]:
        raise DataValidationError(
            "Nominal paid-request plan exceeds max_requests; inspect call_plan.json "
            "and raise the explicit cap or reduce scope before any calls"
        )
    predictions_path = config.output_dir / "predictions.jsonl"
    rows, predictions, screenings = _existing(predictions_path)
    fixture = _load_fixture(config.fixture_responses)
    model_adapters = adapters or {model.id: make_adapter(model, fixture) for model in config.models}
    prompts_by_id = {prompt.id: prompt for prompt in prompts.values()}
    models_by_id = {model.id: model for model in config.models}
    call_cache: dict[str, TracePrediction] = {}
    for prediction in predictions.values():
        template = prompts_by_id.get(prediction.request.prompt_id)
        cached_model = models_by_id.get(prediction.request.model)
        if (
            template is not None
            and cached_model is not None
            and prediction.response is not None
            and not prediction.format_failure
        ):
            call_cache[_call_key(prediction.request, template, cached_model)] = prediction
    metered_models = {model.id for model in config.models if model.metered}
    resumed_requests = sum(
        prediction.attempts
        for prediction in predictions.values()
        if prediction.request.model in metered_models
    )
    resumed_cost = sum(prediction.cost_usd for prediction in predictions.values())
    if resumed_requests > config.budget.max_requests or resumed_cost > config.budget.max_cost_usd:
        raise DataValidationError("Existing run already exceeds the configured request or cost cap")
    budget = BudgetTracker(
        config.budget,
        requests=resumed_requests,
        cost_usd=resumed_cost,
    )
    write_json(
        config.output_dir / "run_manifest.json",
        {
            "schema_version": "4.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "canyonbench_version": __version__,
            "dataset_index_sha256": sha256_file(index_path),
            "prompt_sha256": sha256_file(config.prompt_file),
            "models": [
                model.model_dump(exclude={"adapter": {"api_key_env"}}) for model in config.models
            ],
            "tiers": config.tiers,
            "protocol": config.protocol.model_dump(),
            "budget": config.budget.model_dump(),
            "cache": {
                "key": (
                    "model + source-image SHA-256 + exact system/user prompt + "
                    "temperature + token/image limits + response schema"
                ),
                "valid_response_only": True,
                "cache_hits_are_zero_attempt_zero_cost_logical_predictions": True,
            },
        },
    )
    tier_a = stratified_select(clean, config.protocol.screening_views, seed=config.protocol.seed)
    tier_b = stratified_select(
        clean, min(config.protocol.causal_core_views, len(clean)), seed=config.protocol.seed + 1
    )
    # V3 needs every model ranked under every operator. Metered models run the
    # non-primary operators on this subset of Tier B so the rank correlation
    # covers the whole roster rather than only the credited models.
    agreement_views = {
        f"{row['site_id']}/{row['view_id']}"
        for row in stratified_select(
            tier_b,
            min(config.protocol.operator_agreement_views, len(tier_b)),
            seed=config.protocol.seed + 5,
        )
    }
    tier_c = stratified_select(
        clean, min(config.protocol.prompt_cave_views, len(clean)), seed=config.protocol.seed + 2
    )
    robustness = stratified_select(
        clean, min(config.protocol.robustness_views, len(clean)), seed=config.protocol.seed + 3
    )
    degraded_robustness = stratified_select(
        degraded,
        min(config.protocol.robustness_views, len(degraded)),
        seed=config.protocol.seed + 4,
    )
    write_json(
        config.output_dir / "tier_membership.json",
        {
            "A": [f"{row['site_id']}/{row['view_id']}" for row in tier_a],
            "B": [f"{row['site_id']}/{row['view_id']}" for row in tier_b],
            "C": [f"{row['site_id']}/{row['view_id']}" for row in tier_c],
            "V4": [f"{row['site_id']}/{row['view_id']}" for row in robustness],
            "degraded": [f"{row['site_id']}/{row['view_id']}" for row in degraded_robustness],
        },
    )

    # Tier A is a project checkpoint: finish it for every model before starting B/C.
    if "A" in config.tiers:
        for model in config.models:
            adapter = model_adapters[model.id]
            for row in tier_a:
                image = config.dataset_dir / str(row["image_path"])
                request = _request(
                    tier="A",
                    sequence="screening",
                    model_id=model.id,
                    row=row,
                    prompt=prompts["neutral"],
                    image_path=image,
                    grid_size=6,
                    cell_budget=6,
                )
                prediction = _run_one(
                    request,
                    prompts["neutral"],
                    adapter=adapter,
                    model=model,
                    config=config,
                    budget=budget,
                    rows=rows,
                    predictions=predictions,
                    call_cache=call_cache,
                    output=predictions_path,
                )
                screenings[
                    (
                        model.id,
                        str(row["site_id"]),
                        str(row["view_id"]),
                    )
                ] = prediction
            for row in degraded_robustness:
                image = config.dataset_dir / str(row["image_path"])
                request = _request(
                    tier="A",
                    sequence="screening",
                    model_id=model.id,
                    row=row,
                    prompt=prompts["neutral"],
                    image_path=image,
                    grid_size=6,
                    cell_budget=6,
                )
                _run_one(
                    request,
                    prompts["neutral"],
                    adapter=adapter,
                    model=model,
                    config=config,
                    budget=budget,
                    rows=rows,
                    predictions=predictions,
                    call_cache=call_cache,
                    output=predictions_path,
                )

    for model in config.models:
        adapter = model_adapters[model.id]
        # Paid and credited models take different operator sets in Tier B; see
        # TraceProtocolConfig for why.
        operators = tuple(
            config.protocol.metered_causal_operators
            if model.metered
            else config.protocol.causal_operators
        )
        if "B" in config.tiers:
            for row in tier_b:
                view_directory = (config.dataset_dir / str(row["image_path"])).parent
                intervention_manifest = view_directory / "interventions" / "manifest.json"
                interventions = (
                    read_json(intervention_manifest) if intervention_manifest.exists() else []
                )
                # On the agreement subset a metered model takes every registered
                # operator; elsewhere it takes the primary operator only.
                view_operators = (
                    tuple(config.protocol.causal_operators)
                    if f"{row['site_id']}/{row['view_id']}" in agreement_views
                    else operators
                )
                for item in interventions:
                    if not item.get("accepted") or (
                        item["operator"] == "inpaint" and "inpainting" not in config.analyses
                    ):
                        continue
                    if item["operator"] != "inpaint" and item["operator"] not in view_operators:
                        continue
                    image = Path(str(item["image_path"]))
                    if not image.is_absolute():
                        image = view_directory / "interventions" / image.name
                    request = _request(
                        tier="B",
                        sequence=str(item["sequence"]),
                        model_id=model.id,
                        row=row,
                        prompt=prompts["neutral"],
                        image_path=image,
                        grid_size=6,
                        cell_budget=6,
                        fraction=float(item["fraction"]),
                        operator=str(item["operator"]),
                    )
                    _run_one(
                        request,
                        prompts["neutral"],
                        adapter=adapter,
                        model=model,
                        config=config,
                        budget=budget,
                        rows=rows,
                        predictions=predictions,
                        call_cache=call_cache,
                        output=predictions_path,
                    )
                screening = screenings.get(
                    (
                        model.id,
                        str(row["site_id"]),
                        str(row["view_id"]),
                    )
                )
                if model.benchmark_role != "detector" and screening and screening.response:
                    for operator in operators:
                        dynamic = materialize_self_sequences(
                            view_directory / "rgb.png",
                            screening.response,
                            config.output_dir
                            / "dynamic_interventions"
                            / model.id
                            / str(row["site_id"])
                            / str(row["view_id"])
                            / operator,
                            grid_size=6,
                            cell_budget=6,
                            operator=operator,
                            seed=config.protocol.seed,
                        )
                        for item in dynamic:
                            request = _request(
                                tier="B",
                                sequence=str(item["sequence"]),
                                model_id=model.id,
                                row=row,
                                prompt=prompts["neutral"],
                                image_path=Path(item["image_path"]),
                                grid_size=6,
                                cell_budget=6,
                                fraction=float(item["step"]) / max(int(item["total_steps"]), 1),
                                operator=operator,
                            )
                            _run_one(
                                request,
                                prompts["neutral"],
                                adapter=adapter,
                                model=model,
                                config=config,
                                budget=budget,
                                rows=rows,
                                predictions=predictions,
                                call_cache=call_cache,
                                output=predictions_path,
                            )

        if "C" in config.tiers and model.benchmark_role != "detector":
            for row_index, row in enumerate(tier_c):
                image = config.dataset_dir / str(row["image_path"])
                cave_initials: list[tuple[TracePrediction, PromptTemplate]] = []
                for variant in (
                    "neutral",
                    "false_premise",
                    "uncertainty_aware",
                    "evidence_first",
                    "no_image",
                ):
                    if variant == "neutral":
                        sequence, cave_stage, baseline_kind = "cave", "initial", None
                    elif variant == "false_premise":
                        sequence, cave_stage, baseline_kind = (
                            "false_premise",
                            "initial",
                            None,
                        )
                    elif variant == "no_image":
                        sequence, cave_stage, baseline_kind = "baseline", None, "no_image"
                    else:
                        sequence, cave_stage, baseline_kind = "screening", None, None
                    request = _request(
                        tier="C",
                        sequence=sequence,
                        model_id=model.id,
                        row=row,
                        prompt=prompts[variant],
                        image_path=image,
                        grid_size=6,
                        cell_budget=6,
                        cave_stage=cave_stage,
                        baseline_kind=baseline_kind,
                    )
                    prediction = _run_one(
                        request,
                        prompts[variant],
                        adapter=adapter,
                        model=model,
                        config=config,
                        budget=budget,
                        rows=rows,
                        predictions=predictions,
                        call_cache=call_cache,
                        output=predictions_path,
                    )
                    if variant in {"neutral", "false_premise"}:
                        cave_initials.append((prediction, prompts[variant]))

                unrelated_candidates = [
                    candidate for candidate in clean if candidate["site_id"] != row["site_id"]
                ]
                unrelated_row = (
                    unrelated_candidates[row_index % len(unrelated_candidates)]
                    if unrelated_candidates
                    else row
                )
                controls = materialize_image_controls(
                    image,
                    config.dataset_dir / str(unrelated_row["image_path"]),
                    config.output_dir
                    / "baseline_images"
                    / str(row["site_id"])
                    / str(row["view_id"]),
                    seed=config.protocol.seed + row_index,
                )
                for baseline_kind, control_path in controls.items():
                    request = _request(
                        tier="C",
                        sequence="baseline",
                        model_id=model.id,
                        row=row,
                        prompt=prompts["neutral"],
                        image_path=control_path,
                        grid_size=6,
                        cell_budget=6,
                        baseline_kind=baseline_kind,
                    )
                    _run_one(
                        request,
                        prompts["neutral"],
                        adapter=adapter,
                        model=model,
                        config=config,
                        budget=budget,
                        rows=rows,
                        predictions=predictions,
                        call_cache=call_cache,
                        output=predictions_path,
                    )

                for initial, cave_prompt in cave_initials:
                    if not initial.response or not initial.response.evidence_cells:
                        continue
                    dynamic = materialize_self_sequences(
                        image,
                        initial.response,
                        config.output_dir
                        / "cave_interventions"
                        / model.id
                        / str(row["site_id"])
                        / str(row["view_id"])
                        / cave_prompt.id,
                        grid_size=6,
                        cell_budget=6,
                        seed=config.protocol.seed,
                    )
                    stage_rows = {
                        "necessity": [
                            item for item in dynamic if item["sequence"] == "self_deletion"
                        ][-1],
                        "sufficiency": [
                            item for item in dynamic if item["sequence"] == "self_sufficiency"
                        ][-1],
                        "nuisance": [
                            item for item in dynamic if item["sequence"] == "texture_control"
                        ][-1],
                    }
                    for cave_stage, item in stage_rows.items():
                        request = _request(
                            tier="C",
                            sequence="cave",
                            model_id=model.id,
                            row=row,
                            prompt=cave_prompt,
                            image_path=Path(item["image_path"]),
                            grid_size=6,
                            cell_budget=6,
                            fraction=1,
                            operator="blur",
                            cave_stage=cave_stage,
                        )
                        _run_one(
                            request,
                            cave_prompt,
                            adapter=adapter,
                            model=model,
                            config=config,
                            budget=budget,
                            rows=rows,
                            predictions=predictions,
                            call_cache=call_cache,
                            output=predictions_path,
                        )

        run_sensitivity = "sensitivity" in config.analyses and (
            config.protocol.sensitivity_on_metered_models or not model.metered
        )
        if run_sensitivity and model.benchmark_role != "detector":
            for row in robustness:
                image = config.dataset_dir / str(row["image_path"])
                for grid_size in config.protocol.grid_sizes:
                    for cell_budget in config.protocol.cell_budgets:
                        request = _request(
                            tier="B",
                            sequence="screening",
                            model_id=model.id,
                            row=row,
                            prompt=prompts["neutral"],
                            image_path=image,
                            grid_size=grid_size,
                            cell_budget=cell_budget,
                        )
                        initial = _run_one(
                            request,
                            prompts["neutral"],
                            adapter=adapter,
                            model=model,
                            config=config,
                            budget=budget,
                            rows=rows,
                            predictions=predictions,
                            call_cache=call_cache,
                            output=predictions_path,
                        )
                        if not initial.response or not initial.response.evidence_cells:
                            continue
                        for sweep_operator in PRIMARY_OPERATORS:
                            # V3/V4 always sweep all three operators: the whole
                            # point of the check is cross-operator stability.
                            dynamic = materialize_self_sequences(
                                image,
                                initial.response,
                                config.output_dir
                                / "sensitivity_interventions"
                                / model.id
                                / str(row["site_id"])
                                / str(row["view_id"])
                                / f"g{grid_size}_k{cell_budget}"
                                / sweep_operator,
                                grid_size=grid_size,
                                cell_budget=cell_budget,
                                operator=sweep_operator,
                                seed=config.protocol.seed,
                            )
                            for item in dynamic:
                                dynamic_request = _request(
                                    tier="B",
                                    sequence=str(item["sequence"]),
                                    model_id=model.id,
                                    row=row,
                                    prompt=prompts["neutral"],
                                    image_path=Path(item["image_path"]),
                                    grid_size=grid_size,
                                    cell_budget=cell_budget,
                                    fraction=float(item["step"]) / max(int(item["total_steps"]), 1),
                                    operator=sweep_operator,
                                )
                                _run_one(
                                    dynamic_request,
                                    prompts["neutral"],
                                    adapter=adapter,
                                    model=model,
                                    config=config,
                                    budget=budget,
                                    rows=rows,
                                    predictions=predictions,
                                    call_cache=call_cache,
                                    output=predictions_path,
                                )
    return predictions_path
