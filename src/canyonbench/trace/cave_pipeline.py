"""Tune and apply CAVE from a completed Tier C prediction manifest."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from canyonbench.exceptions import DataValidationError
from canyonbench.io import iter_jsonl, read_json, write_json, write_jsonl
from canyonbench.trace.cave import (
    cave_frontier,
    component_ablation_decisions,
    decide,
    tune_thresholds,
)
from canyonbench.trace.schemas import (
    CaveAblationRecord,
    CaveDecision,
    CaveThresholds,
    TracePrediction,
    TraceResponse,
)


def _truth(dataset_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["site_id"]), str(row["view_id"])): row
        for row in read_json(dataset_dir / "index.json")
        if row.get("variant") == "clean"
    }


def collect_cases(
    predictions_path: Path,
) -> dict[tuple[str, str, str, str], dict[str, TracePrediction]]:
    cases: dict[tuple[str, str, str, str], dict[str, TracePrediction]] = defaultdict(dict)
    for row in iter_jsonl(predictions_path):
        prediction = TracePrediction.model_validate(row)
        request = prediction.request
        if request.sequence not in {"cave", "false_premise"} or request.cave_stage is None:
            continue
        cases[(request.model, request.site_id, request.view_id, request.prompt_id)][
            request.cave_stage
        ] = prediction
    return cases


def _response(case: dict[str, TracePrediction], stage: str) -> TraceResponse | None:
    prediction = case.get(stage)
    return prediction.response if prediction else None


def tune_from_run(
    dataset_dir: Path,
    predictions_path: Path,
    output: Path,
) -> CaveThresholds:
    """Tune from development only; incomplete positive traces count as abstentions."""

    truth = _truth(dataset_dir)
    development = []
    for (_, site_id, view_id, _), case in collect_cases(predictions_path).items():
        metadata = truth.get((site_id, view_id))
        if metadata is None or metadata["split"] != "development":
            continue
        initial = _response(case, "initial")
        if initial is not None:
            development.append(
                (
                    metadata["case_type"] != "negative",
                    initial,
                    _response(case, "necessity"),
                    _response(case, "sufficiency"),
                    _response(case, "nuisance"),
                )
            )
    if not development:
        raise DataValidationError("No complete development-split CAVE traces were found")
    thresholds = tune_thresholds(development)
    write_json(output, thresholds.model_dump(mode="json"))
    frontier_path = output.with_name(f"{output.stem}.frontier.json")
    write_json(
        frontier_path,
        {
            "schema_version": "4.0.0",
            "tuned_split": "development",
            "selection_rule": (
                "maximum balanced accuracy, then coverage, then minimum mean calls per initial"
            ),
            "selected_thresholds": thresholds.model_dump(mode="json"),
            "points": [point.model_dump(mode="json") for point in cave_frontier(development)],
        },
    )
    return thresholds


def apply_from_run(
    dataset_dir: Path,
    predictions_path: Path,
    thresholds: CaveThresholds,
    output: Path,
) -> list[CaveDecision]:
    """Apply frozen development thresholds to all splits without re-tuning."""

    truth = _truth(dataset_dir)
    decisions: list[CaveDecision] = []
    for (model, site_id, view_id, prompt_id), case in sorted(
        collect_cases(predictions_path).items()
    ):
        metadata = truth.get((site_id, view_id))
        if metadata is None:
            continue
        initial = _response(case, "initial")
        if initial is None:
            continue
        request_id = case["initial"].request.request_id
        if initial.answer != "yes":
            decisions.append(
                CaveDecision(
                    request_id=request_id,
                    answer=initial.answer,
                    accepted=True,
                    necessity=0,
                    sufficiency=0,
                    nuisance=0,
                    calls_used=1,
                    reason=(f"non_positive_passthrough:{model}:{prompt_id}:{metadata['split']}"),
                )
            )
            continue
        if not initial.evidence_cells:
            decisions.append(
                CaveDecision(
                    request_id=request_id,
                    answer="abstain",
                    accepted=False,
                    necessity=0,
                    sufficiency=0,
                    nuisance=0,
                    calls_used=1,
                    reason=(f"missing_claimed_evidence:{model}:{prompt_id}:{metadata['split']}"),
                )
            )
            continue
        necessity = _response(case, "necessity")
        sufficiency = _response(case, "sufficiency")
        nuisance = _response(case, "nuisance")
        if necessity is None or sufficiency is None or nuisance is None:
            decisions.append(
                CaveDecision(
                    request_id=request_id,
                    answer="abstain",
                    accepted=False,
                    necessity=0,
                    sufficiency=0,
                    nuisance=1,
                    calls_used=1
                    + sum(response is not None for response in (necessity, sufficiency, nuisance)),
                    reason=f"incomplete_trace:{model}:{prompt_id}:{metadata['split']}",
                )
            )
            continue
        decision = decide(
            request_id,
            initial,
            necessity,
            sufficiency,
            nuisance,
            thresholds,
        )
        decisions.append(
            decision.model_copy(
                update={"reason": (f"{decision.reason}:{model}:{prompt_id}:{metadata['split']}")}
            )
        )
    write_jsonl(output, [decision.model_dump(mode="json") for decision in decisions])
    return decisions


def component_ablations_from_run(
    dataset_dir: Path,
    predictions_path: Path,
    thresholds: CaveThresholds,
    output: Path,
) -> list[CaveAblationRecord]:
    """Apply full and single-component CAVE rules without any test re-tuning."""

    truth = _truth(dataset_dir)
    records: list[CaveAblationRecord] = []
    variants: tuple[Literal["full", "necessity_only", "sufficiency_only", "nuisance_only"], ...] = (
        "full",
        "necessity_only",
        "sufficiency_only",
        "nuisance_only",
    )
    for (model, site_id, view_id, prompt_id), case in sorted(
        collect_cases(predictions_path).items()
    ):
        metadata = truth.get((site_id, view_id))
        if metadata is None:
            continue
        initial = _response(case, "initial")
        if initial is None:
            continue
        request_id = case["initial"].request.request_id
        decisions: dict[str, CaveDecision]
        if initial.answer != "yes":
            passthrough = CaveDecision(
                request_id=request_id,
                answer=initial.answer,
                accepted=True,
                necessity=0,
                sufficiency=0,
                nuisance=0,
                calls_used=1,
                reason="non_positive_passthrough",
            )
            decisions = {variant: passthrough for variant in variants}
        elif not initial.evidence_cells:
            no_evidence = CaveDecision(
                request_id=request_id,
                answer="abstain",
                accepted=False,
                necessity=0,
                sufficiency=0,
                nuisance=0,
                calls_used=1,
                reason="missing_claimed_evidence",
            )
            decisions = {variant: no_evidence for variant in variants}
        else:
            necessity = _response(case, "necessity")
            sufficiency = _response(case, "sufficiency")
            nuisance = _response(case, "nuisance")
            if necessity is None or sufficiency is None or nuisance is None:
                incomplete = CaveDecision(
                    request_id=request_id,
                    answer="abstain",
                    accepted=False,
                    necessity=0,
                    sufficiency=0,
                    nuisance=1,
                    calls_used=1
                    + sum(response is not None for response in (necessity, sufficiency, nuisance)),
                    reason="incomplete_trace",
                )
                decisions = {variant: incomplete for variant in variants}
            else:
                decisions = component_ablation_decisions(
                    request_id,
                    initial,
                    necessity,
                    sufficiency,
                    nuisance,
                    thresholds,
                )
        for variant in variants:
            records.append(
                CaveAblationRecord(
                    model=model,
                    site_id=site_id,
                    view_id=view_id,
                    split=metadata["split"],
                    prompt_id=prompt_id,
                    variant=variant,
                    decision=decisions[variant],
                )
            )
    write_jsonl(output, [record.model_dump(mode="json") for record in records])
    return records
