"""Training-free Causal Visual Evidence Verification (CAVE)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from canyonbench.trace.schemas import (
    CaveDecision,
    CaveFrontierPoint,
    CaveThresholds,
    TraceResponse,
)


def yes_probability(response: TraceResponse | None) -> float:
    if response is None or response.answer == "abstain":
        return 0.5
    confidence = response.confidence / 100
    return confidence if response.answer == "yes" else 1 - confidence


def cave_scores(
    initial: TraceResponse,
    necessity: TraceResponse,
    sufficiency: TraceResponse,
    nuisance: TraceResponse,
) -> tuple[float, float, float]:
    initial_probability = yes_probability(initial)
    necessity_score = initial_probability - yes_probability(necessity)
    sufficiency_score = 1 - abs(initial_probability - yes_probability(sufficiency))
    nuisance_score = abs(initial_probability - yes_probability(nuisance))
    return (
        float(np.clip(necessity_score, -1, 1)),
        float(np.clip(sufficiency_score, -1, 1)),
        float(np.clip(nuisance_score, 0, 1)),
    )


def decide(
    request_id: str,
    initial: TraceResponse,
    necessity: TraceResponse,
    sufficiency: TraceResponse,
    nuisance: TraceResponse,
    thresholds: CaveThresholds,
) -> CaveDecision:
    """Retain non-positive decisions; verify a positive or convert it to abstention."""

    necessity_score, sufficiency_score, nuisance_score = cave_scores(
        initial, necessity, sufficiency, nuisance
    )
    if initial.answer != "yes":
        return CaveDecision(
            request_id=request_id,
            answer=initial.answer,
            accepted=True,
            necessity=necessity_score,
            sufficiency=sufficiency_score,
            nuisance=nuisance_score,
            calls_used=1,
            reason="non_positive_passthrough",
        )
    checks = (
        ("confidence", initial.confidence >= thresholds.confidence_min, 1),
        ("necessity", necessity_score >= thresholds.necessity_min, 2),
        ("sufficiency", sufficiency_score >= thresholds.sufficiency_min, 3),
        ("nuisance", nuisance_score <= thresholds.nuisance_max, 4),
    )
    failed = next(
        ((name, calls_used) for name, passed, calls_used in checks if not passed),
        None,
    )
    if failed is None:
        return CaveDecision(
            request_id=request_id,
            answer="yes",
            accepted=True,
            necessity=necessity_score,
            sufficiency=sufficiency_score,
            nuisance=nuisance_score,
            calls_used=4,
            reason="accepted",
        )
    failed_name, calls_used = failed
    return CaveDecision(
        request_id=request_id,
        answer="abstain",
        accepted=False,
        necessity=necessity_score,
        sufficiency=sufficiency_score,
        nuisance=nuisance_score,
        calls_used=calls_used,
        reason=f"failed_{failed_name}",
    )


DevelopmentRow = tuple[
    bool,
    TraceResponse,
    TraceResponse | None,
    TraceResponse | None,
    TraceResponse | None,
]


def _evaluate_thresholds(
    rows: list[DevelopmentRow],
    thresholds: CaveThresholds,
) -> CaveFrontierPoint:
    predictions: list[bool | None] = []
    labels: list[bool] = []
    calls: list[int] = []
    for index, (label, initial, necessity, sufficiency, nuisance) in enumerate(rows):
        answer: str
        if initial.answer != "yes":
            answer = initial.answer
            calls_used = 1
        elif not initial.evidence_cells:
            answer = "abstain"
            calls_used = 1
        elif necessity is None or sufficiency is None or nuisance is None:
            answer = "abstain"
            calls_used = 1 + sum(
                response is not None for response in (necessity, sufficiency, nuisance)
            )
        else:
            decision = decide(
                f"{index:064x}",
                initial,
                necessity,
                sufficiency,
                nuisance,
                thresholds,
            )
            answer = decision.answer
            calls_used = decision.calls_used
        predictions.append(True if answer == "yes" else False if answer == "no" else None)
        labels.append(label)
        calls.append(calls_used)

    answered = [
        (label, prediction)
        for label, prediction in zip(labels, predictions, strict=True)
        if prediction is not None
    ]
    positives = [pair for pair in answered if pair[0]]
    negatives = [pair for pair in answered if not pair[0]]
    tpr = float(np.mean([prediction for _, prediction in positives])) if positives else 0.0
    tnr = float(np.mean([not prediction for _, prediction in negatives])) if negatives else 0.0
    false_positive_rate = (
        float(np.mean([prediction for _, prediction in negatives])) if negatives else None
    )
    false_negative_rate = (
        float(np.mean([not prediction for _, prediction in positives])) if positives else None
    )
    coverage = len(answered) / len(rows)
    total_calls = sum(calls)
    return CaveFrontierPoint(
        thresholds=thresholds,
        balanced_accuracy=(tpr + tnr) / 2,
        coverage=coverage,
        abstention_rate=1 - coverage,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        mean_calls_per_initial=total_calls / len(rows),
        calls_per_answered_case=(total_calls / len(answered) if answered else None),
        n_cases=len(rows),
        n_answered=len(answered),
    )


def _threshold_grid() -> Iterable[CaveThresholds]:
    for necessity_min in np.linspace(0, 0.5, 6):
        for sufficiency_min in np.linspace(0.5, 1, 6):
            for nuisance_max in np.linspace(0.05, 0.4, 8):
                for confidence_min in (50, 60, 70, 80, 90):
                    yield CaveThresholds(
                        necessity_min=float(necessity_min),
                        sufficiency_min=float(sufficiency_min),
                        nuisance_max=float(nuisance_max),
                        confidence_min=confidence_min,
                    )


def cave_frontier(
    development_rows: Iterable[DevelopmentRow],
) -> list[CaveFrontierPoint]:
    """Return non-dominated development policies over reliability, coverage, and calls."""

    rows = list(development_rows)
    if not rows:
        raise ValueError("development rows are required to tune CAVE")
    candidates = [_evaluate_thresholds(rows, thresholds) for thresholds in _threshold_grid()]
    unique: dict[tuple[float, ...], CaveFrontierPoint] = {}
    for candidate in candidates:
        key = (
            round(candidate.balanced_accuracy, 12),
            round(candidate.coverage, 12),
            round(candidate.mean_calls_per_initial, 12),
            round(
                candidate.false_positive_rate if candidate.false_positive_rate is not None else -1,
                12,
            ),
            round(
                candidate.false_negative_rate if candidate.false_negative_rate is not None else -1,
                12,
            ),
        )
        unique.setdefault(key, candidate)
    points = list(unique.values())
    frontier = [
        candidate
        for candidate in points
        if not any(
            other.balanced_accuracy >= candidate.balanced_accuracy
            and other.coverage >= candidate.coverage
            and other.mean_calls_per_initial <= candidate.mean_calls_per_initial
            and (
                other.balanced_accuracy > candidate.balanced_accuracy
                or other.coverage > candidate.coverage
                or other.mean_calls_per_initial < candidate.mean_calls_per_initial
            )
            for other in points
        )
    ]
    return sorted(
        frontier,
        key=lambda point: (
            point.coverage,
            point.balanced_accuracy,
            -point.mean_calls_per_initial,
        ),
    )


def tune_thresholds(
    development_rows: Iterable[DevelopmentRow],
) -> CaveThresholds:
    """Choose the registered dev policy: accuracy, then coverage, then lower cost."""

    frontier = cave_frontier(development_rows)
    best = max(
        frontier,
        key=lambda point: (
            point.balanced_accuracy,
            point.coverage,
            -point.mean_calls_per_initial,
        ),
    )
    return best.thresholds


def component_ablation_decisions(
    request_id: str,
    initial: TraceResponse,
    necessity: TraceResponse,
    sufficiency: TraceResponse,
    nuisance: TraceResponse,
    thresholds: CaveThresholds,
) -> dict[str, CaveDecision]:
    """Full CAVE beside necessity-only, sufficiency-only, and nuisance-only."""

    variants = {
        "full": thresholds,
        "necessity_only": thresholds.model_copy(
            update={"sufficiency_min": -1.0, "nuisance_max": 1.0}
        ),
        "sufficiency_only": thresholds.model_copy(
            update={"necessity_min": -1.0, "nuisance_max": 1.0}
        ),
        "nuisance_only": thresholds.model_copy(
            update={"necessity_min": -1.0, "sufficiency_min": -1.0}
        ),
    }
    return {
        name: decide(
            request_id,
            initial,
            necessity,
            sufficiency,
            nuisance,
            variant,
        )
        for name, variant in variants.items()
    }
