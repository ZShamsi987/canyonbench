"""Registered CanyonBench-Trace performance, localization, and causal metrics."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.stats import spearmanr  # type: ignore[import-untyped]
from sklearn.metrics import balanced_accuracy_score, f1_score  # type: ignore[import-untyped]

from canyonbench.exceptions import DataValidationError
from canyonbench.io import iter_jsonl, read_json, write_json
from canyonbench.trace.baselines import deterministic_baselines
from canyonbench.trace.cave import yes_probability
from canyonbench.trace.instruments import operator_agreement, suppression_efficacy
from canyonbench.trace.schemas import (
    CaveAblationRecord,
    CaveDecision,
    TracePrediction,
    TraceResponse,
)
from canyonbench.trace.statistics import (
    benjamini_hochberg,
    fit_mixed_effects,
    hierarchical_bootstrap,
    paired_site_comparison,
)


def _binary_answer(response: TraceResponse | None) -> float:
    if response is None or response.answer == "abstain":
        return float("nan")
    return float(response.answer == "yes")


def _finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def _cell_area(cell: str, grid_size: int, width_px: int, height_px: int) -> int:
    row, column = (int(value) for value in cell.split(","))
    y0, y1 = (
        round(row * height_px / grid_size),
        round((row + 1) * height_px / grid_size),
    )
    x0, x1 = (
        round(column * width_px / grid_size),
        round((column + 1) * width_px / grid_size),
    )
    return (y1 - y0) * (x1 - x0)


def _cell_metrics(
    predicted: set[str],
    truth: set[str],
    grid_size: int,
    *,
    target_pixel_counts: dict[str, int],
    width_px: int,
    height_px: int,
) -> dict[str, float]:
    intersection = len(predicted & truth)
    precision = intersection / len(predicted) if predicted else float(not truth)
    recall = intersection / len(truth) if truth else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    union = len(predicted | truth)
    iou = intersection / union if union else 1
    target_area = sum(target_pixel_counts.values())
    pixel_intersection = sum(target_pixel_counts.get(cell, 0) for cell in predicted)
    selected_area = sum(_cell_area(cell, grid_size, width_px, height_px) for cell in predicted)
    pixel_union = target_area + selected_area - pixel_intersection
    mask_weighted_iou = pixel_intersection / pixel_union if pixel_union else 1.0
    coverage = pixel_intersection / target_area if target_area else float(not predicted)
    area_penalty = 1 - len(predicted) / (grid_size * grid_size)
    return {
        "cell_precision": precision,
        "cell_recall": recall,
        "cell_f1": f1,
        "cell_iou": iou,
        "mask_weighted_iou": mask_weighted_iou,
        "evidence_coverage": coverage,
        "cell_count": float(len(predicted)),
        "area_penalized_f1": f1 * max(0, area_penalty),
    }


def _load_metadata(dataset_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    raw = read_json(dataset_dir / "index.json")
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for row in raw:
        view_directory = (dataset_dir / row["image_path"]).parent
        manifest = read_json(dataset_dir / row["manifest_path"])
        derived = {name: value for name, value in manifest["features"].items()}
        camera = manifest["camera"]
        quality = read_json(view_directory / "quality.json")
        combined = {**row, "derived": derived, "camera": camera, "quality": quality}
        metadata[(str(row["site_id"]), str(row["view_id"]))] = combined
    return metadata


def predictions_frame(dataset_dir: Path, predictions_path: Path) -> pd.DataFrame:
    """Join strict predictions to view truth and derive row-level scores."""

    metadata = _load_metadata(dataset_dir)
    run_manifest_path = predictions_path.parent / "run_manifest.json"
    model_roles: dict[str, str] = {}
    if run_manifest_path.exists():
        run_manifest = read_json(run_manifest_path)
        model_roles = {
            str(model["id"]): str(model.get("benchmark_role") or "unregistered")
            for model in run_manifest.get("models", [])
        }
    rows: list[dict[str, Any]] = []
    for raw in iter_jsonl(predictions_path):
        prediction = TracePrediction.model_validate(raw)
        request = prediction.request
        source = metadata.get((request.site_id, request.view_id))
        if source is None:
            # Dynamic runs may retain the clean view id in all normal cases.
            continue
        target = source["derived"][request.target_class]
        response = prediction.response
        true_positive = bool(target["present"])
        predicted = set(response.evidence_cells if response else [])
        grid_key = f"{request.grid_size}x{request.grid_size}"
        truth = set(target["grid_occupancy"][grid_key])
        localization = _cell_metrics(
            predicted,
            truth,
            request.grid_size,
            target_pixel_counts=target["grid_target_pixel_counts"][grid_key],
            width_px=int(source["camera"]["width_px"]),
            height_px=int(source["camera"]["height_px"]),
        )
        rows.append(
            {
                "request_id": request.request_id,
                "model": request.model,
                "model_class": model_roles.get(request.model, "unregistered"),
                "tier": request.tier,
                "sequence": request.sequence,
                "site_id": request.site_id,
                "view_id": request.view_id,
                "group": source["group"],
                "split": source["split"],
                "target_class": request.target_class,
                "case_type": source["case_type"],
                "true_positive": true_positive,
                "extinction": bool(target["extinction"]),
                "answer": response.answer if response else None,
                "predicted_positive": _binary_answer(response),
                "yes_probability": yes_probability(response),
                "confidence": response.confidence if response else np.nan,
                "format_failure": prediction.format_failure,
                "cache_hit": prediction.cache_hit,
                "attempts": prediction.attempts,
                "latency_s": prediction.latency_s,
                "input_tokens": prediction.input_tokens,
                "output_tokens": prediction.output_tokens,
                "cost_usd": prediction.cost_usd,
                "prompt_id": request.prompt_id,
                "prompt_type": request.prompt_id,
                "operator": request.intervention_operator,
                "intervention_type": request.intervention_operator or request.sequence,
                "fraction": request.intervention_fraction,
                "grid_size": request.grid_size,
                "cell_budget": request.cell_budget,
                "cave_stage": request.cave_stage,
                "baseline_kind": request.baseline_kind,
                "gsd_m_per_px": source["camera"]["gsd_m_per_px"],
                "log_gsd": float(np.log(source["camera"]["gsd_m_per_px"])),
                "altitude_agl_m": source["camera"]["altitude_agl_m"],
                "apparent_width_px": target["median_width_px"],
                "feature_area_fraction": target["area_fraction"],
                "oblique": float(source["camera"]["pitch_deg"] > 0),
                "quality_condition": source["quality"]["degradation"],
                "variant": source["variant"],
                **localization,
            }
        )
    if not rows:
        raise DataValidationError("No predictions joined to the dataset index")
    return pd.DataFrame(rows)


def performance_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    answered = frame.dropna(subset=["predicted_positive"])
    labels = answered["true_positive"].astype(bool)
    predictions = answered["predicted_positive"].astype(bool)
    positives = labels
    negatives = ~labels
    return {
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predictions)) if len(set(labels)) == 2 else np.nan
        ),
        "false_positive_rate": (
            float(np.mean(predictions[negatives])) if negatives.any() else np.nan
        ),
        "false_negative_rate": (
            float(np.mean(~predictions[positives])) if positives.any() else np.nan
        ),
        "abstention_rate": float(frame["predicted_positive"].isna().mean()),
        "macro_f1": (
            float(f1_score(labels, predictions, average="macro")) if len(answered) else np.nan
        ),
        "format_failure_rate": float(frame["format_failure"].mean()),
        "cache_hit_rate": (float(frame["cache_hit"].mean()) if "cache_hit" in frame else 0.0),
        "physical_attempts": (int(frame["attempts"].sum()) if "attempts" in frame else len(frame)),
        "input_tokens": (int(frame["input_tokens"].sum()) if "input_tokens" in frame else 0),
        "output_tokens": (int(frame["output_tokens"].sum()) if "output_tokens" in frame else 0),
        "cost_usd": (float(frame["cost_usd"].sum()) if "cost_usd" in frame else 0.0),
        "median_latency_s": (
            float(frame.loc[~frame["cache_hit"], "latency_s"].median())
            if "cache_hit" in frame and (~frame["cache_hit"]).any()
            else np.nan
        ),
        "n_queries": len(frame),
        "n_sites": int(frame["site_id"].nunique()),
    }


def localization_metrics(frame: pd.DataFrame) -> dict[str, float]:
    names = [
        "cell_precision",
        "cell_recall",
        "cell_f1",
        "cell_iou",
        "mask_weighted_iou",
        "evidence_coverage",
        "cell_count",
        "area_penalized_f1",
    ]
    return {name: float(frame[name].mean()) for name in names}


def _baseline_probabilities(frame: pd.DataFrame) -> pd.Series:
    neutral = (
        frame["prompt_id"].str.contains("neutral", case=False)
        if "prompt_id" in frame
        else pd.Series(True, index=frame.index)
    )
    screening = frame.loc[
        (frame["sequence"] == "screening") & (frame["variant"] == "clean") & neutral
    ]
    keys = [
        *(["_bootstrap_site"] if "_bootstrap_site" in screening else []),
        "site_id",
        "view_id",
        "target_class",
    ]
    return screening.groupby(keys, observed=True)["yes_probability"].mean()


def _auc(rows: pd.DataFrame, baselines: pd.Series) -> float:
    """Mean per-view AUC anchored at that view's unedited Tier-A response."""

    values: list[float] = []
    keys = [
        *(["_bootstrap_site"] if "_bootstrap_site" in rows else []),
        "site_id",
        "view_id",
        "target_class",
    ]
    for key, view_rows in rows.groupby(keys, observed=True):
        baseline = baselines.get(key, np.nan)
        if pd.isna(baseline):
            continue
        curve = (
            view_rows.groupby("fraction", observed=True)["yes_probability"]
            .mean()
            .dropna()
            .sort_index()
        )
        curve = pd.concat([pd.Series({0.0: float(baseline)}), curve])
        curve = curve.groupby(level=0).mean().sort_index()
        if len(curve) < 2:
            continue
        fractions = curve.index.to_numpy(dtype=float)
        probabilities = curve.to_numpy(dtype=float)
        values.append(float(np.trapezoid(probabilities, fractions) / fractions.max()))
    return float(np.mean(values)) if values else float("nan")


def _sufficiency_recovery(rows: pd.DataFrame, baselines: pd.Series) -> float:
    """Normalized recovery of the original answer probability after preservation."""

    if rows.empty:
        return float("nan")
    keys = [
        *(["_bootstrap_site"] if "_bootstrap_site" in rows else []),
        "site_id",
        "view_id",
        "target_class",
    ]
    indexed = rows.set_index(keys)
    original = baselines.reindex(indexed.index).to_numpy(dtype=float)
    recovered = indexed["yes_probability"].to_numpy(dtype=float)
    valid = np.isfinite(original) & np.isfinite(recovered)
    if not valid.any():
        return float("nan")
    return float(np.mean(1 - np.abs(recovered[valid] - original[valid])))


def causal_metrics(frame: pd.DataFrame) -> dict[str, float]:
    baselines = _baseline_probabilities(frame)
    curves = {
        sequence: _auc(frame.loc[frame["sequence"] == sequence], baselines)
        for sequence in (
            "oracle_deletion",
            "distractor_deletion",
            "self_deletion",
            "self_sufficiency",
            "random_control",
            "texture_control",
        )
    }
    ocrs = curves["distractor_deletion"] - curves["oracle_deletion"]
    sen = curves["random_control"] - curves["self_deletion"]
    ses = _sufficiency_recovery(
        frame.loc[frame["sequence"] == "self_sufficiency"],
        baselines,
    )
    efs = float(np.sqrt(max(0, sen) * max(0, ses)))
    return {
        **{f"auc_{name}": value for name, value in curves.items()},
        "ocrs": ocrs,
        "sen": sen,
        "ses": ses,
        "efs": efs,
        "osg": ocrs - efs,
    }


def prompt_prior_gap(frame: pd.DataFrame) -> float:
    negatives = frame.loc[~frame["true_positive"]]
    neutral = negatives.loc[negatives["prompt_id"].str.contains("neutral", case=False)]
    false = negatives.loc[negatives["prompt_id"].str.contains("false", case=False)]
    return float(false["predicted_positive"].mean() - neutral["predicted_positive"].mean())


def extinction_positive_rate(frame: pd.DataFrame) -> float:
    extinction = frame.loc[frame["extinction"]]
    return float(extinction["predicted_positive"].mean()) if len(extinction) else np.nan


def acuity_threshold(frame: pd.DataFrame) -> dict[str, float | int]:
    """Fit P(yes)=logistic(a+b*log(width)); report the P=.5 apparent width."""

    source = frame.loc[
        (frame["apparent_width_px"] > 0) & frame["predicted_positive"].notna()
    ].copy()
    if len(source) < 10 or source["predicted_positive"].nunique() < 2:
        return {"threshold_width_px": np.nan, "slope": np.nan, "n": len(source)}
    x = np.log(source["apparent_width_px"].to_numpy(float))
    y = source["predicted_positive"].to_numpy(float)

    def loss(parameters: np.ndarray) -> float:
        logits = parameters[0] + parameters[1] * x
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        return float(
            -np.sum(y * np.log(probabilities + 1e-12) + (1 - y) * np.log(1 - probabilities + 1e-12))
        )

    result = minimize(loss, np.array([0.0, 1.0]), method="BFGS")
    intercept, slope = result.x
    threshold = np.exp(-intercept / slope) if abs(slope) > 1e-9 else np.nan
    return {
        "threshold_width_px": float(threshold),
        "slope": float(slope),
        "n": len(source),
    }


def selective_risk_curve(frame: pd.DataFrame) -> list[dict[str, float]]:
    answered = frame.dropna(subset=["predicted_positive", "confidence"]).copy()
    answered["correct"] = answered["predicted_positive"].astype(bool) == answered[
        "true_positive"
    ].astype(bool)
    answered = answered.sort_values("confidence", ascending=False)
    rows: list[dict[str, float]] = []
    for coverage in np.linspace(0.1, 1, 10):
        count = max(1, round(len(answered) * coverage))
        selected = answered.head(count)
        selected_negatives = selected.loc[~selected["true_positive"].astype(bool)]
        rows.append(
            {
                "coverage": count / max(len(frame), 1),
                "selective_risk": float(1 - selected["correct"].mean()),
                "selective_false_positive_rate": (
                    float(selected_negatives["predicted_positive"].mean())
                    if len(selected_negatives)
                    else np.nan
                ),
                "confidence_threshold": float(selected["confidence"].min()),
            }
        )
    return rows


def selective_causal_faithfulness_curve(
    frame: pd.DataFrame,
) -> list[dict[str, float]]:
    """Recompute causal faithfulness on the highest-confidence causal cases."""

    dynamic_sequences = {
        "oracle_deletion",
        "distractor_deletion",
        "self_deletion",
        "self_sufficiency",
        "random_control",
        "texture_control",
    }
    key_columns = ["site_id", "view_id", "target_class"]
    dynamic_keys = frame.loc[
        frame["sequence"].isin(dynamic_sequences), key_columns
    ].drop_duplicates()
    if dynamic_keys.empty:
        return []
    screening = frame.loc[
        (frame["tier"] == "A")
        & (frame["sequence"] == "screening")
        & (frame["variant"] == "clean")
        & frame["confidence"].notna(),
        [*key_columns, "confidence"],
    ].drop_duplicates(key_columns)
    screening = screening.merge(dynamic_keys, on=key_columns, how="inner")
    if screening.empty:
        return []
    screening = screening.sort_values("confidence", ascending=False)
    rows: list[dict[str, float]] = []
    for requested_coverage in np.linspace(0.1, 1, 10):
        count = max(1, round(len(screening) * requested_coverage))
        selected = screening.head(count)
        selected_frame = frame.merge(
            selected[key_columns],
            on=key_columns,
            how="inner",
        )
        causal = causal_metrics(selected_frame)
        rows.append(
            {
                "coverage": count / len(screening),
                "confidence_threshold": float(selected["confidence"].min()),
                "ocrs": causal["ocrs"],
                "sen": causal["sen"],
                "ses": causal["ses"],
                "efs": causal["efs"],
                "osg": causal["osg"],
            }
        )
    return rows


def prompt_prior_profile(frame: pd.DataFrame) -> dict[str, Any]:
    """Measure the paired false-premise effect across the resolution lattice."""

    initial = frame.loc[frame["cave_stage"] == "initial"].copy()
    if initial.empty:
        return {"n_pairs": 0, "by_altitude": [], "slope_per_log_gsd": np.nan}
    initial["prompt_condition"] = np.where(
        initial["prompt_id"].str.contains("false", case=False),
        "false_premise",
        "neutral",
    )
    keys = [
        "site_id",
        "view_id",
        "target_class",
        "altitude_agl_m",
        "gsd_m_per_px",
        "apparent_width_px",
    ]
    paired = (
        initial.groupby([*keys, "prompt_condition"], observed=True)["yes_probability"]
        .mean()
        .unstack("prompt_condition")
        .dropna(subset=["neutral", "false_premise"])
        .reset_index()
    )
    if paired.empty:
        return {"n_pairs": 0, "by_altitude": [], "slope_per_log_gsd": np.nan}
    paired["prompt_prior_gap"] = paired["false_premise"] - paired["neutral"]
    by_altitude = [
        {
            "altitude_agl_m": float(str(altitude)),
            "mean_prompt_prior_gap": float(values["prompt_prior_gap"].mean()),
            "n_pairs": len(values),
        }
        for altitude, values in paired.groupby("altitude_agl_m", observed=True)
    ]
    slope = (
        float(
            np.polyfit(
                np.log(paired["gsd_m_per_px"].to_numpy(float)),
                paired["prompt_prior_gap"].to_numpy(float),
                1,
            )[0]
        )
        if paired["gsd_m_per_px"].nunique() >= 2
        else np.nan
    )
    return {
        "n_pairs": len(paired),
        "by_altitude": by_altitude,
        "slope_per_log_gsd": slope,
    }


def localization_faithfulness_association(frame: pd.DataFrame) -> dict[str, float | int]:
    """Relate site-level localization overlap to site-level self-faithfulness."""

    screening = frame.loc[
        (frame["tier"] == "A")
        & (frame["sequence"] == "screening")
        & (frame["variant"] == "clean")
        & frame["true_positive"]
    ]
    localization = screening.groupby("site_id", observed=True)["area_penalized_f1"].mean()
    rows = []
    for site_id, site_rows in frame.groupby("site_id", observed=True):
        causal = causal_metrics(site_rows)
        if site_id in localization and np.isfinite(causal["efs"]):
            rows.append((float(localization[site_id]), causal["efs"]))
    if len(rows) < 3:
        return {"spearman_rho": np.nan, "p_value": np.nan, "n_sites": len(rows)}
    result = spearmanr(
        [row[0] for row in rows],
        [row[1] for row in rows],
    )
    return {
        "spearman_rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_sites": len(rows),
    }


def target_deletion_survival_rate(frame: pd.DataFrame) -> float:
    """Rate of positive answers after complete true-evidence suppression."""

    rows = frame.loc[
        (frame["sequence"] == "oracle_deletion") & (frame["fraction"] == 1) & frame["true_positive"]
    ]
    return float(rows["predicted_positive"].mean()) if len(rows) else np.nan


def _reference_baseline_metrics(dataset_dir: Path) -> dict[str, dict[str, float | int]]:
    frame = deterministic_baselines(read_json(dataset_dir / "index.json"))
    output: dict[str, dict[str, float | int]] = {}
    for name, rows in frame.groupby("baseline", observed=True):
        compatible = pd.DataFrame(
            {
                "true_positive": rows["label"].astype(bool),
                "predicted_positive": rows["prediction"].astype(float),
                "format_failure": False,
                "site_id": rows["site_id"],
            }
        )
        output[str(name)] = performance_metrics(compatible)
    return output


def _cave_metrics(frame: pd.DataFrame, decisions_path: Path) -> dict[str, Any]:
    decisions = {
        decision.request_id: decision
        for decision in (CaveDecision.model_validate(row) for row in iter_jsonl(decisions_path))
    }
    initial = frame.loc[frame["cave_stage"] == "initial"].copy()
    rows: list[dict[str, Any]] = []
    for _, row in initial.iterrows():
        decision = decisions.get(str(row["request_id"]))
        if decision is None:
            continue
        predicted = 1.0 if decision.answer == "yes" else 0.0 if decision.answer == "no" else np.nan
        rows.append(
            {
                "request_id": row["request_id"],
                "model": row["model"],
                "site_id": row["site_id"],
                "true_positive": row["true_positive"],
                "predicted_positive": predicted,
                "format_failure": False,
                "prompt_id": row["prompt_id"],
                "calls_used": decision.calls_used,
            }
        )
    if not rows:
        return {}
    cave_frame = pd.DataFrame(rows)
    output: dict[str, Any] = {}
    for model, model_rows in cave_frame.groupby("model", observed=True):
        model_initial = initial.loc[
            (initial["model"] == model) & initial["request_id"].isin(model_rows["request_id"])
        ]

        def comparison(
            direct: pd.DataFrame,
            verified: pd.DataFrame,
        ) -> dict[str, Any]:
            cave_performance = performance_metrics(verified)
            direct_performance = performance_metrics(direct)
            paired_rows = []
            for condition, values in (("direct", direct), ("cave", verified)):
                for _, value in values.iterrows():
                    predicted = value["predicted_positive"]
                    truth = bool(value["true_positive"])
                    paired_rows.append(
                        {
                            "site_id": value["site_id"],
                            "condition": condition,
                            "unsupported_positive": float(not truth and predicted == 1),
                            "strict_correct": float(
                                pd.notna(predicted) and bool(predicted) == truth
                            ),
                        }
                    )
            paired_frame = pd.DataFrame(paired_rows)
            stage_counts = (
                frame.loc[
                    (frame["model"] == str(direct["model"].iloc[0])) & frame["cave_stage"].notna(),
                    ["site_id", "view_id", "prompt_id", "cave_stage"],
                ]
                .merge(
                    direct[["site_id", "view_id", "prompt_id"]].drop_duplicates(),
                    on=["site_id", "view_id", "prompt_id"],
                    how="inner",
                )
                .groupby(["site_id", "view_id", "prompt_id"], observed=True)
                .size()
            )
            return {
                "direct": direct_performance,
                "cave": cave_performance,
                "false_positive_rate_delta_cave_minus_direct": (
                    cave_performance["false_positive_rate"]
                    - direct_performance["false_positive_rate"]
                ),
                "coverage": 1 - cave_performance["abstention_rate"],
                "mean_calls_per_initial": float(verified["calls_used"].mean()),
                "calls_per_initial": float(verified["calls_used"].mean()),
                "executed_trace_calls_per_initial": (
                    float(stage_counts.mean()) if len(stage_counts) else np.nan
                ),
                "paired_cave_minus_direct_unsupported_positive": _paired(
                    paired_frame,
                    value="unsupported_positive",
                    condition="condition",
                    first="cave",
                    second="direct",
                ),
                "paired_cave_minus_direct_strict_accuracy": _paired(
                    paired_frame,
                    value="strict_correct",
                    condition="condition",
                    first="cave",
                    second="direct",
                ),
            }

        aggregate = comparison(model_initial, model_rows)
        by_prompt: dict[str, Any] = {}
        for prompt_id, verified in model_rows.groupby("prompt_id", observed=True):
            direct = model_initial.loc[model_initial["prompt_id"] == prompt_id]
            by_prompt[str(prompt_id)] = comparison(direct, verified)
        false_prompt_ids = [prompt_id for prompt_id in by_prompt if "false" in prompt_id.lower()]
        false_delta = (
            float(
                np.mean(
                    [
                        by_prompt[prompt_id]["false_positive_rate_delta_cave_minus_direct"]
                        for prompt_id in false_prompt_ids
                    ]
                )
            )
            if false_prompt_ids
            else np.nan
        )
        output[str(model)] = {
            **aggregate,
            "by_prompt": by_prompt,
            "false_premise_compliance_delta": false_delta,
            "false_premise_delta_direction": (
                "CAVE false-positive rate minus direct false-positive rate; "
                "negative values indicate mitigation"
            ),
        }
    return output


def _cave_ablation_metrics(
    dataset_dir: Path,
    ablations_path: Path,
) -> dict[str, Any]:
    truth = {
        (str(row["site_id"]), str(row["view_id"])): row["case_type"] != "negative"
        for row in read_json(dataset_dir / "index.json")
        if row.get("variant") == "clean"
    }
    rows = []
    for raw in iter_jsonl(ablations_path):
        record = CaveAblationRecord.model_validate(raw)
        label = truth.get((record.site_id, record.view_id))
        if label is None:
            continue
        answer = record.decision.answer
        rows.append(
            {
                "model": record.model,
                "variant": record.variant,
                "prompt_id": record.prompt_id,
                "site_id": record.site_id,
                "true_positive": label,
                "predicted_positive": (
                    1.0 if answer == "yes" else 0.0 if answer == "no" else np.nan
                ),
                "format_failure": False,
            }
        )
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    output: dict[str, Any] = {}
    for model, model_rows in frame.groupby("model", observed=True):
        aggregate: dict[str, Any] = {
            str(variant): performance_metrics(values)
            for variant, values in model_rows.groupby("variant", observed=True)
        }
        aggregate["by_prompt"] = {
            str(prompt_id): {
                str(variant): performance_metrics(values)
                for variant, values in prompt_rows.groupby("variant", observed=True)
            }
            for prompt_id, prompt_rows in model_rows.groupby("prompt_id", observed=True)
        }
        output[str(model)] = aggregate
    return output


def _paired(
    frame: pd.DataFrame,
    *,
    value: str,
    condition: str,
    first: str,
    second: str,
) -> dict[str, float | int]:
    if frame.empty or not {first, second}.issubset(set(frame[condition].dropna())):
        return {"n_sites": 0, "mean_difference": np.nan, "p_value": np.nan}
    return paired_site_comparison(
        frame,
        value=value,
        condition=condition,
        first=first,
        second=second,
    )


def _registered_paired_comparisons(frame: pd.DataFrame) -> dict[str, Any]:
    def compare(
        values: pd.DataFrame,
        *,
        value: str,
        condition: str,
        first: str,
        second: str,
    ) -> dict[str, Any]:
        return {
            "pooled_models": _paired(
                values,
                value=value,
                condition=condition,
                first=first,
                second=second,
            ),
            "by_model": {
                str(model): _paired(
                    model_rows,
                    value=value,
                    condition=condition,
                    first=first,
                    second=second,
                )
                for model, model_rows in values.groupby("model", observed=True)
            },
        }

    screening = frame.loc[(frame["tier"] == "A") & (frame["sequence"] == "screening")].copy()
    screening["correct"] = (
        screening["predicted_positive"].astype("boolean")
        == screening["true_positive"].astype("boolean")
    ).astype(float)
    clean_degraded = compare(
        screening.assign(condition=screening["variant"]),
        value="correct",
        condition="condition",
        first="clean",
        second="degraded",
    )
    clean = screening.loc[screening["variant"] == "clean"].copy()
    clean["altitude_condition"] = np.where(
        clean["altitude_agl_m"] == clean["altitude_agl_m"].min(),
        "low",
        np.where(
            clean["altitude_agl_m"] == clean["altitude_agl_m"].max(),
            "high",
            "middle",
        ),
    )
    low_high = compare(
        clean,
        value="correct",
        condition="altitude_condition",
        first="low",
        second="high",
    )

    neutral = frame.loc[
        (frame["sequence"] == "cave")
        & (frame["cave_stage"] == "initial")
        & (~frame["true_positive"])
    ].assign(prompt_condition="neutral")
    false = frame.loc[(frame["sequence"] == "false_premise") & (~frame["true_positive"])].assign(
        prompt_condition="false_premise"
    )
    prompt = compare(
        pd.concat([neutral, false], ignore_index=True),
        value="yes_probability",
        condition="prompt_condition",
        first="false_premise",
        second="neutral",
    )

    baseline = frame.loc[
        (frame["tier"] == "A") & (frame["sequence"] == "screening") & (frame["variant"] == "clean")
    ].assign(edit_condition="original")
    target = frame.loc[(frame["sequence"] == "oracle_deletion") & (frame["fraction"] == 1)].assign(
        edit_condition="target"
    )
    distractor = frame.loc[
        (frame["sequence"] == "distractor_deletion") & (frame["fraction"] == 1)
    ].assign(edit_condition="distractor")
    original_target = compare(
        pd.concat([baseline, target], ignore_index=True),
        value="yes_probability",
        condition="edit_condition",
        first="original",
        second="target",
    )
    target_distractor = compare(
        pd.concat([target, distractor], ignore_index=True),
        value="yes_probability",
        condition="edit_condition",
        first="distractor",
        second="target",
    )
    return {
        "clean_minus_degraded_accuracy": clean_degraded,
        "low_minus_high_altitude_accuracy": low_high,
        "false_premise_minus_neutral_yes_probability": prompt,
        "original_minus_target_deleted_yes_probability": original_target,
        "distractor_minus_target_deleted_yes_probability": target_distractor,
    }


def _primary_endpoint_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    screening = frame.loc[
        (frame["tier"] == "A")
        & (frame["sequence"] == "screening")
        & (frame["variant"] == "clean")
        & (~frame["true_positive"])
    ]
    for (model, site_id), values in screening.groupby(["model", "site_id"], observed=True):
        rows.append(
            {
                "model": model,
                "site_id": site_id,
                "endpoint": "macro_fpr",
                "value": float(values["predicted_positive"].mean()),
            }
        )
    for (model, site_id), values in frame.groupby(["model", "site_id"], observed=True):
        causal = causal_metrics(values)
        for endpoint in ("ocrs", "efs"):
            if np.isfinite(causal[endpoint]):
                rows.append(
                    {
                        "model": model,
                        "site_id": site_id,
                        "endpoint": endpoint,
                        "value": causal[endpoint],
                    }
                )
    return pd.DataFrame(rows)


def registered_statistical_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    """Run the pre-registered paired, mixed-effect, and multiplicity analyses."""

    endpoint_rows = _primary_endpoint_rows(frame)
    comparisons: dict[str, dict[str, float | int]] = {}
    raw_p_values: dict[str, float] = {}
    if not endpoint_rows.empty:
        for endpoint, values in endpoint_rows.groupby("endpoint", observed=True):
            models = sorted(str(model) for model in values["model"].unique())
            for first, second in combinations(models, 2):
                name = f"{endpoint}:{first}__{second}"
                comparison = _paired(
                    values,
                    value="value",
                    condition="model",
                    first=first,
                    second=second,
                )
                comparisons[name] = comparison
                raw_p_values[name] = float(comparison["p_value"])
    adjusted = benjamini_hochberg(raw_p_values)
    for name, value in adjusted.items():
        comparisons[name]["p_value_bh"] = value

    answered = frame.dropna(subset=["predicted_positive"]).copy()
    answered["model_positive"] = answered["predicted_positive"].astype(int)
    mixed: dict[str, Any]
    if (
        len(answered) >= 100
        and answered["site_id"].nunique() >= 10
        and answered["model_positive"].nunique() == 2
    ):
        try:
            mixed = fit_mixed_effects(answered, outcome="model_positive")
        except Exception as exc:
            mixed = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    else:
        mixed = {
            "status": "insufficient_variation_or_sites",
            "n_observations": len(answered),
            "n_sites": int(answered["site_id"].nunique()),
        }
    return {
        "unit_of_independence": "base_site",
        "paired_comparisons": _registered_paired_comparisons(frame),
        "primary_pairwise_comparisons": comparisons,
        "multiplicity_method": "benjamini_hochberg",
        "mixed_effects": mixed,
    }


def _v6_suppression_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    detectors = frame.loc[frame["model_class"] == "detector", "model"].unique()
    for model in detectors:
        model_rows = frame.loc[frame["model"] == model]
        original = model_rows.loc[
            (model_rows["tier"] == "A")
            & (model_rows["sequence"] == "screening")
            & (model_rows["variant"] == "clean")
        ].drop_duplicates(["site_id", "view_id", "target_class"])
        original_map = original.set_index(["site_id", "view_id", "target_class"])["yes_probability"]
        by_operator: dict[str, Any] = {}
        for operator in ("blur", "texture", "frequency"):
            suppressed_rows = model_rows.loc[
                (model_rows["sequence"] == "oracle_deletion")
                & (model_rows["operator"] == operator)
                & (model_rows["fraction"] == 1)
            ]
            suppressed_map = suppressed_rows.groupby(
                ["site_id", "view_id", "target_class"], observed=True
            )["yes_probability"].mean()
            before: list[float] = []
            after: list[float] = []
            labels: list[int] = []
            for key, score in original_map.items():
                label = bool(
                    original.loc[
                        (original["site_id"] == key[0])
                        & (original["view_id"] == key[1])
                        & (original["target_class"] == key[2]),
                        "true_positive",
                    ].iloc[0]
                )
                if label and key not in suppressed_map:
                    continue
                before.append(float(score))
                after.append(float(suppressed_map.get(key, score)))
                labels.append(int(label))
            if before:
                by_operator[operator] = suppression_efficacy(
                    np.asarray(before),
                    np.asarray(after),
                    np.asarray(labels),
                )
        output[str(model)] = by_operator
    return output


def score_trace(
    dataset_dir: Path,
    predictions_path: Path,
    output: Path,
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 2026,
    cave_decisions: Path | None = None,
    cave_ablations: Path | None = None,
    cave_frontier_path: Path | None = None,
) -> dict[str, Any]:
    frame = predictions_frame(dataset_dir, predictions_path)
    screening = frame.loc[
        (frame["tier"] == "A") & (frame["sequence"] == "screening") & (frame["variant"] == "clean")
    ]
    by_model: dict[str, Any] = {}
    for model, model_frame in frame.groupby("model", observed=True):
        model_screening = screening.loc[screening["model"] == model]
        primary = model_frame.loc[
            model_frame["operator"].isin(["blur", "texture", "frequency"])
            | model_frame["operator"].isna()
        ]
        causal_by_operator = {}
        for operator in ("blur", "texture", "frequency"):
            operator_rows = model_frame.loc[model_frame["operator"] == operator]
            operator_input = pd.concat(
                [model_screening, operator_rows],
                ignore_index=True,
            )
            causal_by_operator[operator] = causal_metrics(operator_input)
        # O4 is secondary: it is reported as a registered ablation beside O1-O3 and
        # never folded into the primary causal result.
        inpaint_rows = model_frame.loc[model_frame["operator"] == "inpaint"]
        causal_secondary_inpaint = (
            causal_metrics(pd.concat([model_screening, inpaint_rows], ignore_index=True))
            if len(inpaint_rows)
            else {}
        )
        sensitivity: dict[str, dict[str, float]] = {}
        dynamic_sequences = {
            "self_deletion",
            "self_sufficiency",
            "random_control",
            "texture_control",
        }
        for (grid_size, cell_budget, group_operator), rows in model_frame.loc[
            model_frame["sequence"].isin(dynamic_sequences)
            & model_frame["operator"].isin(["blur", "texture", "frequency"])
        ].groupby(["grid_size", "cell_budget", "operator"], observed=True):
            matching_screen = model_frame.loc[
                (model_frame["sequence"] == "screening")
                & (model_frame["grid_size"] == grid_size)
                & (model_frame["cell_budget"] == cell_budget)
                & model_frame["prompt_id"].str.contains("neutral", case=False)
            ]
            baseline_rows = matching_screen if len(matching_screen) else model_screening
            sensitivity[f"g{grid_size}_k{cell_budget}_{group_operator!s}"] = causal_metrics(
                pd.concat([baseline_rows, rows], ignore_index=True)
            )
        results: dict[str, Any] = {
            "performance": performance_metrics(model_screening),
            "localization": localization_metrics(
                model_screening.loc[model_screening["true_positive"]]
            ),
            "causal": causal_metrics(primary),
            "causal_by_operator": causal_by_operator,
            "causal_secondary_inpaint": causal_secondary_inpaint,
            "grid_k_operator_sensitivity": sensitivity,
            "prompt_prior_gap": prompt_prior_gap(model_frame),
            "extinction_positive_rate": extinction_positive_rate(model_screening),
            "acuity": acuity_threshold(model_screening),
            "acuity_by_class": {
                feature: acuity_threshold(rows)
                for feature, rows in model_screening.groupby("target_class", observed=True)
            },
            "selective_risk": selective_risk_curve(model_screening),
            "selective_causal_faithfulness": selective_causal_faithfulness_curve(model_frame),
            "prompt_prior_profile": prompt_prior_profile(model_frame),
            "localization_faithfulness_association": (
                localization_faithfulness_association(model_frame)
            ),
            "target_deletion_survival_rate": target_deletion_survival_rate(model_frame),
            "image_control_baselines": {
                str(name): performance_metrics(rows)
                for name, rows in model_frame.loc[model_frame["sequence"] == "baseline"].groupby(
                    "baseline_kind", observed=True
                )
            },
            "per_class": {
                feature: performance_metrics(rows)
                for feature, rows in model_screening.groupby("target_class", observed=True)
            },
            "per_group": {
                group: performance_metrics(rows)
                for group, rows in model_screening.groupby("group", observed=True)
            },
            "clean_vs_degraded": {
                variant: performance_metrics(rows)
                for variant, rows in model_frame.loc[
                    (model_frame["tier"] == "A") & (model_frame["sequence"] == "screening")
                ].groupby("variant", observed=True)
            },
        }
        if bootstrap_iterations:
            intervals = {
                "balanced_accuracy": hierarchical_bootstrap(
                    model_screening,
                    lambda data: float(performance_metrics(data)["balanced_accuracy"]),
                    iterations=bootstrap_iterations,
                    seed=seed,
                ),
                "macro_fpr": hierarchical_bootstrap(
                    model_screening,
                    lambda data: float(performance_metrics(data)["false_positive_rate"]),
                    iterations=bootstrap_iterations,
                    seed=seed + 1,
                ),
                "localization_area_penalized_f1": hierarchical_bootstrap(
                    model_screening.loc[model_screening["true_positive"]],
                    lambda data: float(localization_metrics(data)["area_penalized_f1"]),
                    iterations=bootstrap_iterations,
                    seed=seed + 2,
                ),
                "ocrs": hierarchical_bootstrap(
                    primary,
                    lambda data: causal_metrics(data)["ocrs"],
                    iterations=bootstrap_iterations,
                    seed=seed + 3,
                ),
                "efs": hierarchical_bootstrap(
                    primary,
                    lambda data: causal_metrics(data)["efs"],
                    iterations=bootstrap_iterations,
                    seed=seed + 4,
                ),
            }
            if model_screening["extinction"].any():
                intervals["extinction_positive_rate"] = hierarchical_bootstrap(
                    model_screening.loc[model_screening["extinction"]],
                    extinction_positive_rate,
                    iterations=bootstrap_iterations,
                    seed=seed + 5,
                )
            prompt_rows = model_frame.loc[model_frame["cave_stage"] == "initial"]
            if len(prompt_rows):
                intervals["prompt_prior_gap"] = hierarchical_bootstrap(
                    prompt_rows,
                    prompt_prior_gap,
                    iterations=bootstrap_iterations,
                    seed=seed + 6,
                )
            target_rows = model_frame.loc[
                (model_frame["sequence"] == "oracle_deletion")
                & (model_frame["fraction"] == 1)
                & model_frame["true_positive"]
            ]
            if len(target_rows):
                intervals["target_deletion_survival_rate"] = hierarchical_bootstrap(
                    target_rows,
                    target_deletion_survival_rate,
                    iterations=bootstrap_iterations,
                    seed=seed + 7,
                )
            results["confidence_intervals"] = intervals
            results["balanced_accuracy_ci"] = intervals["balanced_accuracy"]
        by_model[str(model)] = results
    agreement_rows = []
    for model, values in by_model.items():
        for operator, causal in values["causal_by_operator"].items():
            for metric in ("ocrs", "sen", "ses", "efs", "osg"):
                agreement_rows.append(
                    {
                        "model": model,
                        "operator": operator,
                        "metric": metric,
                        "value": causal[metric],
                    }
                )
    agreement_frame = pd.DataFrame(agreement_rows)
    agreement = {
        metric: operator_agreement(
            agreement_frame.loc[agreement_frame["metric"] == metric],
            metric="value",
        )
        for metric in ("ocrs", "sen", "ses", "efs", "osg")
    }
    model_class_summary: dict[str, Any] = {}
    for model_class, class_rows in frame.groupby("model_class", observed=True):
        models = sorted(str(value) for value in class_rows["model"].unique())
        class_model_values = [by_model[model] for model in models]
        model_class_summary[str(model_class)] = {
            "n_models": len(models),
            "models": models,
            "mean_balanced_accuracy": _finite_mean(
                [values["performance"]["balanced_accuracy"] for values in class_model_values]
            ),
            "mean_area_penalized_localization_f1": _finite_mean(
                [values["localization"]["area_penalized_f1"] for values in class_model_values]
            ),
            "mean_efs": _finite_mean([values["causal"]["efs"] for values in class_model_values]),
        }
    result = {
        "schema_version": "4.0.0",
        "models": by_model,
        "query_count": len(frame),
        "independent_site_count": int(frame["site_id"].nunique()),
        "reference_baselines": _reference_baseline_metrics(dataset_dir),
        "operator_rank_agreement": agreement,
        "model_class_summary": model_class_summary,
        "registered_statistics": registered_statistical_analysis(frame),
        "suppression_efficacy_v6": _v6_suppression_metrics(frame),
        "cave": (
            _cave_metrics(frame, cave_decisions)
            if cave_decisions is not None and cave_decisions.exists()
            else {}
        ),
        "cave_component_ablations": (
            _cave_ablation_metrics(dataset_dir, cave_ablations)
            if cave_ablations is not None and cave_ablations.exists()
            else {}
        ),
        "cave_frontier": (
            read_json(cave_frontier_path)
            if cave_frontier_path is not None and cave_frontier_path.exists()
            else {}
        ),
    }
    write_json(output, result)
    frame.to_csv(output.with_suffix(".rows.csv"), index=False)
    return result
