"""Deterministic primary benchmark metrics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import pandas as pd

from canyonbench.constants import FEATURES
from canyonbench.eval.bootstrap import segment_bootstrap
from canyonbench.exceptions import DataValidationError
from canyonbench.io import iter_jsonl


def safe_divide(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def classification_counts(truth: Iterable[bool], prediction: Iterable[bool]) -> dict[str, int]:
    pairs = list(zip(truth, prediction, strict=True))
    return {
        "tp": sum(actual and predicted for actual, predicted in pairs),
        "fp": sum(not actual and predicted for actual, predicted in pairs),
        "tn": sum(not actual and not predicted for actual, predicted in pairs),
        "fn": sum(actual and not predicted for actual, predicted in pairs),
    }


def presence_summary(frame: pd.DataFrame) -> dict[str, Any]:
    per_feature: dict[str, Any] = {}
    total_valid = total = exact = 0
    for feature in FEATURES:
        truth = frame[feature]
        prediction = frame["response"].map(
            lambda value, current_feature=feature: (
                value.get(current_feature) if isinstance(value, dict) else None
            )
        )
        scorable = truth.isin(["yes", "no"])
        valid = prediction.isin(["yes", "no"]) & scorable
        actual = truth.loc[valid].eq("yes").tolist()
        predicted = prediction.loc[valid].eq("yes").tolist()
        counts = classification_counts(actual, predicted)
        precision = safe_divide(counts["tp"], counts["tp"] + counts["fp"])
        recall = safe_divide(counts["tp"], counts["tp"] + counts["fn"])
        f1 = (
            safe_divide(2 * precision * recall, precision + recall)
            if precision is not None and recall is not None
            else None
        )
        feature_total = int(scorable.sum())
        feature_exact = int(((truth == prediction) & scorable).sum())
        per_feature[feature] = {
            **counts,
            "n": feature_total,
            "n_valid": int(valid.sum()),
            "invalid_rate": safe_divide(feature_total - int(valid.sum()), feature_total),
            "exact_accuracy": safe_divide(feature_exact, feature_total),
            "false_positive_rate": safe_divide(counts["fp"], counts["fp"] + counts["tn"]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        total += feature_total
        total_valid += int(valid.sum())
        exact += feature_exact
    return {
        "per_feature": per_feature,
        "macro_f1": float(
            np.mean([value["f1"] for value in per_feature.values() if value["f1"] is not None])
        ),
        "micro_exact_accuracy": safe_divide(exact, total),
        "invalid_rate": safe_divide(total - total_valid, total),
    }


def vegetation_rows(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["prediction_percent"] = output["response"].map(
        lambda value: value.get("percent") if isinstance(value, dict) else np.nan
    )
    output["truth_percent"] = pd.to_numeric(output["vegetation_fraction"], errors="coerce") * 100
    output["error"] = output["prediction_percent"] - output["truth_percent"]
    return output


def vegetation_summary(frame: pd.DataFrame) -> dict[str, Any]:
    values = vegetation_rows(frame)
    valid = values["error"].notna()
    errors = values.loc[valid, "error"].astype(float)
    return {
        "n": len(values),
        "n_valid": int(valid.sum()),
        "invalid_rate": safe_divide(len(values) - int(valid.sum()), len(values)),
        "signed_error": float(errors.mean()) if len(errors) else None,
        "mae": float(errors.abs().mean()) if len(errors) else None,
        "overestimation_rate": float((errors > 0).mean()) if len(errors) else None,
    }


def grounding_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw_record in frame.to_dict(orient="records"):
        record = cast(dict[str, Any], raw_record)
        truth_value = record.get("vegetation_cells")
        if isinstance(truth_value, str):
            import ast

            truth_value = ast.literal_eval(truth_value)
        truth = set(truth_value or [])
        response = record.get("response")
        prediction = set(response.get("cells", [])) if isinstance(response, dict) else set()
        valid = isinstance(response, dict)
        points = response.get("points") if isinstance(response, dict) else None
        point_cells = (
            [f"{min(int(float(y) * 4), 3)},{min(int(float(x) * 4), 3)}" for x, y in points]
            if points
            else []
        )
        counts = classification_counts(
            [cell in truth for cell in (f"{r},{c}" for r in range(4) for c in range(4))],
            [cell in prediction for cell in (f"{r},{c}" for r in range(4) for c in range(4))],
        )
        rows.append(
            {
                **record,
                **counts,
                "valid_response": valid,
                "point_count": len(point_cells),
                "point_hits": sum(cell in truth for cell in point_cells),
                "truth_cells_pointed": len(set(point_cells) & truth),
                "truth_cell_count": len(truth),
            }
        )
    return pd.DataFrame(rows)


def grounding_summary(frame: pd.DataFrame) -> dict[str, Any]:
    values = grounding_rows(frame)
    tp, fp, fn = (int(values[column].sum()) for column in ("tp", "fp", "fn"))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    point_count = int(values["point_count"].sum())
    truth_cell_count = int(values["truth_cell_count"].sum())
    return {
        "n": len(values),
        "invalid_rate": float((~values["valid_response"]).mean()) if len(values) else None,
        "region_precision": precision,
        "region_recall": recall,
        "region_f1": (
            safe_divide(2 * precision * recall, precision + recall)
            if precision is not None and recall is not None
            else None
        ),
        "point_precision": safe_divide(int(values["point_hits"].sum()), point_count),
        "point_cell_recall": safe_divide(
            int(values["truth_cells_pointed"].sum()), truth_cell_count
        ),
    }


def false_premise_summary(frame: pd.DataFrame) -> dict[str, Any]:
    values = frame.copy()
    values["complied"] = values["response"].map(
        lambda response: response.get("premise_correct") if isinstance(response, dict) else np.nan
    )
    variants: dict[str, Any] = {}
    for variant, group in values.groupby("variant"):
        valid = group["complied"].isin([True, False])
        variants[str(variant)] = {
            "n": len(group),
            "n_valid": int(valid.sum()),
            "compliance_rate": float(group.loc[valid, "complied"].mean()) if valid.any() else None,
        }
    delta = None
    if "leading" in variants and "evidence_first" in variants:
        leading = variants["leading"]["compliance_rate"]
        mitigation = variants["evidence_first"]["compliance_rate"]
        if leading is not None and mitigation is not None:
            delta = leading - mitigation
    return {"variants": variants, "mitigation_delta": delta}


def _attach_predictions(frames: pd.DataFrame, predictions: list[dict[str, Any]]) -> pd.DataFrame:
    prediction_frame = pd.DataFrame(predictions)
    if prediction_frame.empty:
        raise DataValidationError("Prediction file is empty")
    joined = prediction_frame.merge(
        frames,
        on="image",
        how="left",
        validate="many_to_one",
        suffixes=("", "_gt"),
    )
    segment_column = "segment_id_gt" if "segment_id_gt" in joined else "segment_id"
    missing_truth = joined[segment_column].isna()
    if missing_truth.any():
        images = joined.loc[missing_truth, "image"].head(5).tolist()
        raise DataValidationError(f"Predictions refer to images outside the release: {images}")
    return joined


def score_benchmark(
    frames: pd.DataFrame,
    predictions: list[dict[str, Any]],
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    joined = _attach_predictions(frames, predictions)
    output: dict[str, Any] = {
        "raw_frame_count": int(frames.shape[0]),
        "effective_segment_count": int(frames["segment_id"].nunique()),
        "models": {},
    }
    for model, model_rows in joined.groupby("model"):
        model_output: dict[str, Any] = {}
        for (probe, variant), group in model_rows.groupby(["probe", "variant"]):
            key = f"{probe}:{variant}"
            if probe == "presence":
                summary = presence_summary(group)
                ci = segment_bootstrap(
                    group,
                    lambda sample: float(presence_summary(sample)["macro_f1"]),
                    iterations=bootstrap_iterations,
                    seed=seed,
                )
                summary["macro_f1_ci"] = ci
            elif probe == "vegetation":
                summary = vegetation_summary(group)
                ci = segment_bootstrap(
                    group,
                    lambda sample: float(vegetation_rows(sample)["error"].abs().mean()),
                    iterations=bootstrap_iterations,
                    seed=seed,
                )
                summary["mae_ci"] = ci
            elif probe == "grounding":
                summary = grounding_summary(group)
                ci = segment_bootstrap(
                    group,
                    lambda sample: float(grounding_summary(sample)["region_f1"] or 0),
                    iterations=bootstrap_iterations,
                    seed=seed,
                )
                summary["region_f1_ci"] = ci
            elif probe == "false_premise":
                summary = false_premise_summary(group)
            else:
                summary = {"n": len(group), "note": "caption scoring is secondary"}
            model_output[key] = summary
        false_rows = model_rows.loc[model_rows["probe"] == "false_premise"]
        if not false_rows.empty:
            model_output["false_premise:combined"] = false_premise_summary(false_rows)
        output["models"][str(model)] = model_output
    return output


def load_and_score(
    frames_path: str,
    predictions_path: str,
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    return score_benchmark(
        pd.read_csv(frames_path),
        list(iter_jsonl(predictions_path)),
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
