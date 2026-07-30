"""Result-independent validation instruments V1-V6."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, roc_auc_score  # type: ignore[import-untyped]
from sklearn.model_selection import GroupKFold, cross_val_predict  # type: ignore[import-untyped]

from canyonbench.io import read_json
from canyonbench.trace.degradation import measure_quality


def synthetic_road_insert(
    image: np.ndarray,
    *,
    apparent_width_px: float,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """V1: insert a photometrically matched, anti-aliased road-like segment."""

    generator = np.random.default_rng(seed)
    height, width = image.shape[:2]
    start = (int(width * 0.15), int(generator.uniform(0.25, 0.75) * height))
    end = (int(width * 0.85), int(generator.uniform(0.25, 0.75) * height))
    mask_hi = np.zeros((height * 4, width * 4), np.uint8)
    cv2.line(
        mask_hi,
        (start[0] * 4, start[1] * 4),
        (end[0] * 4, end[1] * 4),
        255,
        max(1, round(apparent_width_px * 4)),
        cv2.LINE_AA,
    )
    mask = cv2.resize(mask_hi, (width, height), interpolation=cv2.INTER_AREA) / 255
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    road_luminance = float(np.quantile(gray, 0.6))
    colour = np.full_like(image, road_luminance)
    alpha = mask[..., None] * 0.75
    inserted = np.clip(image * (1 - alpha) + colour * alpha, 0, 255).astype(np.uint8)
    return inserted, (mask > 0.2).astype(np.uint8)


def build_synthetic_width_series(
    image_path: Path,
    output_dir: Path,
    widths: list[float] | None = None,
) -> list[dict[str, Any]]:
    widths = widths or [0.5, 0.75, 1, 1.5, 2, 3, 4, 6]
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, width in enumerate(widths):
        inserted, mask = synthetic_road_insert(image, apparent_width_px=width, seed=2026 + index)
        path = output_dir / f"synthetic_road_w{str(width).replace('.', 'p')}.png"
        mask_path = output_dir / f"synthetic_road_w{str(width).replace('.', 'p')}__mask.png"
        Image.fromarray(inserted).save(path)
        Image.fromarray(mask * 255).save(mask_path)
        rows.append(
            {
                "apparent_width_px": width,
                "image_path": str(path),
                "mask_path": str(mask_path),
            }
        )
    return rows


def _image_features(path: Path) -> list[float]:
    with Image.open(path) as source:
        image = np.asarray(source.convert("RGB"))
    values = measure_quality(image)
    return [
        values.laplacian_variance,
        values.dark_channel_mean,
        values.luminance_mean,
        values.saturation_mean,
        values.contrast_std,
        values.jpeg_blockiness,
        values.horizon_frequency,
        values.obstruction_fraction,
    ]


def edit_detectability_audit(
    rows: list[dict[str, str]],
    *,
    groups: list[str],
) -> dict[str, float | int]:
    """V2: group-cross-validated classifier for untouched/target/distractor edits."""

    features = np.asarray([_image_features(Path(row["image_path"])) for row in rows])
    labels = np.asarray([row["label"] for row in rows])
    if len(set(groups)) < 2 or len(set(labels)) < 2:
        raise ValueError("edit audit requires at least two groups and two labels")
    splits = min(5, len(set(groups)))
    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, random_state=2026, class_weight="balanced"
    )
    prediction = cross_val_predict(
        model,
        features,
        labels,
        groups=np.asarray(groups),
        cv=GroupKFold(n_splits=splits),
        method="predict",
    )
    target_control = np.isin(labels, ["target", "distractor"])
    return {
        "accuracy_all": float(accuracy_score(labels, prediction)),
        "accuracy_target_vs_distractor": float(
            accuracy_score(labels[target_control], prediction[target_control])
        ),
        "n_images": len(rows),
        "n_sites": len(set(groups)),
    }


def dataset_edit_detectability_audit(dataset_dir: Path) -> dict[str, dict[str, float | int]]:
    """V2: construct untouched/target/distractor examples from frozen bundles."""

    index = [row for row in read_json(dataset_dir / "index.json") if row["variant"] == "clean"]
    output: dict[str, dict[str, float | int]] = {}
    for operator in ("blur", "texture", "frequency", "inpaint"):
        rows: list[dict[str, str]] = []
        groups: list[str] = []
        for row in index:
            if row["case_type"] == "negative":
                continue
            view_dir = (dataset_dir / str(row["image_path"])).parent
            manifest_path = view_dir / "interventions" / "manifest.json"
            if not manifest_path.exists():
                continue
            records = read_json(manifest_path)
            candidates = [
                record
                for record in records
                if record["operator"] == operator
                and record["fraction"] == 1
                and record.get("accepted")
            ]
            if len(candidates) != 2:
                continue
            rows.append({"image_path": str(view_dir / "rgb.png"), "label": "untouched"})
            groups.append(str(row["site_id"]))
            for record in candidates:
                path = Path(str(record["image_path"]))
                if not path.is_absolute():
                    path = view_dir / path
                rows.append(
                    {
                        "image_path": str(path),
                        "label": (
                            "target" if record["sequence"] == "oracle_deletion" else "distractor"
                        ),
                    }
                )
                groups.append(str(row["site_id"]))
        output[operator] = edit_detectability_audit(rows, groups=groups)
    return output


def operator_agreement(frame: pd.DataFrame, *, metric: str) -> dict[str, float]:
    """V3: Spearman model-ranking agreement across O1-O3."""

    pivot = frame.pivot_table(index="model", columns="operator", values=metric, aggfunc="mean")
    pivot = pivot.reindex(columns=["blur", "texture", "frequency"])
    output: dict[str, float] = {}
    for first, second in combinations(["blur", "texture", "frequency"], 2):
        values = pivot[[first, second]].dropna()
        output[f"{first}__{second}"] = (
            float(spearmanr(values[first], values[second]).statistic)
            if len(values) >= 3
            else float("nan")
        )
    return output


def suppression_efficacy(
    original_scores: np.ndarray, suppressed_scores: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """V6: confirm the independent detector's target signal is removed."""

    if not (original_scores.shape == suppressed_scores.shape == labels.shape):
        raise ValueError("detector arrays must have identical shapes")
    classes = np.unique(labels)
    return {
        "mean_signal_before": float(np.mean(original_scores)),
        "mean_signal_after": float(np.mean(suppressed_scores)),
        "mean_signal_removed": float(np.mean(original_scores - suppressed_scores)),
        "auc_before": (
            float(roc_auc_score(labels, original_scores)) if len(classes) == 2 else float("nan")
        ),
        "auc_after": (
            float(roc_auc_score(labels, suppressed_scores)) if len(classes) == 2 else float("nan")
        ),
    }
