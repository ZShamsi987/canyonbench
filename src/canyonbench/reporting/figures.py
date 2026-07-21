"""Versioned figure builders for the benchmark paper and model cards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canyonbench.exceptions import DataValidationError


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise DataValidationError(
            "Figure generation requires the analysis extra: pip install 'canyonbench[analysis]'"
        ) from exc
    return plt


def plot_ascent_association(
    frame: pd.DataFrame,
    output: str | Path,
    *,
    outcome: str,
    exposure: str = "alt_m",
    bins: int = 12,
) -> Path:
    required = {outcome, exposure, "segment_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"Ascent figure is missing columns: {missing}")
    clean = frame[[outcome, exposure, "segment_id"]].dropna().sort_values(exposure)
    if len(clean) < bins:
        raise DataValidationError(
            "Ascent figure requires at least one observation per requested bin"
        )
    clean["bin"] = pd.qcut(clean[exposure], bins, duplicates="drop")
    summary = clean.groupby("bin", observed=True).agg(
        x=(exposure, "mean"),
        y=(outcome, "mean"),
        se=(outcome, lambda values: values.std(ddof=1) / np.sqrt(len(values))),
    )
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    for _, segment in clean.groupby("segment_id"):
        axis.plot(segment[exposure], segment[outcome], color="#94A3B8", alpha=0.18, linewidth=0.7)
    axis.errorbar(
        summary["x"],
        summary["y"],
        yerr=1.96 * summary["se"].fillna(0),
        color="#9F2D20",
        marker="o",
        linewidth=1.8,
        capsize=2,
        label="binned mean +/- 95% normal interval",
    )
    axis.set_xlabel(exposure.replace("_", " "))
    axis.set_ylabel(outcome.replace("_", " "))
    axis.legend(frameon=False, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    plt.close(figure)
    return destination


def plot_registration_residuals(
    residuals: pd.DataFrame,
    output: str | Path,
) -> Path:
    required = {"holdout_rmse_m", "threshold_m", "reliable"}
    missing = sorted(required - set(residuals.columns))
    if missing:
        raise DataValidationError(f"Registration figure is missing columns: {missing}")
    ordered = residuals.sort_values("holdout_rmse_m").reset_index(drop=True)
    colors = ["#2E7D32" if bool(value) else "#B3261E" for value in ordered["reliable"]]
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(6.2, 3.4), constrained_layout=True)
    axis.scatter(np.arange(len(ordered)), ordered["holdout_rmse_m"], c=colors, s=20)
    axis.plot(np.arange(len(ordered)), ordered["threshold_m"], color="#334155", linewidth=1)
    axis.set_xlabel("registered frame (sorted)")
    axis.set_ylabel("held-out reprojection RMSE (m)")
    axis.spines[["top", "right"]].set_visible(False)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    plt.close(figure)
    return destination
