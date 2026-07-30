"""Site-aware uncertainty, paired tests, multiplicity, and mixed effects."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import-untyped]


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    *,
    iterations: int = 2000,
    seed: int = 2026,
) -> dict[str, float | int]:
    """Bootstrap groups, then sites within group; never sample individual renders."""

    required = {"site_id", "group"}
    if not required.issubset(frame):
        raise ValueError(f"hierarchical bootstrap requires {sorted(required)}")
    generator = np.random.default_rng(seed)
    groups = sorted(frame["group"].dropna().unique())
    if not groups:
        raise ValueError("no geographic groups to bootstrap")
    estimates: list[float] = []
    for _ in range(iterations):
        sampled_groups = generator.choice(groups, size=len(groups), replace=True)
        pieces: list[pd.DataFrame] = []
        for group_index, group in enumerate(sampled_groups):
            subset = frame.loc[frame["group"] == group]
            sites = sorted(subset["site_id"].unique())
            sampled_sites = generator.choice(sites, size=len(sites), replace=True)
            for site_index, site in enumerate(sampled_sites):
                piece = subset.loc[subset["site_id"] == site].copy()
                piece["_bootstrap_site"] = f"{group_index}:{site_index}:{site}"
                pieces.append(piece)
        value = metric(pd.concat(pieces, ignore_index=True))
        if np.isfinite(value):
            estimates.append(float(value))
    point = float(metric(frame))
    if not estimates:
        return {
            "estimate": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_iterations": 0,
            "independent_sites": int(frame["site_id"].nunique()),
            "geographic_groups": len(groups),
        }
    return {
        "estimate": point,
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_iterations": len(estimates),
        "independent_sites": int(frame["site_id"].nunique()),
        "geographic_groups": len(groups),
    }


def paired_site_comparison(
    frame: pd.DataFrame,
    *,
    value: str,
    condition: str,
    first: str,
    second: str,
) -> dict[str, float | int]:
    pivot = (
        frame.groupby(["site_id", condition], observed=True)[value]
        .mean()
        .unstack(condition)
        .dropna(subset=[first, second])
    )
    if len(pivot) < 2:
        return {"n_sites": len(pivot), "mean_difference": float("nan"), "p_value": float("nan")}
    differences = pivot[first] - pivot[second]
    result = stats.ttest_rel(pivot[first], pivot[second])
    return {
        "n_sites": len(pivot),
        "mean_difference": float(differences.mean()),
        "standard_error": float(stats.sem(differences)),
        "p_value": float(result.pvalue),
    }


def benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
    finite = [(name, value) for name, value in p_values.items() if np.isfinite(value)]
    ordered = sorted(finite, key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {name: float("nan") for name in p_values}
    running = 1.0
    for rank, (name, value) in reversed(list(enumerate(ordered, start=1))):
        running = min(running, value * count / rank)
        adjusted[name] = float(min(1, running))
    return adjusted


def fit_mixed_effects(frame: pd.DataFrame, *, outcome: str) -> dict[str, Any]:
    """Fit the preregistered site/group/class mixed logistic model."""

    import statsmodels.api as sm  # type: ignore[import-untyped]

    fixed = [
        column
        for column in (
            "log_gsd",
            "apparent_width_px",
            "feature_area_fraction",
            "oblique",
            "quality_condition",
            "prompt_type",
            "intervention_type",
            "model_class",
        )
        if column in frame and frame[column].nunique(dropna=True) > 1
    ]
    if not fixed:
        raise ValueError("no registered fixed-effect columns are present")
    terms = [
        f"C({column})"
        if frame[column].dtype == object or str(frame[column].dtype).startswith("category")
        else column
        for column in fixed
    ]
    formula = f"{outcome} ~ " + " + ".join(terms)
    # BinomialBayesMixedGLM supports crossed variance components through formulas.
    variance = {
        "site": "0 + C(site_id)",
        "group": "0 + C(group)",
        "feature": "0 + C(target_class)",
    }
    model = sm.BinomialBayesMixedGLM.from_formula(formula, variance, frame)
    result = model.fit_vb()
    return {
        "formula": formula,
        "fixed_effect_names": model.exog_names,
        "fixed_effect_mean": result.fe_mean.tolist(),
        "fixed_effect_sd": result.fe_sd.tolist(),
        "variance_component_mean": result.vcp_mean.tolist(),
        "variance_component_sd": result.vcp_sd.tolist(),
        "n_observations": len(frame),
    }
