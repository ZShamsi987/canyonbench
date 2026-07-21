"""Association-only ascent analysis with documented confound controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from canyonbench.exceptions import DataValidationError

DEFAULT_CONTROLS = (
    "contrast_std",
    "sharpness_edge_var",
    "clipped_high_fraction",
    "clipped_low_fraction",
    "feature_prevalence",
)


def controlled_association(
    frame: pd.DataFrame,
    *,
    outcome: str,
    exposure: str = "alt_m",
    controls: tuple[str, ...] = DEFAULT_CONTROLS,
    phase: str = "phase",
) -> dict[str, Any]:
    """Fit OLS with robust segment-clustered covariance; never interpreted causally."""

    required = {outcome, exposure, "segment_id", phase, *controls}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"Association table is missing fields: {missing}")
    try:
        import statsmodels.formula.api as smf  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DataValidationError(
            "Controlled analysis requires the analysis extra: pip install 'canyonbench[analysis]'"
        ) from exc
    columns = [outcome, exposure, "segment_id", phase, *controls]
    clean = frame[columns].dropna().copy()
    if clean["segment_id"].nunique() < 2:
        raise DataValidationError("Clustered inference requires at least two trajectory segments")
    formula = f"{outcome} ~ {exposure} + " + " + ".join(controls) + f" + C({phase})"
    fit = smf.ols(formula, data=clean).fit(
        cov_type="cluster", cov_kwds={"groups": clean["segment_id"]}
    )
    interval = fit.conf_int().loc[exposure]
    return {
        "interpretation": "association_not_causation",
        "formula": formula,
        "n_frames": len(clean),
        "effective_segments": int(clean["segment_id"].nunique()),
        "coefficient": float(fit.params[exposure]),
        "std_error": float(fit.bse[exposure]),
        "p_value": float(fit.pvalues[exposure]),
        "ci_95": [float(interval.iloc[0]), float(interval.iloc[1])],
        "r_squared": float(fit.rsquared),
    }


def stratified_trends(
    frame: pd.DataFrame,
    *,
    outcome: str,
    exposure: str = "alt_m",
    prevalence: str = "feature_prevalence",
    quality: str = "contrast_std",
    bins: int = 3,
) -> list[dict[str, Any]]:
    required = {outcome, exposure, prevalence, quality}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"Stratification table is missing fields: {missing}")
    clean = frame[list(required)].dropna().copy()
    clean["prevalence_stratum"] = pd.qcut(clean[prevalence], bins, duplicates="drop")
    clean["quality_stratum"] = pd.qcut(clean[quality], bins, duplicates="drop")
    output: list[dict[str, Any]] = []
    for (prevalence_bin, quality_bin), group in clean.groupby(
        ["prevalence_stratum", "quality_stratum"], observed=True
    ):
        if len(group) < 3 or group[exposure].nunique() < 2:
            continue
        coefficient = np.polyfit(group[exposure], group[outcome], 1)[0]
        output.append(
            {
                "prevalence_stratum": str(prevalence_bin),
                "quality_stratum": str(quality_bin),
                "n": len(group),
                "slope": float(coefficient),
            }
        )
    return output
