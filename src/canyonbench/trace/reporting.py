"""Publication-ready tables and figures from frozen trace metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from canyonbench.constants import TRACE_NOMINAL_FEATURE_WIDTHS_M
from canyonbench.io import read_json

# Okabe-Ito categorical steps, validated for deuteran/tritan separation against a
# light surface; each class also carries a distinct marker and a direct label so
# identity never depends on colour alone.
CLASS_STYLE: tuple[tuple[str, str, str], ...] = (
    ("field", "#009E73", "s"),
    ("water", "#0072B2", "o"),
    ("road", "#D55E00", "^"),
)
INK = "#1f2328"
MUTED_INK = "#6b7280"
EXTINCTION_FILL = "#D55E00"


def _save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _latex_escape(value: object) -> str:
    text = str(value)
    for source, replacement in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
    ):
        text = text.replace(source, replacement)
    return text


def _latex_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(_latex_escape(column) for column in columns) + r" \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [
            f"{value:.3f}" if isinstance(value, float) else _latex_escape(value) for value in row
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _measured_widths(dataset_dir: Path) -> pd.DataFrame:
    """Median apparent width per class and altitude from the frozen bundles."""

    index = [row for row in read_json(dataset_dir / "index.json") if row["variant"] == "clean"]
    rows: list[dict[str, Any]] = []
    for row in index:
        manifest_path = dataset_dir / str(row["manifest_path"])
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        camera = manifest["camera"]
        for feature, derived in manifest["features"].items():
            if not derived["present"]:
                continue
            rows.append(
                {
                    "target_class": feature,
                    "altitude_agl_m": float(camera["altitude_agl_m"]),
                    "geometry": manifest["geometry"],
                    "gsd_m_per_px": float(camera["gsd_m_per_px"]),
                    "median_width_px": float(derived["median_width_px"]),
                    "extinction": bool(derived["extinction"]),
                    "resolvable": bool(derived["resolvable"]),
                }
            )
    return pd.DataFrame(rows)


def extinction_ladder(
    *,
    altitudes_agl_m: list[float],
    width_px: int = 1024,
    horizontal_fov_deg: float = 40.0,
    gate_px: float = 1.0,
) -> pd.DataFrame:
    """Geometric GSD and per-class apparent width for the registered lattice."""

    rows: list[dict[str, Any]] = []
    for altitude in sorted(altitudes_agl_m):
        ground_width_m = 2 * altitude * np.tan(np.radians(horizontal_fov_deg / 2))
        gsd = ground_width_m / width_px
        row: dict[str, Any] = {
            "altitude_agl_m": altitude,
            "ground_width_m": ground_width_m,
            "gsd_m_per_px": gsd,
        }
        for feature, ground_width in TRACE_NOMINAL_FEATURE_WIDTHS_M.items():
            row[f"{feature}_apparent_width_px"] = ground_width / gsd
            row[f"{feature}_sub_pixel"] = bool(ground_width / gsd < gate_px)
        rows.append(row)
    return pd.DataFrame(rows)


def build_extinction_figure(
    dataset_dir: Path,
    output_dir: Path,
    *,
    altitudes_agl_m: list[float] | None = None,
) -> list[Path]:
    """Figure 2: ground sample distance and per-class evidence extinction.

    The geometric ladder is exact from the camera model; measured median widths
    from the frozen bundles are overlaid so the figure reports the instrument
    rather than only its design.
    """

    measured = _measured_widths(dataset_dir)
    if altitudes_agl_m is None:
        altitudes_agl_m = (
            sorted(measured["altitude_agl_m"].unique().tolist())
            if not measured.empty
            else [3000.0, 8000.0, 16000.0, 24000.0]
        )
    ladder = extinction_ladder(altitudes_agl_m=altitudes_agl_m)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    table_path = output_dir / "extinction_ladder.csv"
    ladder.to_csv(table_path, index=False)
    outputs.append(table_path)
    latex_path = output_dir / "extinction_ladder.tex"
    latex_path.write_text(_latex_table(ladder), encoding="utf-8")
    outputs.append(latex_path)

    altitude_km = ladder["altitude_agl_m"].to_numpy(float) / 1000
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.8))

    left.plot(
        altitude_km,
        ladder["gsd_m_per_px"],
        color=INK,
        linewidth=2,
        marker="o",
        markersize=6,
    )
    for altitude, gsd in zip(altitude_km, ladder["gsd_m_per_px"], strict=True):
        left.annotate(
            f"{gsd:.1f} m/px",
            (altitude, gsd),
            xytext=(5, -10),
            textcoords="offset points",
            fontsize=7,
            color=MUTED_INK,
        )
    left.axhline(1.0, color=MUTED_INK, linewidth=1, linestyle=":")
    left.annotate(
        "NAIP source ~1 m/px",
        (altitude_km[-1], 1.0),
        xytext=(-4, 6),
        textcoords="offset points",
        fontsize=7,
        ha="right",
        color=MUTED_INK,
    )
    left.set_xlabel("Altitude AGL (km)")
    left.set_ylabel("Ground sample distance (m/px)")
    left.set_title(f"Ground sample distance\n({1024} px, {40}$^\\circ$ HFOV)", fontsize=9)
    left.grid(True, linewidth=0.4, alpha=0.35)
    left.set_axisbelow(True)

    for feature, colour, marker in CLASS_STYLE:
        widths = ladder[f"{feature}_apparent_width_px"].to_numpy(float)
        right.plot(
            altitude_km,
            widths,
            color=colour,
            linewidth=2,
            marker=marker,
            markersize=6,
            label=f"{feature} (~{TRACE_NOMINAL_FEATURE_WIDTHS_M[feature]:.0f} m)",
        )
        right.annotate(
            feature,
            (altitude_km[-1], widths[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=7,
            color=colour,
            va="center",
        )
        if not measured.empty:
            observed = (
                measured.loc[measured["target_class"] == feature]
                .groupby("altitude_agl_m", observed=True)["median_width_px"]
                .median()
                .sort_index()
            )
            if len(observed):
                right.scatter(
                    observed.index.to_numpy(float) / 1000,
                    observed.to_numpy(float),
                    facecolors="none",
                    edgecolors=colour,
                    s=64,
                    linewidths=1.2,
                    zorder=3,
                )
    right.axhline(1.0, color=MUTED_INK, linewidth=1, linestyle="--")
    right.axhspan(1e-2, 1.0, color=EXTINCTION_FILL, alpha=0.35, zorder=0)
    right.annotate(
        "sub-pixel: any 'yes' is ungrounded by optics",
        (0.03, 0.04),
        xycoords="axes fraction",
        fontsize=7,
        color=INK,
    )
    right.set_yscale("log")
    right.set_ylim(1e-1, 5e3)
    # Headroom on the right keeps the direct class labels inside the axes.
    right.set_xlim(altitude_km.min() - 1.5, altitude_km.max() + 4.0)
    right.set_xlabel("Altitude AGL (km)")
    right.set_ylabel("Apparent width (px, log)")
    right.set_title(
        "Per-class evidence extinction\n(line: geometric; ring: measured median)",
        fontsize=9,
    )
    right.grid(True, linewidth=0.4, alpha=0.35)
    right.set_axisbelow(True)
    right.legend(fontsize=7, loc="upper right", frameon=False)
    figure.tight_layout()
    figure_path = output_dir / "gsd_extinction.png"
    _save(figure, figure_path)
    outputs.extend([figure_path, figure_path.with_suffix(".pdf")])
    return outputs


def _nearest_coverage(
    curve: list[dict[str, Any]],
    value: str,
    target: float = 0.8,
) -> float:
    if not curve:
        return float("nan")
    row = min(curve, key=lambda item: abs(float(item["coverage"]) - target))
    return float(row[value])


def build_report(metrics_path: Path, rows_path: Path, output_dir: Path) -> list[Path]:
    """Write tidy tables and every registered diagnostic overview figure."""

    metrics = read_json(metrics_path)
    rows = pd.read_csv(rows_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    summary_rows = []
    for model, values in metrics["models"].items():
        summary_rows.append(
            {
                "model": model,
                **values["performance"],
                **values["localization"],
                **values["causal"],
                "prompt_prior_gap": values["prompt_prior_gap"],
                "extinction_positive_rate": values["extinction_positive_rate"],
                "acuity_threshold_width_px": values["acuity"]["threshold_width_px"],
                "target_deletion_survival_rate": values["target_deletion_survival_rate"],
                "selective_risk_at_80pct": _nearest_coverage(
                    values["selective_risk"],
                    "selective_risk",
                ),
                "selective_fpr_at_80pct": _nearest_coverage(
                    values["selective_risk"],
                    "selective_false_positive_rate",
                ),
                "selective_efs_at_80pct": _nearest_coverage(
                    values["selective_causal_faithfulness"],
                    "efs",
                ),
                "cave_coverage": metrics.get("cave", {}).get(model, {}).get("coverage", np.nan),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "model_summary.csv"
    summary.to_csv(summary_path, index=False)
    outputs.append(summary_path)
    latex_path = output_dir / "model_summary.tex"
    latex_path.write_text(_latex_table(summary), encoding="utf-8")
    outputs.append(latex_path)

    screening = rows.loc[
        (rows["tier"] == "A")
        & (rows["sequence"] == "screening")
        & (rows["variant"] == "clean")
        & (rows["apparent_width_px"] > 0)
    ].copy()
    if not screening.empty:
        figure, axis = plt.subplots(figsize=(6.5, 4))
        bins = np.geomspace(
            max(screening["apparent_width_px"].min(), 0.1),
            screening["apparent_width_px"].max(),
            8,
        )
        screening["width_bin"] = pd.cut(
            screening["apparent_width_px"], bins.tolist(), duplicates="drop"
        )
        for model, values in screening.groupby("model", observed=True):
            curve = values.groupby("width_bin", observed=True).agg(
                width=("apparent_width_px", "median"),
                detected=("predicted_positive", "mean"),
            )
            axis.plot(curve["width"], curve["detected"], marker="o", label=model)
        axis.set_xscale("log")
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Apparent feature width (pixels)")
        axis.set_ylabel("P(model answers yes)")
        axis.set_title("Scale-dependent feature acuity")
        axis.legend(fontsize=7, ncol=2)
        figure_path = output_dir / "acuity_curves.png"
        _save(figure, figure_path)
        outputs.extend([figure_path, figure_path.with_suffix(".pdf")])

        figure, axis = plt.subplots(figsize=(6.5, 4))
        for model, values in metrics["models"].items():
            curve = values["selective_causal_faithfulness"]
            if curve:
                axis.plot(
                    [row["coverage"] for row in curve],
                    [row["efs"] for row in curve],
                    marker="o",
                    label=model,
                )
        axis.set_xlabel("Coverage")
        axis.set_ylabel("Evidence faithfulness score (EFS)")
        axis.set_title("Causal faithfulness-coverage frontier")
        axis.legend(fontsize=7, ncol=2)
        figure_path = output_dir / "selective_causal_faithfulness.png"
        _save(figure, figure_path)
        outputs.extend([figure_path, figure_path.with_suffix(".pdf")])

    if not summary.empty:
        figure, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=False)
        positions = np.arange(len(summary))
        for axis, metric, title in zip(
            axes,
            ("ocrs", "efs", "osg"),
            (
                "Oracle reliance (OCRS)",
                "Self-faithfulness (EFS)",
                "Oracle-self gap (OSG)",
            ),
            strict=True,
        ):
            axis.barh(positions, summary[metric].fillna(0))
            axis.axvline(0, color="black", linewidth=0.7)
            axis.set_title(title, fontsize=9)
            axis.set_yticks(positions, summary["model"] if axis is axes[0] else [])
        figure_path = output_dir / "causal_summary.png"
        _save(figure, figure_path)
        outputs.extend([figure_path, figure_path.with_suffix(".pdf")])

        figure, axis = plt.subplots(figsize=(5.2, 4.4))
        axis.scatter(
            summary["balanced_accuracy"],
            summary["efs"],
            s=42,
        )
        for _, row in summary.iterrows():
            axis.annotate(
                str(row["model"]),
                (row["balanced_accuracy"], row["efs"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
        axis.set_xlabel("Balanced accuracy")
        axis.set_ylabel("Evidence faithfulness score (EFS)")
        axis.set_title("Task performance versus causal faithfulness")
        figure_path = output_dir / "accuracy_vs_efs.png"
        _save(figure, figure_path)
        outputs.extend([figure_path, figure_path.with_suffix(".pdf")])

        figure, axis = plt.subplots(figsize=(6.5, 4))
        for model, values in metrics["models"].items():
            curve = values["selective_risk"]
            if curve:
                axis.plot(
                    [row["coverage"] for row in curve],
                    [row["selective_risk"] for row in curve],
                    marker="o",
                    label=model,
                )
        axis.set_xlabel("Coverage")
        axis.set_ylabel("Selective risk")
        axis.set_title("Reliability-coverage frontier")
        axis.legend(fontsize=7, ncol=2)
        figure_path = output_dir / "selective_risk.png"
        _save(figure, figure_path)
        outputs.extend([figure_path, figure_path.with_suffix(".pdf")])

        figure, axis = plt.subplots(figsize=(6.5, 4))
        for model, values in metrics["models"].items():
            curve = values["selective_risk"]
            if curve:
                axis.plot(
                    [row["coverage"] for row in curve],
                    [row["selective_false_positive_rate"] for row in curve],
                    marker="o",
                    label=model,
                )
        axis.set_xlabel("Coverage")
        axis.set_ylabel("Selective false-positive rate")
        axis.set_title("False-positive reliability-coverage frontier")
        axis.legend(fontsize=7, ncol=2)
        figure_path = output_dir / "selective_false_positive_rate.png"
        _save(figure, figure_path)
        outputs.extend([figure_path, figure_path.with_suffix(".pdf")])

    prompt_profiles = {
        model: values["prompt_prior_profile"]["by_altitude"]
        for model, values in metrics["models"].items()
        if values["prompt_prior_profile"]["by_altitude"]
    }
    if prompt_profiles:
        figure, axis = plt.subplots(figsize=(6.5, 4))
        for model, profile in prompt_profiles.items():
            axis.plot(
                [row["altitude_agl_m"] / 1000 for row in profile],
                [row["mean_prompt_prior_gap"] for row in profile],
                marker="o",
                label=model,
            )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xlabel("Altitude AGL (km; resolution-associated)")
        axis.set_ylabel("False-premise minus neutral P(yes)")
        axis.set_title("Prompt-prior gap across the resolution lattice")
        axis.legend(fontsize=7, ncol=2)
        figure_path = output_dir / "prompt_prior_profile.png"
        _save(figure, figure_path)
        outputs.extend([figure_path, figure_path.with_suffix(".pdf")])

    extinction = screening.loc[screening["extinction"]]
    if not extinction.empty:
        figure, axis = plt.subplots(figsize=(6.5, 4))
        for model, values in extinction.groupby("model", observed=True):
            profile = (
                values.groupby("altitude_agl_m", observed=True)["predicted_positive"]
                .mean()
                .sort_index()
            )
            axis.plot(
                profile.index.to_numpy(float) / 1000,
                profile.to_numpy(dtype=float),
                marker="o",
                label=model,
            )
        axis.set_xlabel("Altitude AGL (km; resolution-associated)")
        axis.set_ylabel("Extinction-band positive rate")
        axis.set_ylim(-0.02, 1.02)
        axis.set_title("Optics-gated extinction behavior")
        axis.legend(fontsize=7, ncol=2)
        figure_path = output_dir / "extinction_by_altitude.png"
        _save(figure, figure_path)
        outputs.extend([figure_path, figure_path.with_suffix(".pdf")])

    trace_sequences = (
        "oracle_deletion",
        "distractor_deletion",
        "self_deletion",
        "self_sufficiency",
        "random_control",
        "texture_control",
    )
    traces = rows.loc[rows["sequence"].isin(trace_sequences) & rows["fraction"].notna()]
    if not traces.empty:
        models = sorted(str(model) for model in traces["model"].unique())
        figure, axes = plt.subplots(
            len(models),
            1,
            figsize=(7.2, max(3.4, 2.7 * len(models))),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for axis, model in zip(axes[:, 0], models, strict=True):
            values = traces.loc[traces["model"] == model]
            for sequence, sequence_rows in values.groupby("sequence", observed=True):
                trace_curve = (
                    sequence_rows.groupby("fraction", observed=True)["yes_probability"]
                    .mean()
                    .sort_index()
                )
                axis.plot(
                    trace_curve.index,
                    trace_curve.values,
                    marker="o",
                    label=sequence,
                )
            axis.set_title(model, fontsize=9)
            axis.set_ylabel("Mean P(yes)")
            axis.set_ylim(-0.02, 1.02)
        axes[-1, 0].set_xlabel("Intervened fraction")
        axes[0, 0].legend(fontsize=6, ncol=3)
        figure.suptitle("Registered causal intervention traces", fontsize=11)
        figure_path = output_dir / "intervention_traces.png"
        _save(figure, figure_path)
        outputs.extend([figure_path, figure_path.with_suffix(".pdf")])

    operator_rows = []
    for model, values in metrics["models"].items():
        for operator, causal in values["causal_by_operator"].items():
            operator_rows.append(
                {
                    "model": model,
                    "operator": operator,
                    "efs": causal["efs"],
                }
            )
    operator_frame = pd.DataFrame(operator_rows)
    if not operator_frame.empty:
        matrix = operator_frame.pivot(index="model", columns="operator", values="efs")
        figure, axis = plt.subplots(figsize=(5.5, max(2.6, 0.45 * len(matrix.index) + 1.5)))
        image = axis.imshow(matrix.to_numpy(float), aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(matrix.columns)), matrix.columns)
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        axis.set_title("EFS stability across intervention operators")
        figure.colorbar(image, ax=axis, label="EFS")
        figure_path = output_dir / "operator_efs_heatmap.png"
        _save(figure, figure_path)
        outputs.extend([figure_path, figure_path.with_suffix(".pdf")])

    sensitivity_rows = []
    for model, values in metrics["models"].items():
        for condition, causal in values["grid_k_operator_sensitivity"].items():
            sensitivity_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "efs": causal["efs"],
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    if not sensitivity.empty:
        matrix = sensitivity.pivot(index="model", columns="condition", values="efs")
        figure, axis = plt.subplots(
            figsize=(
                max(7, 0.45 * len(matrix.columns) + 2),
                max(2.8, 0.45 * len(matrix.index) + 1.5),
            )
        )
        image = axis.imshow(matrix.to_numpy(float), aspect="auto", cmap="magma")
        axis.set_xticks(
            range(len(matrix.columns)),
            matrix.columns,
            rotation=60,
            ha="right",
            fontsize=7,
        )
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        axis.set_title("Grid, cell-budget, and operator sensitivity")
        figure.colorbar(image, ax=axis, label="EFS")
        figure_path = output_dir / "grid_k_operator_sensitivity.png"
        _save(figure, figure_path)
        outputs.extend([figure_path, figure_path.with_suffix(".pdf")])

    cave = metrics.get("cave", {})
    if cave:
        cave_rows = []
        for model, values in cave.items():
            cave_rows.append(
                {
                    "model": model,
                    "direct_balanced_accuracy": values["direct"]["balanced_accuracy"],
                    "cave_balanced_accuracy": values["cave"]["balanced_accuracy"],
                    "direct_false_positive_rate": values["direct"]["false_positive_rate"],
                    "cave_false_positive_rate": values["cave"]["false_positive_rate"],
                    "coverage": values["coverage"],
                    "mean_calls_per_initial": values["mean_calls_per_initial"],
                    "false_premise_compliance_delta": values["false_premise_compliance_delta"],
                }
            )
        cave_summary = pd.DataFrame(cave_rows)
        cave_path = output_dir / "cave_summary.csv"
        cave_summary.to_csv(cave_path, index=False)
        outputs.append(cave_path)

    frontier = metrics.get("cave_frontier", {})
    points = frontier.get("points", []) if isinstance(frontier, dict) else []
    if points:
        figure, axis = plt.subplots(figsize=(6.2, 4.5))
        scatter = axis.scatter(
            [point["coverage"] for point in points],
            [point["balanced_accuracy"] for point in points],
            c=[point["mean_calls_per_initial"] for point in points],
            cmap="plasma",
            s=46,
        )
        axis.set_xlabel("Coverage")
        axis.set_ylabel("Balanced accuracy")
        axis.set_title("CAVE reliability-coverage-cost frontier")
        figure.colorbar(scatter, ax=axis, label="Mean calls per initial")
        figure_path = output_dir / "cave_frontier.png"
        _save(figure, figure_path)
        outputs.extend([figure_path, figure_path.with_suffix(".pdf")])
    return outputs
