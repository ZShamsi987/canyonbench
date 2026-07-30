"""Objective 5-10% coauthor audit sampling and agreement checks."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image, ImageDraw

from canyonbench.exceptions import DataValidationError
from canyonbench.io import read_json, write_csv, write_json
from canyonbench.trace.schemas import AuditRecord

AUDIT_FIELDS = [
    "site",
    "view",
    "auditor",
    "overlay_aligned",
    "feature_resolvable",
    "obvious_edit_artifact",
    "source_mismatch",
    "notes",
]


def _panel(path: Path, *, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return cast(Image.Image, image)


def _audit_sheet(
    dataset_dir: Path,
    row: dict[str, Any],
    output: Path,
) -> Path | None:
    image_value = row.get("image_path")
    if not image_value:
        return None
    image_path = dataset_dir / str(image_value)
    view_dir = image_path.parent
    mask_path = view_dir / f"{row['target_class']}_mask.png"
    if not image_path.exists() or not mask_path.exists():
        return None
    with Image.open(image_path) as source:
        rgb = np.asarray(source.convert("RGB"))
    panel_size = (rgb.shape[1], rgb.shape[0])
    clean = Image.fromarray(rgb)
    with Image.open(mask_path) as source:
        mask = np.asarray(source) > 0
    overlay = rgb.copy()
    overlay[mask] = np.clip(0.45 * overlay[mask] + 0.55 * np.array([255, 0, 0]), 0, 255)
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay_path = output.with_name(output.stem + "__overlay.png")
    Image.fromarray(overlay.astype(np.uint8)).save(overlay_path)

    panels: list[tuple[str, Image.Image]] = [
        ("clean RGB", clean),
        ("target overlay", _panel(overlay_path, size=panel_size)),
    ]
    intervention_path = view_dir / "interventions" / "manifest.json"
    if intervention_path.exists():
        records = read_json(intervention_path)
        for operator in ("blur", "texture", "frequency"):
            for sequence in ("oracle_deletion", "distractor_deletion"):
                match = next(
                    (
                        record
                        for record in records
                        if record["operator"] == operator
                        and record["sequence"] == sequence
                        and record["fraction"] == 1
                    ),
                    None,
                )
                if match is None:
                    continue
                path = Path(str(match["image_path"]))
                if not path.is_absolute():
                    path = view_dir / path
                if path.exists():
                    panels.append(
                        (
                            f"{operator}: "
                            f"{'target' if sequence == 'oracle_deletion' else 'distractor'}",
                            _panel(path, size=panel_size),
                        )
                    )
    width, panel_height = panel_size
    label_height = 36
    row_height = panel_height + label_height
    sheet = Image.new("RGB", (width * 2, row_height * 4), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, panel) in enumerate(panels[:8]):
        x = (index % 2) * width
        y = (index // 2) * row_height
        sheet.paste(panel, (x, y + label_height))
        draw.text((x + 8, y + 8), label, fill="black")
    sheet.save(output, optimize=True)
    overlay_path.unlink(missing_ok=True)
    return output


def create_audit_sample(
    dataset_dir: Path,
    output_csv: Path,
    *,
    fraction: float = 0.1,
    seed: int = 2026,
    auditors: tuple[str, str] = ("auditor_1", "auditor_2"),
) -> Path:
    """Stratify audit views and duplicate every row for exactly two auditors."""

    if not 0.05 <= fraction <= 0.1:
        raise ValueError("registered audit fraction must be between 5 and 10 percent")
    index = [row for row in read_json(dataset_dir / "index.json") if row["variant"] == "clean"]
    target = round(len(index) * fraction)
    generator = np.random.default_rng(seed)
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in index:
        strata[(row["group"], row["target_class"], row["case_type"])].append(row)
    selected: list[dict[str, Any]] = []
    keys = sorted(strata)
    while len(selected) < target:
        for key in keys:
            if not strata[key] or len(selected) >= target:
                continue
            position = int(generator.integers(0, len(strata[key])))
            selected.append(strata[key].pop(position))
    rows = [
        {
            "site": row["site_id"],
            "view": row["view_id"],
            "auditor": auditor,
            "overlay_aligned": "",
            "feature_resolvable": "",
            "obvious_edit_artifact": "",
            "source_mismatch": "",
            "notes": "",
        }
        for row in selected
        for auditor in auditors
    ]
    write_csv(output_csv, rows, AUDIT_FIELDS)
    assets_root = output_csv.parent / f"{output_csv.stem}_assets"
    asset_rows = []
    for row in selected:
        sheet = _audit_sheet(
            dataset_dir,
            row,
            assets_root / f"{row['site_id']}__{row['view_id']}.png",
        )
        asset_rows.append(
            {
                "site": row["site_id"],
                "view": row["view_id"],
                "target_class": row["target_class"],
                "case_type": row["case_type"],
                "review_sheet": (
                    str(sheet.relative_to(output_csv.parent)) if sheet is not None else None
                ),
            }
        )
    write_json(
        output_csv.with_suffix(".manifest.json"),
        {
            "schema_version": "4.0.0",
            "audit_fraction": fraction,
            "unique_views": len(selected),
            "rows": len(rows),
            "auditors": list(auditors),
            "objective_binary_only": True,
            "review_assets": asset_rows,
        },
    )
    return output_csv


def _boolean(value: str, *, field: str, line: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise DataValidationError(f"audit row {line}: {field} must be yes/no")


def load_audit(path: Path) -> list[AuditRecord]:
    rows: list[AuditRecord] = []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for line, row in enumerate(csv.DictReader(handle), 2):
                value = dict(row)
                for field in (
                    "overlay_aligned",
                    "feature_resolvable",
                    "obvious_edit_artifact",
                    "source_mismatch",
                ):
                    value[field] = _boolean(str(value[field]), field=field, line=line)
                rows.append(AuditRecord.model_validate(value))
    except (OSError, csv.Error, ValueError, KeyError) as exc:
        raise DataValidationError(f"Invalid audit CSV {path}: {exc}") from exc
    return rows


def summarize_audit(records: list[AuditRecord]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[AuditRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.site, record.view)].append(record)
    if any(len(rows) != 2 for rows in grouped.values()):
        raise DataValidationError("every audited view must have exactly two independent auditors")
    if any(len({row.auditor for row in rows}) != 2 for rows in grouped.values()):
        raise DataValidationError("every audited view must have two distinct auditor identities")
    fields = (
        "overlay_aligned",
        "feature_resolvable",
        "obvious_edit_artifact",
        "source_mismatch",
    )
    agreement = {
        field: float(
            np.mean(
                [getattr(rows[0], field) == getattr(rows[1], field) for rows in grouped.values()]
            )
        )
        for field in fields
    }
    prevalence = {
        field: float(np.mean([getattr(row, field) for row in records])) for field in fields
    }
    failures: Counter[str] = Counter()
    for row in records:
        if not row.overlay_aligned:
            failures["overlay_misaligned"] += 1
        if not row.feature_resolvable:
            failures["unresolvable"] += 1
        if row.obvious_edit_artifact:
            failures["obvious_edit_artifact"] += 1
        if row.source_mismatch:
            failures["source_mismatch"] += 1
    view_failures = {
        "overlay_misaligned": sum(
            any(not row.overlay_aligned for row in rows) for rows in grouped.values()
        ),
        "unresolvable": sum(
            any(not row.feature_resolvable for row in rows) for rows in grouped.values()
        ),
        "obvious_edit_artifact": sum(
            any(row.obvious_edit_artifact for row in rows) for rows in grouped.values()
        ),
        "source_mismatch": sum(
            any(row.source_mismatch for row in rows) for rows in grouped.values()
        ),
    }
    return {
        "unique_views": len(grouped),
        "audit_rows": len(records),
        "agreement": agreement,
        "prevalence": prevalence,
        "failure_votes": dict(failures),
        "conservative_failure_views": view_failures,
        "source_mismatch_view_rate": (
            view_failures["source_mismatch"] / len(grouped) if grouped else 0
        ),
    }


def extinction_band_validation(
    records: list[AuditRecord],
    dataset_dir: Path,
) -> dict[str, Any]:
    """Close the fourth extinction criterion: humans see no trace where optics say none.

    Criteria (i)-(iii) of the extinction band are automatic (apparent width,
    calibrated local contrast, and the exclusion-only detector). This joins the
    audited ``feature_resolvable`` votes onto the derived per-view flags so the
    band's calibration is a reported result rather than an assumption.
    """

    index = {
        (str(row["site_id"]), str(row["view_id"])): row
        for row in read_json(dataset_dir / "index.json")
        if row["variant"] == "clean"
    }
    votes: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for record in records:
        votes[(record.site, record.view)].append(record.feature_resolvable)
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unmatched: list[str] = []
    for key, resolvable_votes in sorted(votes.items()):
        row = index.get(key)
        if row is None:
            unmatched.append("/".join(key))
            continue
        machine_case = str(row["case_type"])
        strata[machine_case].append(
            {
                "site_id": key[0],
                "view_id": key[1],
                "target_class": str(row["target_class"]),
                "human_any_resolvable": any(resolvable_votes),
                "human_all_resolvable": all(resolvable_votes),
                "auditors_agree": len(set(resolvable_votes)) == 1,
            }
        )

    def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
        if not rows:
            return {"views": 0}
        return {
            "views": len(rows),
            "human_any_resolvable_rate": float(
                np.mean([row["human_any_resolvable"] for row in rows])
            ),
            "human_all_resolvable_rate": float(
                np.mean([row["human_all_resolvable"] for row in rows])
            ),
            "auditor_agreement_rate": float(np.mean([row["auditors_agree"] for row in rows])),
        }

    extinction_rows = strata.get("extinction", [])
    positive_rows = strata.get("positive", [])
    # A confirmed extinction view is one where neither auditor saw any trace.
    confirmed = [row for row in extinction_rows if not row["human_any_resolvable"]]
    contradicted = [row for row in extinction_rows if row["human_all_resolvable"]]
    return {
        "audited_views": sum(len(rows) for rows in strata.values()),
        "unmatched_audit_views": unmatched,
        "by_machine_case_type": {
            case_type: summarize(rows) for case_type, rows in sorted(strata.items())
        },
        "extinction_band": {
            "audited_views": len(extinction_rows),
            "human_confirmed_no_trace": len(confirmed),
            "human_confirmation_rate": (
                len(confirmed) / len(extinction_rows) if extinction_rows else float("nan")
            ),
            "human_contradicted_views": [
                f"{row['site_id']}/{row['view_id']}" for row in contradicted
            ],
            "human_contradiction_rate": (
                len(contradicted) / len(extinction_rows) if extinction_rows else float("nan")
            ),
            "by_class": {
                target_class: summarize(
                    [row for row in extinction_rows if row["target_class"] == target_class]
                )
                for target_class in sorted({row["target_class"] for row in extinction_rows})
            },
        },
        "resolvable_positive_control": {
            "audited_views": len(positive_rows),
            "human_visible_rate": (
                float(np.mean([row["human_any_resolvable"] for row in positive_rows]))
                if positive_rows
                else float("nan")
            ),
        },
        "interpretation": (
            "The band is validated when audited extinction views are confirmed as "
            "showing no trace and audited resolvable positives are confirmed visible. "
            "Contradicted extinction views must be regenerated or excluded, never "
            "relabelled."
        ),
    }
