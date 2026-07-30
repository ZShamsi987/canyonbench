"""Full structural, hash, leakage, and protocol validation for v4 artifacts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from canyonbench.io import read_json, sha256_file
from canyonbench.trace.config import load_source_manifest
from canyonbench.trace.schemas import (
    GateResult,
    InterventionRecord,
    ProjectConfig,
    QualityParameters,
    ViewManifest,
)


@dataclass(frozen=True)
class ValidationIssue:
    severity: Literal["error", "warning"]
    code: str
    message: str
    artifact: str | None = None


def _issue(
    issues: list[ValidationIssue],
    code: str,
    message: str,
    artifact: Path | None = None,
    *,
    warning: bool = False,
) -> None:
    issues.append(
        ValidationIssue(
            severity="warning" if warning else "error",
            code=code,
            message=message,
            artifact=str(artifact) if artifact else None,
        )
    )


def _distance_m(first: dict[str, Any], second: dict[str, Any]) -> float:
    radius = 6_371_008.8
    first_latitude = math.radians(float(first["latitude"]))
    second_latitude = math.radians(float(second["latitude"]))
    delta_latitude = second_latitude - first_latitude
    delta_longitude = math.radians(float(second["longitude"]) - float(first["longitude"]))
    value = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(first_latitude) * math.cos(second_latitude) * math.sin(delta_longitude / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(value))


def validate_dataset(
    directory: Path,
    *,
    project: ProjectConfig | None = None,
    require_interventions: bool = True,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    index_path = directory / "index.json"
    manifest_path = directory / "dataset_manifest.json"
    if not index_path.exists() or not manifest_path.exists():
        _issue(issues, "MISSING_ROOT_MANIFEST", "index.json and dataset_manifest.json are required")
        return issues
    try:
        index = read_json(index_path)
        dataset_manifest = read_json(manifest_path)
    except Exception as exc:
        _issue(issues, "INVALID_ROOT_MANIFEST", str(exc))
        return issues
    clean = [row for row in index if row.get("variant") == "clean"]
    degraded = [row for row in index if row.get("variant") == "degraded"]
    expected_clean = project.dataset.clean_view_count if project else 960
    expected_degraded = project.dataset.degraded_view_count if project else 240
    if len(clean) != expected_clean:
        _issue(issues, "CLEAN_VIEW_COUNT", f"expected {expected_clean}; found {len(clean)}")
    if len(degraded) != expected_degraded:
        _issue(
            issues,
            "DEGRADED_VIEW_COUNT",
            f"expected {expected_degraded}; found {len(degraded)}",
        )
    expected_sites = project.dataset.site_count if project is not None else 120
    if dataset_manifest.get("site_count") != expected_sites:
        _issue(
            issues,
            "SITE_COUNT",
            f"expected {expected_sites} base sites; found {dataset_manifest.get('site_count')}",
        )

    clean_keys = {(row["site_id"], row["view_id"]) for row in clean}
    if len(clean_keys) != len(clean):
        _issue(issues, "DUPLICATE_VIEW", "clean site/view keys are not unique")
    site_rows: dict[str, dict[str, object]] = {}
    identifier_owners: dict[str, set[str]] = {}
    for row in clean:
        site_rows.setdefault(str(row["site_id"]), row)
    for site_id in sorted(site_rows):
        source_path = directory / site_id / "source_manifest.json"
        if not source_path.exists():
            continue
        try:
            source = load_source_manifest(source_path)
        except Exception as exc:
            _issue(issues, "SOURCE_SCHEMA_ERROR", str(exc), source_path)
            continue
        records = [
            source.imagery,
            *[
                record
                for feature_records in source.feature_sources.values()
                for record in feature_records
            ],
            *source.detector_sources.values(),
        ]
        if source.terrain is not None:
            records.append(source.terrain)
        for source_record in records:
            if source_record.redistribution != "allowed":
                _issue(
                    issues,
                    "SOURCE_REDISTRIBUTION_NOT_ALLOWED",
                    f"{site_id}/{source_record.source_id} is "
                    f"{source_record.redistribution}; public release remains blocked",
                    source_path,
                    warning=True,
                )
            namespace = f"{source_record.provider}:{source_record.product}"
            for identifier in source_record.tile_ids:
                identifier_owners.setdefault(f"{namespace}:tile:{identifier}", set()).add(site_id)
            for identifier in source_record.feature_ids:
                identifier_owners.setdefault(f"{namespace}:feature:{identifier}", set()).add(
                    site_id
                )
    for identifier, owners in identifier_owners.items():
        if len(owners) > 1:
            _issue(
                issues,
                "SHARED_SOURCE_OBJECT",
                f"{identifier} is shared by independent sites {sorted(owners)}",
            )
    minimum_separation = (
        project.dataset.minimum_site_separation_m if project is not None else 22000.0
    )
    ordered_sites = sorted(site_rows)
    for position, first_id in enumerate(ordered_sites):
        for second_id in ordered_sites[position + 1 :]:
            separation = _distance_m(site_rows[first_id], site_rows[second_id])
            if separation < minimum_separation:
                _issue(
                    issues,
                    "OVERLAPPING_SITE_FOOTPRINT",
                    f"{first_id}/{second_id} centres are {separation:.1f} m apart "
                    f"(< {minimum_separation:.1f} m)",
                )

    split_counts: dict[str, int] = {}
    for row in site_rows.values():
        split = str(row["split"])
        split_counts[split] = split_counts.get(split, 0) + 1
    fractions = (
        project.dataset.split_fractions
        if project is not None
        else {"development": 0.2, "validation": 0.2, "test": 0.6}
    )
    expected_splits = {
        split: round(expected_sites * fraction) for split, fraction in fractions.items()
    }
    if split_counts != expected_splits:
        _issue(
            issues,
            "SITE_SPLIT_COUNTS",
            f"expected {expected_splits}; found {split_counts}",
        )
    for row in clean:
        image_path = directory / row["image_path"]
        view_dir = image_path.parent
        required = [
            image_path,
            view_dir / "water_mask.png",
            view_dir / "road_mask.png",
            view_dir / "field_mask.png",
            view_dir / "camera.json",
            view_dir / "quality.json",
            view_dir / "derived.json",
            view_dir / "view_manifest.json",
            directory / row["site_id"] / "source_manifest.json",
            directory / row["site_id"] / "gate_results.json",
        ]
        for path in required:
            if not path.exists():
                _issue(issues, "MISSING_ARTIFACT", "required view artifact is absent", path)
        if any(not path.exists() for path in required):
            continue
        try:
            view = ViewManifest.model_validate(read_json(view_dir / "view_manifest.json"))
            load_source_manifest(directory / row["site_id"] / "source_manifest.json")
            gates = [
                GateResult.model_validate(item)
                for item in read_json(directory / row["site_id"] / "gate_results.json")
            ]
        except Exception as exc:
            _issue(issues, "SCHEMA_ERROR", str(exc), view_dir)
            continue
        if view.rgb_sha256 != sha256_file(image_path):
            _issue(issues, "RGB_HASH_MISMATCH", "RGB hash differs from manifest", image_path)
        source_path = directory / row["site_id"] / "source_manifest.json"
        if view.source_manifest_sha256 != sha256_file(source_path):
            _issue(
                issues,
                "SOURCE_HASH_MISMATCH",
                "source manifest hash differs from view manifest",
                source_path,
            )
        shapes: set[tuple[int, ...]] = set()
        with Image.open(image_path) as image:
            shapes.add(np.asarray(image).shape[:2])
        for feature in ("water", "road", "field"):
            mask_path = view_dir / f"{feature}_mask.png"
            with Image.open(mask_path) as source:
                mask = np.asarray(source)
            shapes.add(mask.shape[:2])
            if not set(np.unique(mask).tolist()).issubset({0, 1, 255}):
                _issue(issues, "NON_BINARY_MASK", f"{feature} mask is not binary", mask_path)
            if view.mask_sha256[feature] != sha256_file(mask_path):
                _issue(
                    issues,
                    "MASK_HASH_MISMATCH",
                    f"{feature} mask hash differs from manifest",
                    mask_path,
                )
        if len(shapes) != 1:
            _issue(issues, "PIXEL_MISALIGNMENT", f"RGB/mask shapes differ: {shapes}", view_dir)
        target_gate = next((gate for gate in gates if gate.feature == row["target_class"]), None)
        if target_gate is None or not target_gate.accepted:
            _issue(issues, "TARGET_GATE_FAILURE", "target did not pass G1-G4", view_dir)
        target_feature = view.features[view.target_class]
        if row["case_type"] != target_feature.case_type:
            _issue(
                issues,
                "VIEW_CASE_TYPE_MISMATCH",
                f"index={row['case_type']} derived={target_feature.case_type}",
                view_dir,
            )
        if row["case_type"] != "negative" and not target_feature.present:
            _issue(
                issues,
                "TARGET_OUTSIDE_VIEW",
                "non-negative target is absent from the rendered camera footprint",
                view_dir,
            )
        if (
            row["case_type"] != "negative"
            and not target_feature.resolvable
            and not target_feature.extinction
        ):
            _issue(
                issues,
                "UNRESOLVABLE_UNGATED_VIEW",
                "target is neither resolvable nor an empirical extinction case",
                view_dir,
            )
        if require_interventions and row["case_type"] != "negative":
            intervention = view_dir / "interventions" / "manifest.json"
            if not intervention.exists():
                _issue(
                    issues,
                    "MISSING_INTERVENTIONS",
                    "positive view has no intervention set",
                    view_dir,
                )
            else:
                raw_records = read_json(intervention)
                intervention_records: list[InterventionRecord] = []
                for raw_record in raw_records:
                    try:
                        intervention_record = InterventionRecord.model_validate(raw_record)
                        intervention_records.append(intervention_record)
                    except Exception as exc:
                        _issue(
                            issues,
                            "INTERVENTION_SCHEMA_ERROR",
                            str(exc),
                            intervention,
                        )
                        continue
                    edit_path = (
                        intervention_record.image_path
                        if intervention_record.image_path.is_absolute()
                        else view_dir / intervention_record.image_path
                    )
                    if not edit_path.exists():
                        _issue(
                            issues,
                            "MISSING_INTERVENTION_IMAGE",
                            "intervention image is absent",
                            edit_path,
                        )
                    elif intervention_record.image_sha256 != sha256_file(edit_path):
                        _issue(
                            issues,
                            "INTERVENTION_HASH_MISMATCH",
                            "intervention image hash differs from record",
                            edit_path,
                        )
                combinations = {
                    (record.operator, record.sequence, record.fraction)
                    for record in intervention_records
                }
                expected = {
                    (operator, sequence, fraction)
                    for operator in ("blur", "texture", "frequency", "inpaint")
                    for sequence in ("oracle_deletion", "distractor_deletion")
                    for fraction in (0.25, 0.5, 0.75, 1.0)
                }
                if combinations != expected:
                    _issue(
                        issues,
                        "INCOMPLETE_INTERVENTIONS",
                        f"missing={sorted(expected - combinations)}",
                        intervention,
                    )
                rejected_primary = [
                    record
                    for record in intervention_records
                    if record.operator in {"blur", "texture", "frequency"} and not record.accepted
                ]
                if rejected_primary:
                    _issue(
                        issues,
                        "PRIMARY_INTERVENTION_REJECTED",
                        f"{len(rejected_primary)} required O1-O3 records are rejected",
                        intervention,
                    )
                maximum_smd = (
                    project.interventions.maximum_match_smd if project is not None else 0.25
                )
                unbalanced = [
                    record
                    for record in intervention_records
                    if any(
                        abs(value) > maximum_smd
                        for value in record.standardized_mean_differences.values()
                    )
                ]
                if unbalanced:
                    _issue(
                        issues,
                        "UNBALANCED_DISTRACTOR",
                        f"{len(unbalanced)} intervention records exceed |SMD| <= {maximum_smd}",
                        intervention,
                    )
    for row in degraded:
        image_path = directory / row["image_path"]
        quality_path = image_path.parent / "quality.json"
        parent_path = image_path.parent / "parent.json"
        if not image_path.exists() or not quality_path.exists() or not parent_path.exists():
            _issue(
                issues,
                "INCOMPLETE_DEGRADED_VIEW",
                "degraded RGB, quality, and parent records are required",
                image_path.parent,
            )
            continue
        try:
            quality = QualityParameters.model_validate(read_json(quality_path))
        except Exception as exc:
            _issue(issues, "QUALITY_SCHEMA_ERROR", str(exc), quality_path)
            continue
        if quality.degradation == "none":
            _issue(
                issues,
                "EMPTY_DEGRADATION",
                "a degraded view must apply exactly one non-default factor",
                quality_path,
            )
    return issues


def validation_report(issues: list[ValidationIssue]) -> dict[str, object]:
    return {
        "passed": not any(issue.severity == "error" for issue in issues),
        "error_count": sum(issue.severity == "error" for issue in issues),
        "warning_count": sum(issue.severity == "warning" for issue in issues),
        "issues": [asdict(issue) for issue in issues],
    }
