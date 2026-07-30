"""Merge per-host prediction logs into one scored input.

Open-weight inference runs on Lambda and proprietary inference runs over the
OpenRouter API, so a registered run produces one append-only log per host. The
request ID is a content hash of the full query, so merging is exact: an identical
key from two hosts must carry an identical request, and a disagreement is a
protocol error rather than something to silently deduplicate away.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from canyonbench.exceptions import DataValidationError
from canyonbench.io import iter_jsonl, read_json, write_json, write_jsonl
from canyonbench.trace.schemas import TracePrediction


def _merge_run_manifests(sources: list[Path], output: Path) -> dict[str, Any]:
    """Carry every host's roster forward beside the merged log.

    Scoring reads `run_manifest.json` next to the predictions to recover each
    model's benchmark role. Without a merged manifest every model would score as
    `unregistered` and the per-class comparison behind H4 would collapse.
    """

    models: dict[str, dict[str, Any]] = {}
    manifests: list[dict[str, Any]] = []
    for source in sources:
        path = source.parent / "run_manifest.json"
        if not path.is_file():
            continue
        manifest = read_json(path)
        manifests.append(
            {
                "path": str(path),
                "created_at": manifest.get("created_at"),
                "canyonbench_version": manifest.get("canyonbench_version"),
                "dataset_index_sha256": manifest.get("dataset_index_sha256"),
                "tiers": manifest.get("tiers"),
            }
        )
        for model in manifest.get("models", []):
            identifier = str(model["id"])
            existing = models.get(identifier)
            if existing is not None and existing != model:
                raise DataValidationError(
                    f"model {identifier} is declared differently by two hosts; "
                    "both must run the same frozen roster entry"
                )
            models[identifier] = model

    indexes = {
        manifest["dataset_index_sha256"]
        for manifest in manifests
        if manifest.get("dataset_index_sha256")
    }
    if len(indexes) > 1:
        raise DataValidationError(
            "hosts ran against different dataset indexes; the frozen bundle must "
            f"be identical on both: {sorted(indexes)}"
        )

    merged = {
        "schema_version": "4.2.0",
        "merged_from": manifests,
        "dataset_index_sha256": next(iter(indexes), None),
        "models": [models[identifier] for identifier in sorted(models)],
    }
    write_json(output.parent / "run_manifest.json", merged)
    return merged


def merge_predictions(sources: list[Path], output: Path) -> dict[str, Any]:
    """Concatenate prediction logs, rejecting conflicting duplicate request IDs."""

    if not sources:
        raise DataValidationError("at least one predictions.jsonl is required")
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise DataValidationError(f"missing prediction logs: {missing}")

    merged: dict[str, dict[str, Any]] = {}
    origin: dict[str, Path] = {}
    per_source: list[dict[str, Any]] = []
    duplicates = 0
    for source in sources:
        rows = list(iter_jsonl(source))
        models: set[str] = set()
        for row in rows:
            prediction = TracePrediction.model_validate(row)
            key = prediction.request.request_id
            models.add(prediction.request.model)
            if key in merged:
                if merged[key] != row:
                    raise DataValidationError(
                        f"request {key} differs between {origin[key]} and {source}; "
                        "the same query cannot have two recorded answers"
                    )
                duplicates += 1
                continue
            merged[key] = row
            origin[key] = source
        per_source.append(
            {
                "path": str(source),
                "rows": len(rows),
                "models": sorted(models),
            }
        )

    ordered = [merged[key] for key in sorted(merged)]
    write_jsonl(output, ordered)
    manifest = _merge_run_manifests(sources, output)
    models_by_id: dict[str, int] = {}
    for row in ordered:
        model = str(row["request"]["model"])
        models_by_id[model] = models_by_id.get(model, 0) + 1
    roles = {str(model["id"]): model.get("benchmark_role") for model in manifest.get("models", [])}
    unregistered = sorted(name for name in models_by_id if name not in roles)
    report = {
        "schema_version": "4.2.0",
        "output": str(output),
        "sources": per_source,
        "merged_rows": len(ordered),
        "identical_duplicates_dropped": duplicates,
        "rows_by_model": dict(sorted(models_by_id.items())),
        "roles": dict(sorted((name, role) for name, role in roles.items() if name in models_by_id)),
        # A model without a role scores as `unregistered` and drops out of the
        # per-class comparison, so surface it here rather than at analysis time.
        "models_without_a_run_manifest": unregistered,
    }
    write_json(output.with_suffix(".merge.json"), report)
    return report
