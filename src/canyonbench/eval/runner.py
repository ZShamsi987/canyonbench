"""Resumable, manifest-backed execution of the structured probe battery."""

from __future__ import annotations

import hashlib
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from canyonbench.eval.adapters import Adapter, make_adapter
from canyonbench.eval.budget import BudgetTracker
from canyonbench.eval.probes import Probe, load_probes
from canyonbench.exceptions import BudgetExceededError, DataValidationError
from canyonbench.io import iter_jsonl, sha256_file, write_json, write_jsonl
from canyonbench.schemas import RunConfig
from canyonbench.version import __version__


def load_run_config(path: str | Path) -> RunConfig:
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        config = RunConfig.model_validate(value)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise DataValidationError(f"Invalid run configuration {source}: {exc}") from exc
    base = source.resolve().parent
    updates: dict[str, Path] = {}
    for field in ("release_dir", "output_dir", "probes_file"):
        candidate = getattr(config, field)
        if not candidate.is_absolute():
            updates[field] = (base / candidate).resolve()
    return config.model_copy(update=updates)


def request_key(model_id: str, image: str, probe: Probe) -> str:
    content = f"{model_id}\0{image}\0{probe.name}\0{probe.variant}\0{probe.prompt}"
    return hashlib.sha256(content.encode()).hexdigest()


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_existing(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    rows = list(iter_jsonl(path))
    return rows, {str(row["request_id"]) for row in rows if "request_id" in row}


def _eligible(row: pd.Series, probe: Probe) -> bool:
    if probe.name == "grounding":
        value = row.get("registration_reliable", False)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "yes"}
    return True


def run(
    config: RunConfig,
    *,
    adapters: dict[str, Adapter] | None = None,
    fixture_responses: dict[str, str] | None = None,
) -> Path:
    frames_path = config.release_dir / "frames.csv"
    if not frames_path.exists():
        raise DataValidationError(f"Release table does not exist: {frames_path}")
    frames = pd.read_csv(frames_path)
    if "split" in frames:
        frames = frames.loc[frames["split"] == config.split]
    frames = frames.sort_values("elapsed_s")
    if config.image_limit is not None:
        frames = frames.head(config.image_limit)
    probes = [probe for probe in load_probes(config.probes_file) if probe.name in config.probes]
    if not probes:
        raise DataValidationError("No configured probes were found in the probe file")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = config.output_dir / "predictions.jsonl"
    rows, completed = _load_existing(predictions_path)
    budget = BudgetTracker(config.budget)
    model_adapters = adapters or {
        model.id: make_adapter(model, fixture_responses) for model in config.models
    }
    write_json(
        config.output_dir / "run_manifest.json",
        {
            "schema_version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "canyonbench_version": __version__,
            "source_commit": _git_commit(Path.cwd()),
            "frames_sha256": sha256_file(frames_path),
            "probes_sha256": sha256_file(config.probes_file),
            "split": config.split,
            "seed": config.seed,
            "models": [
                model.model_dump(exclude={"adapter": {"api_key_env"}}) for model in config.models
            ],
            "probes": [{"name": probe.name, "variant": probe.variant} for probe in probes],
            "budget": config.budget.model_dump(),
        },
    )

    for model in config.models:
        adapter = model_adapters[model.id]
        for _, frame in frames.iterrows():
            image_value = frame.get("image_path")
            image_path = (
                Path(str(image_value))
                if pd.notna(image_value)
                else config.release_dir / "frames" / str(frame["image"])
            )
            if not image_path.is_absolute():
                image_path = config.release_dir / image_path
            for probe in probes:
                if not _eligible(frame, probe):
                    continue
                key = request_key(model.id, str(frame["image"]), probe)
                if key in completed:
                    continue
                budget.reserve_request()
                started = time.monotonic()
                error: str | None = None
                raw_response: str | None = None
                parsed_response: dict[str, Any] | None = None
                input_tokens = output_tokens = 0
                cost = 0.0
                provider_request_id = finish_reason = None
                try:
                    response = adapter.complete(
                        image_path=image_path,
                        system=probe.system,
                        prompt=probe.prompt,
                        json_schema=probe.response_schema,
                        model=model,
                    )
                    raw_response = response.content
                    input_tokens, output_tokens = response.input_tokens, response.output_tokens
                    provider_request_id = response.provider_request_id
                    finish_reason = response.raw_finish_reason
                    cost = budget.record_tokens(input_tokens, output_tokens)
                    parsed = probe.response_model.model_validate_json(raw_response)
                    parsed_response = parsed.model_dump()
                except BudgetExceededError:
                    # A hard budget is a run boundary, not a malformed model response.
                    raise
                except Exception as exc:  # error is a scored outcome and run continues
                    error = f"{type(exc).__name__}: {exc}"
                result = {
                    "request_id": key,
                    "model": model.id,
                    "image": frame["image"],
                    "segment_id": frame.get("segment_id"),
                    "probe": probe.name,
                    "variant": probe.variant,
                    "response": parsed_response,
                    "raw_response": raw_response,
                    "error": error,
                    "latency_s": time.monotonic() - started,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost,
                    "provider_request_id": provider_request_id,
                    "finish_reason": finish_reason,
                }
                rows.append(result)
                completed.add(key)
                write_jsonl(predictions_path, rows)
    return predictions_path
