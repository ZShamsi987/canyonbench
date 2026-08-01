"""Configuration loading with paths resolved relative to the declaring file."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from canyonbench.exceptions import DataValidationError
from canyonbench.trace.schemas import (
    CandidateSeed,
    PreregistrationConfig,
    ProjectConfig,
    PromptTemplate,
    SiteSpec,
    SourceAcquisitionConfig,
    SourceManifest,
    TraceRunConfig,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


def _yaml(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise DataValidationError(f"Could not load YAML {path}: {exc}") from exc
    if value is None:
        raise DataValidationError(f"YAML document is empty: {path}")
    return value


def _validate(path: Path, model: type[ModelT], value: Any) -> ModelT:
    try:
        return model.model_validate(value)
    except ValueError as exc:
        raise DataValidationError(f"Invalid {model.__name__} in {path}: {exc}") from exc


def _absolute(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def load_project_config(path: str | Path) -> ProjectConfig:
    source = Path(path).resolve()
    config = _validate(source, ProjectConfig, _yaml(source))
    base = source.parent
    dataset = config.dataset.model_copy(
        update={
            name: _absolute(base, getattr(config.dataset, name))
            for name in ("source_root", "output_root", "site_manifest")
        }
    )
    if dataset.quality_calibration is not None:
        dataset = dataset.model_copy(
            update={"quality_calibration": _absolute(base, dataset.quality_calibration)}
        )
    return config.model_copy(update={"dataset": dataset})


# A service that is stood up by hand gets its address from the environment, so
# the frozen roster never has to be edited to record where it happened to land.
# `CANYONBENCH_ENDPOINT__<MODEL ID>` overrides one model, with every character
# outside [A-Z0-9] replaced by an underscore.
ENDPOINT_ENV_PREFIX = "CANYONBENCH_ENDPOINT__"


def endpoint_env_var(model_id: str) -> str:
    """The environment variable that overrides one model's base URL."""

    return ENDPOINT_ENV_PREFIX + re.sub(r"[^A-Z0-9]+", "_", model_id.upper()).strip("_")


def apply_endpoint_overrides(config: TraceRunConfig) -> tuple[TraceRunConfig, dict[str, str]]:
    """Point models at the addresses their services were actually started on."""

    applied: dict[str, str] = {}
    models = []
    for model in config.models:
        override = os.environ.get(endpoint_env_var(model.id))
        if override:
            applied[model.id] = override
            model = model.model_copy(
                update={"adapter": model.adapter.model_copy(update={"base_url": override})}
            )
        models.append(model)
    return (config.model_copy(update={"models": models}) if applied else config), applied


def load_run_config_for_host(
    path: Path,
    *,
    dataset_dir: Path | None = None,
    output_dir: Path | None = None,
) -> TraceRunConfig:
    """Load the frozen roster with host-specific paths and endpoints applied.

    One config is the roster of record for the whole project, but Adroit and
    Lambda mount the dataset and write results in different places, and a
    hand-started service lands on whatever address the operator chose. Overriding
    those keeps the model definitions, prices, protocol, and intervention
    settings identical on both hosts.
    """

    config = load_trace_run_config(path)
    updates: dict[str, Path] = {}
    if dataset_dir is not None:
        updates["dataset_dir"] = dataset_dir.expanduser().resolve()
    if output_dir is not None:
        updates["output_dir"] = output_dir.expanduser().resolve()
    config = config.model_copy(update=updates) if updates else config
    config, _ = apply_endpoint_overrides(config)
    return config


def load_trace_run_config(path: str | Path) -> TraceRunConfig:
    source = Path(path).resolve()
    config = _validate(source, TraceRunConfig, _yaml(source))
    base = source.parent
    updates = {
        name: _absolute(base, getattr(config, name))
        for name in ("dataset_dir", "output_dir", "prompt_file")
    }
    if config.fixture_responses is not None:
        updates["fixture_responses"] = _absolute(base, config.fixture_responses)
    return config.model_copy(update=updates)


def load_sites(path: str | Path) -> list[SiteSpec]:
    source = Path(path).resolve()
    value = _yaml(source)
    if isinstance(value, dict) and "sites" in value:
        value = value["sites"]
    if not isinstance(value, list):
        raise DataValidationError(f"Site manifest must be a list or {{sites: [...]}}: {source}")
    base = source.parent
    sites: list[SiteSpec] = []
    for index, row in enumerate(value):
        site = _validate(source, SiteSpec, row)
        updates: dict[str, Any] = {
            name: _absolute(base, getattr(site, name))
            for name in ("imagery_path", "source_manifest_path")
        }
        for name in ("dem_path",):
            candidate = getattr(site, name)
            if candidate is not None:
                updates[name] = _absolute(base, candidate)
        for name in ("primary_mask_paths", "secondary_mask_paths", "detector_score_paths"):
            updates[name] = {
                feature: _absolute(base, candidate)
                for feature, candidate in getattr(site, name).items()
            }
        site = site.model_copy(update=updates)
        if any(existing.site_id == site.site_id for existing in sites):
            raise DataValidationError(
                f"Duplicate site_id {site.site_id} at row {index + 1} in {source}"
            )
        sites.append(site)
    return sites


def load_source_manifest(path: str | Path) -> SourceManifest:
    source = Path(path).resolve()
    return _validate(source, SourceManifest, _yaml(source))


def load_source_acquisition_config(path: str | Path) -> SourceAcquisitionConfig:
    source = Path(path).resolve()
    return _validate(source, SourceAcquisitionConfig, _yaml(source))


def load_candidate_seeds(path: str | Path) -> list[CandidateSeed]:
    source = Path(path).resolve()
    value = _yaml(source)
    if isinstance(value, dict) and "candidates" in value:
        value = value["candidates"]
    if not isinstance(value, list):
        raise DataValidationError(
            f"Candidate manifest must be a list or {{candidates: [...]}}: {source}"
        )
    seeds = [_validate(source, CandidateSeed, row) for row in value]
    ids = [seed.candidate_id for seed in seeds]
    if len(ids) != len(set(ids)):
        raise DataValidationError(f"Candidate ids are not unique in {source}")
    return seeds


def load_prompts(path: str | Path) -> list[PromptTemplate]:
    source = Path(path).resolve()
    value = _yaml(source)
    if isinstance(value, dict) and "prompts" in value:
        value = value["prompts"]
    if not isinstance(value, list):
        raise DataValidationError(f"Prompt file must contain a prompt list: {source}")
    prompts = [_validate(source, PromptTemplate, row) for row in value]
    ids = [prompt.id for prompt in prompts]
    if len(ids) != len(set(ids)):
        raise DataValidationError(f"Prompt ids are not unique in {source}")
    variants = {prompt.variant for prompt in prompts}
    required = {"neutral", "false_premise", "uncertainty_aware", "evidence_first", "no_image"}
    if variants != required:
        raise DataValidationError(
            f"Prompt file must contain exactly the five registered variants; got {sorted(variants)}"
        )
    return prompts


def load_preregistration(path: str | Path) -> PreregistrationConfig:
    source = Path(path).resolve()
    return _validate(source, PreregistrationConfig, _yaml(source))
