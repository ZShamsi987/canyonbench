"""Versioned constrained-output probe definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from canyonbench.exceptions import DataValidationError
from canyonbench.schemas import (
    CaptionResponse,
    FalsePremiseResponse,
    GroundingResponse,
    PresenceResponse,
    StrictModel,
    VegetationResponse,
)

ProbeName = Literal["presence", "vegetation", "grounding", "false_premise", "caption"]

RESPONSE_MODELS: dict[str, type[StrictModel]] = {
    "presence": PresenceResponse,
    "vegetation": VegetationResponse,
    "grounding": GroundingResponse,
    "false_premise": FalsePremiseResponse,
    "caption": CaptionResponse,
}


@dataclass(frozen=True)
class Probe:
    name: ProbeName
    variant: str
    system: str
    prompt: str
    max_tokens: int | None = None

    @property
    def response_model(self) -> type[StrictModel]:
        return RESPONSE_MODELS[self.name]

    @property
    def response_schema(self) -> dict[str, Any]:
        return self.response_model.model_json_schema()


def load_probes(path: str | Path) -> list[Probe]:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise DataValidationError(f"Could not load probes {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise DataValidationError("Probe file must declare schema_version: 1")
    records = raw.get("probes")
    if not isinstance(records, list) or not records:
        raise DataValidationError("Probe file must contain a non-empty probes list")
    probes: list[Probe] = []
    for index, value in enumerate(records):
        try:
            probe = Probe(**value)
        except (TypeError, KeyError) as exc:
            raise DataValidationError(f"Invalid probe at index {index}: {exc}") from exc
        if probe.name not in RESPONSE_MODELS:
            raise DataValidationError(f"Unknown probe name: {probe.name}")
        probes.append(probe)
    keys = [(probe.name, probe.variant) for probe in probes]
    if len(keys) != len(set(keys)):
        raise DataValidationError("Probe name/variant pairs must be unique")
    return probes
