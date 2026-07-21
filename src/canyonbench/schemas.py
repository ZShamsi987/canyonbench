"""Validated public records and run configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from canyonbench.constants import FEATURES

PresenceValue = Literal["yes", "no", "uncertain"]
CaptionValue = Literal["yes", "no", "hedged"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PresenceLabels(StrictModel):
    water: PresenceValue
    road: PresenceValue
    building: PresenceValue
    forest: PresenceValue
    snow: PresenceValue
    field: PresenceValue


class PresenceAnnotation(PresenceLabels):
    image: str = Field(pattern=r"^img_\d{6,}\.jpg$")
    annotator: str = Field(min_length=1, max_length=32)


class QualityLabels(StrictModel):
    cloud: Literal["none", "partial", "heavy"]
    clarity: Literal["clear", "moderate", "heavy"]
    balloon: Literal["none", "partial"]
    sharpness: Literal["sharp", "blurred"]
    exposure: Literal["ok", "over", "under"]
    glare: Literal["none", "present"]


class QualityAnnotation(QualityLabels):
    image: str = Field(pattern=r"^img_\d{6,}\.jpg$")
    annotator: str = Field(min_length=1, max_length=32)


class GridAnnotation(StrictModel):
    image: str = Field(pattern=r"^img_\d{6,}\.jpg$")
    annotator: str = Field(min_length=1, max_length=32)
    cells: dict[str, bool]

    @model_validator(mode="after")
    def complete_grid(self) -> GridAnnotation:
        expected = {f"{row},{column}" for row in range(4) for column in range(4)}
        if set(self.cells) != expected:
            missing = sorted(expected - set(self.cells))
            extra = sorted(set(self.cells) - expected)
            raise ValueError(
                f"grid must contain exactly 16 cells; missing={missing}, extra={extra}"
            )
        return self


class CaptionAssertions(StrictModel):
    water: CaptionValue
    road: CaptionValue
    building: CaptionValue
    forest: CaptionValue
    snow: CaptionValue
    field: CaptionValue


class CaptionJudgeAnnotation(StrictModel):
    caption_id: str = Field(min_length=1)
    annotator: str = Field(min_length=1, max_length=32)
    asserts: CaptionAssertions


class PresenceResponse(StrictModel):
    """Binary constrained response for the primary feature-presence probe."""

    water: Literal["yes", "no"]
    road: Literal["yes", "no"]
    building: Literal["yes", "no"]
    forest: Literal["yes", "no"]
    snow: Literal["yes", "no"]
    field: Literal["yes", "no"]


class VegetationResponse(StrictModel):
    percent: Annotated[float, Field(ge=0, le=100)]


class GroundingResponse(StrictModel):
    cells: list[str]
    points: list[tuple[float, float]] | None = None

    @model_validator(mode="after")
    def valid_cells_and_points(self) -> GroundingResponse:
        valid = {f"{row},{column}" for row in range(4) for column in range(4)}
        invalid = sorted(set(self.cells) - valid)
        if invalid:
            raise ValueError(f"invalid grid cells: {invalid}")
        if len(set(self.cells)) != len(self.cells):
            raise ValueError("grid cell list contains duplicates")
        if self.points is not None:
            for x, y in self.points:
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    raise ValueError("normalized pointing coordinates must be in [0, 1]")
        return self


class FalsePremiseResponse(StrictModel):
    premise_correct: bool
    answer: str = Field(min_length=1, max_length=2000)


class CaptionResponse(StrictModel):
    description: str = Field(min_length=1, max_length=4000)


class AdapterConfig(StrictModel):
    kind: Literal["openai_compatible", "fixture"] = "openai_compatible"
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: float = Field(default=120, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)


class ModelConfig(StrictModel):
    id: str
    adapter: AdapterConfig
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0, ge=0, le=2)
    image_max_side: int = Field(default=1536, ge=128)
    supports_json_schema: bool = True
    supports_pointing: bool = False


class BudgetConfig(StrictModel):
    max_requests: int = Field(gt=0)
    max_cost_usd: float = Field(ge=0)
    input_per_million_usd: float = Field(default=0, ge=0)
    output_per_million_usd: float = Field(default=0, ge=0)


class RunConfig(StrictModel):
    release_dir: Path
    output_dir: Path
    models: list[ModelConfig] = Field(min_length=1)
    probes_file: Path
    probes: list[Literal["presence", "vegetation", "grounding", "false_premise", "caption"]]
    split: str = "test"
    image_limit: int | None = Field(default=None, gt=0)
    seed: int = 2026
    budget: BudgetConfig

    @model_validator(mode="after")
    def unique_models_and_probes(self) -> RunConfig:
        ids = [model.id for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("model ids must be unique")
        if len(self.probes) != len(set(self.probes)):
            raise ValueError("probe names must be unique")
        return self


def presence_dict(record: PresenceLabels) -> dict[str, PresenceValue]:
    return {feature: getattr(record, feature) for feature in FEATURES}
