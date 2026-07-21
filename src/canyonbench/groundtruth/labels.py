"""Load and validate fixed-schema human annotations."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from canyonbench.exceptions import DataValidationError
from canyonbench.io import iter_jsonl
from canyonbench.schemas import (
    CaptionJudgeAnnotation,
    GridAnnotation,
    PresenceAnnotation,
    QualityAnnotation,
)

Record = TypeVar("Record", bound=BaseModel)


def load_records(path: str | Path, model: type[Record]) -> list[Record]:
    records: list[Record] = []
    for line_number, value in enumerate(iter_jsonl(path), 1):
        try:
            records.append(model.model_validate(value))
        except ValidationError as exc:
            raise DataValidationError(f"{path}:{line_number}: schema violation: {exc}") from exc
    return records


def load_presence(path: str | Path) -> list[PresenceAnnotation]:
    return load_records(path, PresenceAnnotation)


def load_quality(path: str | Path) -> list[QualityAnnotation]:
    return load_records(path, QualityAnnotation)


def load_grid(path: str | Path) -> list[GridAnnotation]:
    return load_records(path, GridAnnotation)


def load_judge_validation(path: str | Path) -> list[CaptionJudgeAnnotation]:
    return load_records(path, CaptionJudgeAnnotation)
