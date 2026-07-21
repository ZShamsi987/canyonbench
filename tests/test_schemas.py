from __future__ import annotations

import pytest
from pydantic import ValidationError

from canyonbench.schemas import GridAnnotation, GroundingResponse, PresenceAnnotation


def test_presence_schema_is_closed() -> None:
    record = {
        "image": "img_006806.jpg",
        "annotator": "ZH",
        "water": "yes",
        "road": "no",
        "building": "no",
        "forest": "no",
        "snow": "no",
        "field": "uncertain",
    }
    assert PresenceAnnotation.model_validate(record).water == "yes"
    with pytest.raises(ValidationError):
        PresenceAnnotation.model_validate({**record, "extra": True})


def test_grid_requires_all_cells() -> None:
    with pytest.raises(ValidationError, match="16 cells"):
        GridAnnotation(image="img_006806.jpg", annotator="ZH", cells={"0,0": True})
    cells = {f"{row},{column}": False for row in range(4) for column in range(4)}
    assert len(GridAnnotation(image="img_006806.jpg", annotator="ZH", cells=cells).cells) == 16


def test_grounding_response_rejects_bad_cells_and_points() -> None:
    with pytest.raises(ValidationError, match="invalid grid"):
        GroundingResponse(cells=["4,0"])
    with pytest.raises(ValidationError, match="coordinates"):
        GroundingResponse(cells=["0,0"], points=[(1.2, 0.5)])
