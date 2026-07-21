from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canyonbench.eval.bootstrap import segment_bootstrap
from canyonbench.eval.calibration import calibration_by_mechanism, expected_calibration_error
from canyonbench.eval.judge import assertions_frame, rule_based_assertions, validate_judge
from canyonbench.eval.metrics import (
    false_premise_summary,
    grounding_summary,
    presence_summary,
    score_benchmark,
    vegetation_summary,
)
from canyonbench.exceptions import DataValidationError


def ground_truth() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image": "img_006806.jpg",
                "segment_id": "s1",
                "vegetation_fraction": 0.1,
                "vegetation_cells": ["0,0"],
                "water": "yes",
                "road": "no",
                "building": "no",
                "forest": "no",
                "snow": "no",
                "field": "uncertain",
            },
            {
                "image": "img_006900.jpg",
                "segment_id": "s2",
                "vegetation_fraction": 0.0,
                "vegetation_cells": [],
                "water": "no",
                "road": "yes",
                "building": "no",
                "forest": "no",
                "snow": "no",
                "field": "no",
            },
        ]
    )


def test_primary_summaries() -> None:
    truth = ground_truth()
    presence = truth.assign(
        response=[
            {
                "water": "yes",
                "road": "no",
                "building": "no",
                "forest": "no",
                "snow": "no",
                "field": "no",
            },
            {
                "water": "yes",
                "road": "yes",
                "building": "no",
                "forest": "no",
                "snow": "no",
                "field": "no",
            },
        ]
    )
    summary = presence_summary(presence)
    assert summary["per_feature"]["water"]["false_positive_rate"] == 1
    assert summary["per_feature"]["field"]["n"] == 1
    vegetation = vegetation_summary(truth.assign(response=[{"percent": 15}, {"percent": 0}]))
    assert vegetation["mae"] == 2.5
    grounding = grounding_summary(
        truth.assign(response=[{"cells": ["0,0"], "points": [(0.1, 0.1)]}, {"cells": []}])
    )
    assert grounding["region_f1"] == 1
    assert grounding["point_precision"] == 1


def test_false_premise_delta() -> None:
    frame = pd.DataFrame(
        [
            {"variant": "leading", "response": {"premise_correct": True}},
            {"variant": "evidence_first", "response": {"premise_correct": False}},
        ]
    )
    assert false_premise_summary(frame)["mitigation_delta"] == 1


def test_segment_bootstrap_discloses_effective_n() -> None:
    frame = pd.DataFrame({"segment_id": ["a", "a", "b"], "value": [1.0, 1.0, 3.0]})
    result = segment_bootstrap(frame, lambda data: data.value.mean(), iterations=100, seed=1)
    assert result["estimate"] == pytest.approx(5 / 3)
    assert result["effective_segments"] == 2
    assert result["lower"] is not None


def test_score_benchmark_groups_models_and_bootstraps() -> None:
    truth = ground_truth()
    predictions = [
        {
            "model": "m",
            "image": row.image,
            "probe": "vegetation",
            "variant": "neutral",
            "response": {"percent": 10 if index == 0 else 0},
        }
        for index, row in enumerate(truth.itertuples())
    ]
    scored = score_benchmark(truth, predictions, bootstrap_iterations=100)
    assert scored["effective_segment_count"] == 2
    assert scored["models"]["m"]["vegetation:neutral"]["mae"] == 0


def test_rule_judge_negation_hedge_and_validation() -> None:
    result = rule_based_assertions("There is no water. A possible road crosses the rocky terrain.")
    assert result["water"] == "no"
    assert result["road"] == "hedged"
    predictions = assertions_frame([("c1", "A river is visible without roads.")])
    human_record = {
        "caption_id": "c1",
        "water": "yes",
        "road": "no",
        "building": "no",
        "forest": "no",
        "snow": "no",
        "field": "no",
    }
    human = pd.DataFrame([human_record, human_record])
    validated = validate_judge(predictions, human)
    assert validated["exact_agreement"] == 1
    with pytest.raises(DataValidationError, match="identity"):
        validate_judge(predictions.assign(model="secret"), human)


def test_calibration_is_separate_by_mechanism() -> None:
    perfect = expected_calibration_error(np.array([0.0, 1.0]), np.array([False, True]), bins=2)
    assert perfect["ece"] == 0
    frame = pd.DataFrame(
        {"mechanism": ["verbal", "token"], "confidence": [1.0, 0.0], "correct": [True, False]}
    )
    result = calibration_by_mechanism(frame, bins=2)
    assert set(result) == {"verbal", "token"}
