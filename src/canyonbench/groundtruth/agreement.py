"""Inter-annotator agreement and conservative categorical adjudication helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from canyonbench.exceptions import DataValidationError


def cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float:
    if len(left) != len(right) or not left:
        raise DataValidationError("Kappa requires equally sized, non-empty label sequences")
    categories = sorted(set(left) | set(right))
    observed = float(np.mean(np.asarray(left) == np.asarray(right)))
    expected = sum(
        (left.count(category) / len(left)) * (right.count(category) / len(right))
        for category in categories
    )
    if np.isclose(expected, 1):
        return 1.0 if np.isclose(observed, 1) else 0.0
    return float((observed - expected) / (1 - expected))


def majority_vote(values: Iterable[str], *, tie_value: str = "uncertain") -> str:
    counts = Counter(values)
    if not counts:
        raise DataValidationError("Cannot adjudicate an empty label set")
    most_common = counts.most_common()
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return tie_value
    return most_common[0][0]


def agreement_by_field(
    records: list[dict[str, Any]],
    *,
    key_field: str,
    annotator_field: str,
    label_fields: Iterable[str],
) -> dict[str, float]:
    by_annotator: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_annotator.setdefault(str(record[annotator_field]), {})[str(record[key_field])] = record
    if len(by_annotator) != 2:
        raise DataValidationError("Pairwise agreement currently requires exactly two annotators")
    first_id, second_id = sorted(by_annotator)
    shared = sorted(set(by_annotator[first_id]) & set(by_annotator[second_id]))
    if not shared:
        raise DataValidationError("Annotators have no shared records")
    return {
        field: cohen_kappa(
            [str(by_annotator[first_id][key][field]) for key in shared],
            [str(by_annotator[second_id][key][field]) for key in shared],
        )
        for field in label_fields
    }
