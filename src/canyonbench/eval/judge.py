"""Secondary caption judge, rule fallback, and human validation metrics."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

from canyonbench.constants import FEATURES
from canyonbench.eval.metrics import classification_counts, safe_divide
from canyonbench.exceptions import DataValidationError

TERMS: dict[str, tuple[str, ...]] = {
    "water": ("water", "river", "lake", "reservoir", "stream"),
    "road": ("road", "trail", "highway", "track", "path"),
    "building": ("building", "structure", "rooftop", "dam", "tower"),
    "forest": ("forest", "woodland", "tree canopy", "dense trees"),
    "snow": ("snow", "ice", "snowfield"),
    "field": ("cultivated field", "farmland", "agricultural field", "crop field"),
}
NEGATION = re.compile(r"\b(no|not|without|absent|lacks?|neither|nor)\b", re.IGNORECASE)
HEDGE = re.compile(
    r"\b(possible|possibly|perhaps|maybe|may|might|appears?|likely)\b", re.IGNORECASE
)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?;])\s+|\n+", text) if part.strip()]


def rule_based_assertions(caption: str) -> dict[str, str]:
    """Conservative fixed-ontology fallback: yes, no, or hedged per feature."""

    output = dict.fromkeys(FEATURES, "no")
    for sentence in _sentences(caption):
        lower = sentence.lower()
        for feature, terms in TERMS.items():
            matches = [
                match
                for term in terms
                if (match := re.search(rf"\b{re.escape(term)}s?\b", lower)) is not None
            ]
            if not matches:
                continue
            # Scope cues to the short phrase preceding the ontology term. Sentence-wide
            # negation would incorrectly make "river ... without roads" negate river.
            match = min(matches, key=lambda value: value.start())
            prefix_words = lower[: match.start()].split()[-5:]
            prefix = " ".join(prefix_words)
            if HEDGE.search(prefix):
                output[feature] = "hedged"
            elif NEGATION.search(prefix):
                output[feature] = "no"
            else:
                output[feature] = "yes"
    return output


def validate_judge(
    predictions: pd.DataFrame,
    human_annotations: pd.DataFrame,
) -> dict[str, Any]:
    """Validate assertions without using image or evaluated-model identity."""

    required = {"caption_id", *FEATURES}
    for label, frame in (("predictions", predictions), ("human annotations", human_annotations)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise DataValidationError(f"Judge {label} missing fields: {missing}")
    if "model" in predictions.columns:
        raise DataValidationError(
            "Blinded judge predictions must not contain evaluated model identity"
        )
    annotation_counts = human_annotations.groupby("caption_id").size()
    if (annotation_counts < 2).any():
        raise DataValidationError("Judge validation requires two human annotations per caption")

    def mode_or_no(values: pd.Series) -> str:
        modes = values.mode()
        return str(modes.iloc[0]) if len(modes) else "no"

    truth = human_annotations.groupby("caption_id", sort=False)[list(FEATURES)].agg(mode_or_no)
    merged = predictions.merge(truth, on="caption_id", suffixes=("_pred", "_truth"))
    per_feature: dict[str, Any] = {}
    all_actual: list[bool] = []
    all_predicted: list[bool] = []
    for feature in FEATURES:
        actual = merged[f"{feature}_truth"].eq("yes").tolist()
        predicted = merged[f"{feature}_pred"].eq("yes").tolist()
        counts = classification_counts(actual, predicted)
        all_actual.extend(actual)
        all_predicted.extend(predicted)
        per_feature[feature] = {
            **counts,
            "precision": safe_divide(counts["tp"], counts["tp"] + counts["fp"]),
            "recall": safe_divide(counts["tp"], counts["tp"] + counts["fn"]),
            "agreement": float((merged[f"{feature}_pred"] == merged[f"{feature}_truth"]).mean()),
        }
    total = classification_counts(all_actual, all_predicted)
    return {
        "n_captions": len(merged),
        "per_feature": per_feature,
        "micro_precision": safe_divide(total["tp"], total["tp"] + total["fp"]),
        "micro_recall": safe_divide(total["tp"], total["tp"] + total["fn"]),
        "exact_agreement": float(
            merged.apply(
                lambda row: all(
                    row[f"{feature}_pred"] == row[f"{feature}_truth"] for feature in FEATURES
                ),
                axis=1,
            ).mean()
        ),
    }


def assertions_frame(captions: Iterable[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"caption_id": caption_id, **rule_based_assertions(text)} for caption_id, text in captions]
    )
