"""Flatten nested benchmark metrics into paper-friendly tidy tables."""

from __future__ import annotations

from typing import Any

import pandas as pd


def flatten_metrics(metrics: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, sections in metrics.get("models", {}).items():
        for probe_variant, values in sections.items():
            probe, _, variant = probe_variant.partition(":")
            for metric, value in values.items():
                if isinstance(value, (str, int, float)) or value is None:
                    rows.append(
                        {
                            "model": model,
                            "probe": probe,
                            "variant": variant,
                            "metric": metric,
                            "value": value,
                        }
                    )
    return pd.DataFrame(rows)
