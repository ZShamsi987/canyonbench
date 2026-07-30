"""V5 image controls and deterministic non-VLM reference baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from canyonbench.io import write_json


def materialize_image_controls(
    image_path: Path,
    unrelated_path: Path,
    output_dir: Path,
    *,
    seed: int,
) -> dict[str, Path]:
    """Create blank, block-shuffled, and unrelated-image controls."""

    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    with Image.open(unrelated_path) as source:
        unrelated = np.asarray(source.convert("RGB").resize((image.shape[1], image.shape[0])))
    blank = np.full_like(image, int(np.mean(image)))
    grid = 8
    blocks: list[np.ndarray] = []
    height, width = image.shape[:2]
    for row in range(grid):
        y0, y1 = round(row * height / grid), round((row + 1) * height / grid)
        for column in range(grid):
            x0, x1 = round(column * width / grid), round((column + 1) * width / grid)
            blocks.append(image[y0:y1, x0:x1].copy())
    order = np.random.default_rng(seed).permutation(len(blocks))
    shuffled = np.empty_like(image)
    position = 0
    for row in range(grid):
        y0, y1 = round(row * height / grid), round((row + 1) * height / grid)
        for column in range(grid):
            x0, x1 = round(column * width / grid), round((column + 1) * width / grid)
            block = blocks[int(order[position])]
            shuffled[y0:y1, x0:x1] = cv2.resize(
                block, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA
            )
            position += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    values = {"blank": blank, "shuffled": shuffled, "unrelated": unrelated}
    paths: dict[str, Path] = {}
    for name, value in values.items():
        paths[name] = output_dir / f"{name}.png"
        Image.fromarray(value).save(paths[name], optimize=True)
    write_json(
        output_dir / "manifest.json",
        {
            "source_image": str(image_path),
            "unrelated_source": str(unrelated_path),
            "seed": seed,
            "controls": {name: str(path) for name, path in paths.items()},
        },
    )
    return paths


def deterministic_baselines(index: list[dict[str, Any]]) -> pd.DataFrame:
    """Always/base-rate/geographic-prior predictions with dev-only fitting."""

    clean = pd.DataFrame([row for row in index if row.get("variant") == "clean"])
    if clean.empty:
        raise ValueError("dataset index contains no clean views")
    clean["label"] = clean["case_type"].isin(["positive", "extinction"]).astype(int)
    development = clean.loc[clean["split"] == "development"]
    rows: list[dict[str, Any]] = []
    for feature, feature_rows in clean.groupby("target_class", observed=True):
        dev = development.loc[development["target_class"] == feature]
        base_rate = float(dev["label"].mean()) if len(dev) else 0.5
        majority = int(base_rate >= 0.5)
        geo_model: LogisticRegression | None = None
        if len(dev) >= 4 and dev["label"].nunique() == 2:
            geo_model = LogisticRegression(random_state=2026).fit(
                dev[["longitude", "latitude"]], dev["label"]
            )
        for _, row in feature_rows.iterrows():
            geographic_probability = (
                float(
                    geo_model.predict_proba(
                        pd.DataFrame(
                            [[row["longitude"], row["latitude"]]],
                            columns=["longitude", "latitude"],
                        )
                    )[0, 1]
                )
                if geo_model is not None
                else base_rate
            )
            for name, prediction, probability in (
                ("always_yes", 1, 1.0),
                ("always_no", 0, 0.0),
                ("base_rate", majority, base_rate),
                ("geographic_prior", int(geographic_probability >= 0.5), geographic_probability),
            ):
                rows.append(
                    {
                        "baseline": name,
                        "site_id": row["site_id"],
                        "view_id": row["view_id"],
                        "split": row["split"],
                        "target_class": feature,
                        "label": int(row["label"]),
                        "prediction": prediction,
                        "yes_probability": probability,
                    }
                )
    return pd.DataFrame(rows)
