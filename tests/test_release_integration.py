from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

from canyonbench.groundtruth.masks import grid_labels
from canyonbench.groundtruth.release import build_release
from canyonbench.io import read_json, write_jsonl
from canyonbench.validation import validate_release


def test_build_release_materializes_complete_valid_release(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "labels/adjudicated").mkdir(parents=True)
    (data / "masks/adjudicated").mkdir(parents=True)
    (data / "registration").mkdir()
    source_frame = tmp_path / "img_006806.jpg"
    Image.new("RGB", (8, 8), "green").save(source_frame)
    mask_path = data / "masks/adjudicated/img_006806.png"
    mask_image = Image.new("L", (8, 8), 0)
    for y in range(2):
        for x in range(2):
            mask_image.putpixel((x, y), 255)
    mask_image.save(mask_path)
    import numpy as np

    cells = grid_labels(np.asarray(mask_image) == 255)
    presence = {
        "image": source_frame.name,
        "annotator": "ADJ",
        "water": "no",
        "road": "no",
        "building": "no",
        "forest": "no",
        "snow": "no",
        "field": "no",
    }
    quality = {
        "image": source_frame.name,
        "annotator": "ADJ",
        "cloud": "none",
        "clarity": "clear",
        "balloon": "none",
        "sharpness": "sharp",
        "exposure": "ok",
        "glare": "none",
    }
    write_jsonl(data / "labels/adjudicated/presence.jsonl", [presence])
    write_jsonl(data / "labels/adjudicated/quality.jsonl", [quality])
    write_jsonl(
        data / "labels/adjudicated/grid.jsonl",
        [{"image": source_frame.name, "annotator": "ADJ", "cells": cells}],
    )
    pd.DataFrame(
        [{"image": source_frame.name, "n_points": 8, "holdout_rmse_m": 10, "reliable": True}]
    ).to_csv(data / "registration/residuals.csv", index=False)
    sampled = pd.DataFrame(
        [
            {
                "image": source_frame.name,
                "image_path": str(source_frame),
                "elapsed_s": 6806,
                "phase": "Floating",
                "lat": 36.8,
                "lon": -111.5,
                "alt_m": 23000,
                "segment_id": "seg_1",
                "spatial_block": "block_1",
                "split": "test",
                "sampling_raw_count": 10,
            }
        ]
    )
    output = tmp_path / "release"
    master = build_release(sampled, data, output)
    assert master.vegetation_fraction.iloc[0] == 4 / 64
    assert (output / "frames/img_006806.jpg").exists()
    assert read_json(output / "release_manifest.json")["raw_frame_count"] == 10
    assert validate_release(output) == []
    assert len(json.loads((output / "release_manifest.json").read_text())) >= 4
