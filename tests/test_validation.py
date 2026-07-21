from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

from canyonbench.validation import validate_release


def valid_release(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    (root / "frames").mkdir(parents=True)
    (root / "masks").mkdir()
    image = "img_006806.jpg"
    Image.new("RGB", (8, 8), "brown").save(root / "frames" / image)
    Image.new("L", (8, 8), 0).save(root / "masks" / "img_006806.png")
    pd.DataFrame(
        [
            {
                "image": image,
                "elapsed_s": 6806,
                "phase": "Floating",
                "lat": 36.8,
                "lon": -111.5,
                "alt_m": 23000,
                "segment_id": "seg_1",
                "spatial_block": "block_1",
                "split": "test",
                "vegetation_fraction": 0,
                "registration_reliable": False,
                "vegetation_cells": None,
                "water": "no",
                "road": "no",
                "building": "no",
                "forest": "no",
                "snow": "no",
                "field": "no",
            }
        ]
    ).to_csv(root / "frames.csv", index=False)
    (root / "release_manifest.json").write_text(
        json.dumps({"n_frames": 1, "effective_segment_count": 1}), encoding="utf-8"
    )
    return root


def test_valid_release_has_no_issues(tmp_path: Path) -> None:
    assert validate_release(valid_release(tmp_path)) == []


def test_release_detects_zero_gps_and_missing_files(tmp_path: Path) -> None:
    root = valid_release(tmp_path)
    frame = pd.read_csv(root / "frames.csv")
    frame.loc[0, "lat"] = 0
    frame.to_csv(root / "frames.csv", index=False)
    (root / "frames" / "img_006806.jpg").unlink()
    codes = {issue.code for issue in validate_release(root)}
    assert {"zero_gps", "missing_image_file"} <= codes
