from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from canyonbench.io import read_json
from canyonbench.registration.batch import register_manifest
from canyonbench.registration.overlay import warp_reference_to_frame


def test_batch_registration_and_identity_overlay(tmp_path: Path) -> None:
    points = tmp_path / "points.csv"
    image = np.array(
        [[0, 0], [10, 0], [0, 10], [10, 10], [5, 5], [2, 8], [8, 2], [8, 8]],
        dtype=float,
    )
    mapped = image * 2 + 100
    pd.DataFrame(
        {
            "image_x": image[:, 0],
            "image_y": image[:, 1],
            "map_x": mapped[:, 0],
            "map_y": mapped[:, 1],
            "role": ["fit"] * 6 + ["holdout"] * 2,
        }
    ).to_csv(points, index=False)
    manifest = pd.DataFrame(
        [{"image": "img_006806.jpg", "points_path": str(points), "threshold_m": np.nan}]
    )
    output = tmp_path / "registration"
    residuals = register_manifest(manifest, output_dir=output, default_threshold_m=1)
    assert bool(residuals.reliable.iloc[0]) is True
    matrix = read_json(output / "img_006806.homography.json")
    assert len(matrix["H"]) == 3

    reference = np.arange(16, dtype=np.uint8).reshape(4, 4)
    warped = warp_reference_to_frame(
        reference, np.eye(3), image_width=4, image_height=4, nearest=True
    )
    np.testing.assert_array_equal(warped, reference)
