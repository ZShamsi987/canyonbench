from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from canyonbench.exceptions import DataValidationError
from canyonbench.groundtruth.agreement import cohen_kappa, majority_vote
from canyonbench.groundtruth.masks import (
    dice,
    grid_labels,
    load_binary_mask,
    small_components,
    vegetation_fraction,
)
from canyonbench.groundtruth.vari import calibrate_vari, intersection_over_union, vari
from canyonbench.groundtruth.weak_labels import compare_weak_label_set, mask_comparison


def test_mask_contract_grid_and_dice(tmp_path: Path) -> None:
    array = np.zeros((8, 8), dtype=np.uint8)
    array[:2, :2] = 255
    path = tmp_path / "mask.png"
    Image.fromarray(array).save(path)
    mask = load_binary_mask(path, (8, 8))
    assert vegetation_fraction(mask) == 4 / 64
    cells = grid_labels(mask, n=4, threshold=0.01)
    assert cells["0,0"] is True
    assert sum(cells.values()) == 1
    assert dice(mask, mask) == 1
    assert small_components(mask) == []


def test_mask_rejects_nonbinary(tmp_path: Path) -> None:
    path = tmp_path / "bad.png"
    Image.fromarray(np.full((2, 2), 127, dtype=np.uint8)).save(path)
    with pytest.raises(DataValidationError, match="values"):
        load_binary_mask(path)


def test_small_component_detection() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True
    mask[2:4, 2:4] = True
    assert small_components(mask) == [1]


def test_vari_calibration_selects_best_candidate() -> None:
    image = np.array([[[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]]], dtype=np.float32)
    index = vari(image)
    target = np.array([[True, False]])
    calibration = calibrate_vari([index], [target], thresholds=np.array([-1.0, 0.0, 1.0]))
    assert calibration.threshold == 0
    assert calibration.mean_iou == 1
    assert intersection_over_union(target, target) == 1


def test_agreement_and_vote() -> None:
    assert cohen_kappa(["yes", "no", "yes"], ["yes", "no", "yes"]) == 1
    assert majority_vote(["yes", "no"]) == "uncertain"
    assert majority_vote(["yes", "yes", "no"]) == "yes"


def test_weak_label_comparison_stays_explicit() -> None:
    human = np.array([[True, False], [False, False]])
    weak = np.array([[True, True], [False, False]])
    metrics = mask_comparison(weak, human)
    assert metrics["precision"] == 0.5
    frame = compare_weak_label_set(["img_006806.jpg"], [weak], [human], source="VARI")
    assert frame.weak_label_source.iloc[0] == "VARI"
