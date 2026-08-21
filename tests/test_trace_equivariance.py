"""Tests for viewpoint equivariance of grounding."""

from __future__ import annotations

import numpy as np
import pytest

from canyonbench.trace.equivariance import (
    CellGrid,
    ViewPair,
    answer_invariance,
    cell_centres,
    grounding_equivariance,
    map_cells,
)

GRID = CellGrid(grid_size=4, width_px=1024, height_px=1024)


def _identity_pair() -> ViewPair:
    return ViewPair(source_to_a=np.eye(3), source_to_b=np.eye(3))


def test_cell_bounds_match_the_metrics_convention() -> None:
    # Cell "0,0" is the top-left quarter-width block, as _cell_area assumes.
    assert GRID.bounds("0,0") == (0.0, 0.0, 256.0, 256.0)
    assert GRID.bounds("3,3") == (768.0, 768.0, 1024.0, 1024.0)


def test_locate_inverts_bounds_for_every_cell() -> None:
    for cell in GRID.cells:
        x0, y0, x1, y1 = GRID.bounds(cell)
        assert GRID.locate((x0 + x1) / 2, (y0 + y1) / 2) == cell


def test_locate_rejects_points_outside_the_frame() -> None:
    assert GRID.locate(-1, 10) is None
    assert GRID.locate(10, 1024) is None


def test_identity_transform_maps_cells_to_themselves() -> None:
    cells = ["1,1", "2,3"]
    mapped, lost = map_cells(cells, _identity_pair(), GRID, GRID)
    assert mapped == set(cells)
    assert lost == 0


def test_view_transform_is_composed_not_estimated() -> None:
    # A view shifted by half a cell in source space must compose to that shift.
    source_to_a = np.eye(3)
    source_to_b = np.array([[1, 0, 256.0], [0, 1, 0], [0, 0, 1]])
    pair = ViewPair(source_to_a=source_to_a, source_to_b=source_to_b)
    mapped, lost = map_cells(["1,0"], pair, GRID, GRID)
    assert mapped == {"1,1"}
    assert lost == 0


def test_cells_leaving_the_frame_are_counted_not_dropped() -> None:
    pair = ViewPair(np.eye(3), np.array([[1, 0, 4096.0], [0, 1, 0], [0, 0, 1]]))
    mapped, lost = map_cells(["0,0", "1,1"], pair, GRID, GRID)
    assert mapped == set()
    assert lost == 2


def test_perfect_equivariance_beats_its_random_baseline() -> None:
    cells = ["1,1", "1,2"]
    score = grounding_equivariance(cells, cells, _identity_pair(), GRID, GRID)
    assert score["equivariance"] == pytest.approx(1.0)
    assert score["lift"] > 0.5
    assert score["cells_left_frame"] == 0


def test_disjoint_grounding_scores_zero_equivariance() -> None:
    score = grounding_equivariance(["0,0"], ["3,3"], _identity_pair(), GRID, GRID)
    assert score["equivariance"] == pytest.approx(0.0)
    assert score["lift"] <= 0


def test_naming_every_cell_does_not_earn_credit() -> None:
    # A model that covers the frame scores high raw overlap but no lift, which is
    # the whole reason the baseline is reported beside the score.
    everything = GRID.cells
    score = grounding_equivariance(everything, everything, _identity_pair(), GRID, GRID)
    assert score["equivariance"] == pytest.approx(1.0)
    assert score["lift"] == pytest.approx(0.0, abs=1e-9)


def test_points_behind_the_horizon_do_not_manufacture_coordinates() -> None:
    # A homography with a vanishing denominator must lose the cell, not place it.
    degenerate = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    pair = ViewPair(np.eye(3), degenerate)
    mapped, lost = map_cells(["2,2"], pair, GRID, GRID)
    assert mapped == set()
    assert lost == 1


def test_cell_centres_shape_is_stable_when_empty() -> None:
    assert cell_centres(GRID, []).shape == (0, 2)


def test_answer_invariance_counts_pairs_not_views() -> None:
    assert answer_invariance({"a": "yes", "b": "yes", "c": "yes"})["invariance"] == 1.0
    mixed = answer_invariance({"a": "yes", "b": "no", "c": "yes"})
    assert mixed["invariance"] == pytest.approx(1 / 3)
    assert mixed["pairs"] == 3


def test_answer_invariance_is_undefined_for_a_single_view() -> None:
    assert np.isnan(answer_invariance({"a": "yes"})["invariance"])
