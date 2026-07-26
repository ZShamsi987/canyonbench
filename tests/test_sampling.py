from __future__ import annotations

import pandas as pd

from canyonbench.pipeline.sampling import (
    assign_geographic_splits,
    assign_segments,
    hash_distance,
    haversine_m,
    sample_frames,
)


def test_haversine_and_hash_distance() -> None:
    assert haversine_m((36.8, -111.5), (36.8, -111.5)) == 0
    assert 80 < haversine_m((0, 0), (0, 0.001)) < 120
    assert hash_distance("0000000000000000", "ffffffffffffffff") == 64


def test_sampling_enforces_interval_then_change() -> None:
    frame = pd.DataFrame(
        [
            {"elapsed_s": 0, "lat": 36.8, "lon": -111.5, "phash": "0000000000000000"},
            {"elapsed_s": 10, "lat": 36.9, "lon": -111.5, "phash": "ffffffffffffffff"},
            {"elapsed_s": 60, "lat": 36.8, "lon": -111.5, "phash": "0000000000000000"},
            {"elapsed_s": 120, "lat": 36.81, "lon": -111.5, "phash": "0000000000000000"},
        ]
    )
    result = sample_frames(frame, min_interval_s=60, distance_m=500, phash_distance=8)
    assert result.elapsed_s.tolist() == [0, 120]
    assert result.sample_reason.tolist() == ["first", "distance"]


def test_segments_and_geographic_splits_do_not_leak() -> None:
    frame = pd.DataFrame(
        [
            {"elapsed_s": 0, "lat": 36.8, "lon": -111.5, "phase": "Launching"},
            {"elapsed_s": 60, "lat": 36.801, "lon": -111.5, "phase": "Launching"},
            {"elapsed_s": 1000, "lat": 36.9, "lon": -111.6, "phase": "Floating"},
        ]
    )
    segmented = assign_segments(frame, max_gap_s=300)
    assert segmented.segment_id.nunique() == 2
    split = assign_geographic_splits(segmented, block_size_m=5000)
    assert split.groupby("segment_id").split.nunique().max() == 1
    assert split.groupby("spatial_block").split.nunique().max() == 1


def test_segments_have_a_bounded_duration_and_preserve_block_splits() -> None:
    frame = pd.DataFrame(
        [
            {
                "elapsed_s": second,
                "lat": 36.8,
                "lon": -111.5 + index * 0.02,
                "phase": "Floating",
            }
            for index, second in enumerate(range(0, 1801, 300))
        ]
    )

    segmented = assign_segments(frame, max_duration_s=600)
    split = assign_geographic_splits(segmented, block_size_m=500)

    spans = split.groupby("segment_id").elapsed_s.agg(lambda values: values.max() - values.min())
    assert spans.max() < 600
    assert split.groupby("segment_id").split.nunique().max() == 1
    assert split.groupby("spatial_block").split.nunique().max() == 1
