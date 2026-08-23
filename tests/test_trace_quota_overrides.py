"""Tests for asymmetric per-class site quotas."""

from __future__ import annotations

import pytest

from canyonbench.trace.schemas import SiteQuota


def test_quota_defaults_to_an_even_split() -> None:
    row = SiteQuota(group="flight_corridor", per_class=20)
    for feature in ("water", "road", "field"):
        assert row.positives(feature) == 10
        assert row.negatives(feature) == 10


def test_quota_override_applies_to_one_class_only() -> None:
    row = SiteQuota(group="flight_corridor", per_class=20, positive_overrides={"road": 7})
    assert (row.positives("road"), row.negatives("road")) == (7, 13)
    # The class total is preserved, so the site count does not change.
    assert row.positives("road") + row.negatives("road") == 20
    # Water and field keep the balanced design.
    assert (row.positives("water"), row.negatives("water")) == (10, 10)
    assert (row.positives("field"), row.negatives("field")) == (10, 10)


def test_quota_override_cannot_exceed_the_class_total() -> None:
    with pytest.raises(ValueError, match="exceeds per_class"):
        SiteQuota(group="cross_biome", per_class=8, positive_overrides={"road": 9})
