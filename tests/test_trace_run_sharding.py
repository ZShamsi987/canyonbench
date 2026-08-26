"""Sharding must partition the run exactly: no duplicates, nothing dropped."""

from __future__ import annotations

import zlib


def _in_shard(site_id: str, view_id: str, index: int, count: int) -> bool:
    """Mirror of the runner's partition rule."""

    return zlib.crc32(f"{site_id}/{view_id}".encode()) % count == index


VIEWS = [
    (f"site_{site:04d}", f"view_a{alt}km_{geometry}")
    for site in range(1, 41)
    for alt in (3, 8, 16, 24)
    for geometry in ("nadir", "oblique")
]


def test_shards_partition_every_view_exactly_once() -> None:
    for count in (2, 3, 4, 6, 8):
        seen: list[tuple[str, str]] = []
        for index in range(count):
            seen += [v for v in VIEWS if _in_shard(*v, index, count)]
        # Union is the whole run, and no view appears in two shards.
        assert sorted(seen) == sorted(VIEWS)
        assert len(seen) == len(set(seen))


def test_a_view_keeps_all_its_tiers_in_one_shard() -> None:
    # Tier B's self-evidence sequences consume that view's own Tier A cells, so
    # the partition must depend only on the view, never on tier or request.
    for count in (3, 5):
        for site, view in VIEWS:
            owners = {i for i in range(count) if _in_shard(site, view, i, count)}
            assert len(owners) == 1


def test_partition_is_stable_across_processes() -> None:
    # crc32 is fixed by the algorithm, unlike hash(), which is salted per process.
    assert zlib.crc32(b"site_0001/view_a3km_nadir") == 2658372034


def test_single_shard_is_the_whole_run() -> None:
    assert [v for v in VIEWS if _in_shard(*v, 0, 1)] == VIEWS


def test_shards_are_roughly_balanced() -> None:
    count = 4
    sizes = [len([v for v in VIEWS if _in_shard(*v, i, count)]) for i in range(count)]
    # A badly skewed partition would leave one worker running long after the
    # others finish, which is the whole point of sharding.
    assert max(sizes) - min(sizes) <= len(VIEWS) * 0.1
