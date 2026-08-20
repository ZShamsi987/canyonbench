#!/usr/bin/env python3
"""Cheaply shortlist field-negative seeds before the authoritative CDL screen.

WHY THIS EXISTS.  The registered field-negative rule admits a site only when no
cultivated pixel falls inside the buffered maximum camera footprint.  Measured
against the frozen pool, that rule accepts roughly one candidate in ten, so the
quota is met by searching a large pool rather than by relaxing the rule.  The
authoritative screen cannot do that search: it costs one CropScape request or one
national-archive window per candidate, it has repeatedly been throttled to a 503,
and reading whole archives is what the login node kills for memory.

This stage answers the same question against the USDA CDL Cultivated Layer served
as cloud-optimised GeoTIFFs by Microsoft Planetary Computer.  Each check is a
windowed range read of one footprint: no archive, no bulk download, a few hundred
kilobytes and about a second, and safe to run several at a time.  A pool of
thousands is therefore screenable in under an hour instead of a day.

WHAT THIS IS NOT.  The Cultivated Layer published there ends at 2021, so it can
disagree with the CDL year that G1 will require for a 2022 or 2023 NAIP site.
This stage is consequently a *pre-filter*, never an authority: it only decides
which candidates are worth an authoritative check.  Every candidate it accepts
must still pass ``screen_field_negatives.py`` against the CDL year acquisition
will actually use, and that script remains the sole producer of vetted seeds.
Because the shortlist is a small fraction of the pool, the authoritative pass
shrinks from thousands of requests to a few dozen.

Land newly cultivated after 2021 is caught by the authoritative pass.  Land
retired from cultivation after 2021 is rejected here and never revisited; that
costs yield, never correctness, which is the safe direction for a pre-filter.

    python scripts/adroit/prefilter_field_negatives.py \
        configs/trace_sources.yaml \
        "$CANYONBENCH_DATA/manifests/trace_candidates_expanded.yaml" \
        "$CANYONBENCH_DATA/manifests/field_negative_shortlist.yaml" \
        "$CANYONBENCH_DATA/reports/field-negative-prefilter.json" \
        --state "$CANYONBENCH_DATA/cache/field-negative-prefilter.json"
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import rasterio  # type: ignore[import-untyped]
from rasterio.warp import transform_bounds  # type: ignore[import-untyped]
from rasterio.windows import Window, from_bounds  # type: ignore[import-untyped]

from canyonbench.exceptions import DataValidationError
from canyonbench.io import read_json, write_json
from canyonbench.trace.acquisition import (
    NAIP_STAC_SEARCH,
    _Grid,
    _http_client,
    _signed_planetary_computer_href,
    write_candidate_manifest,
)
from canyonbench.trace.config import load_candidate_seeds, load_source_acquisition_config
from canyonbench.trace.schemas import CandidateSeed

CULTIVATED_COLLECTION = "usda-cdl"
CULTIVATED_ASSET = "cultivated"
# The Cultivated Layer is a two-value raster: 1 non-cultivated, 2 cultivated.
CULTIVATED_VALUE = 2
# The most recent year Planetary Computer publishes for this layer.  A newer
# year is preferred automatically if the collection is ever extended.
PREFILTER_YEAR = 2021


def _cultivated_href(client: httpx.Client, longitude: float, latitude: float, year: int) -> str:
    response = client.post(
        NAIP_STAC_SEARCH,
        json={
            "collections": [CULTIVATED_COLLECTION],
            "intersects": {"type": "Point", "coordinates": [longitude, latitude]},
            "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
            "limit": 10,
        },
        timeout=httpx.Timeout(60, connect=20),
    )
    response.raise_for_status()
    for feature in response.json().get("features", []):
        asset = (feature.get("assets") or {}).get(CULTIVATED_ASSET)
        if asset and asset.get("href"):
            return str(asset["href"])
    raise DataValidationError(
        f"No {CULTIVATED_COLLECTION}/{CULTIVATED_ASSET} coverage at {longitude},{latitude}"
    )


def _cultivated_pixels(path: str, bounds_wgs84: tuple[float, float, float, float]) -> int:
    """Count cultivated pixels inside one footprint without reading the tile."""

    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise DataValidationError(f"Cultivated Layer tile has no CRS: {path}")
        bounds = transform_bounds("EPSG:4326", dataset.crs, *bounds_wgs84, densify_pts=21)
        window = from_bounds(*bounds, transform=dataset.transform)
        window = (
            window.round_offsets()
            .round_lengths()
            .intersection(Window(0, 0, dataset.width, dataset.height))
        )
        if window.width <= 0 or window.height <= 0:
            raise DataValidationError(f"Cultivated Layer does not cover {bounds_wgs84}")
        return int((dataset.read(1, window=window) == CULTIVATED_VALUE).sum())


def _check(seed: CandidateSeed, *, config: Any, year: int) -> dict[str, Any]:
    """Count cultivated pixels over the same support G2 will decide on.

    ``evaluate_site`` zeroes every mask outside the buffered maximum camera
    footprint before ``negative_clear`` is computed, so the screened extent is
    ``negative_screen_half_extent_m`` plus the registered 30 m safety buffer --
    not the wider source package, which contains evidence no camera can see.
    """

    grid = _Grid(
        seed.longitude,
        seed.latitude,
        config.working_resolution_m,
        config.negative_screen_half_extent_m + 30,
    )
    with _http_client() as client:
        signed = _signed_planetary_computer_href(
            client,
            _cultivated_href(client, seed.longitude, seed.latitude, year),
        )
    return {
        "candidate_id": seed.candidate_id,
        "region_id": seed.region_id,
        "group": seed.group,
        "cultivated_px": _cultivated_pixels(signed, grid.wgs84_bounds),
        "prefilter_year": year,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--state",
        type=Path,
        help="Resumable per-candidate results; reused so a rerun costs no requests.",
    )
    parser.add_argument("--group", choices=("flight_corridor", "regional_ood", "cross_biome"))
    parser.add_argument("--year", type=int, default=PREFILTER_YEAR)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--limit",
        type=int,
        help="Stop after this many candidates; the state file resumes the rest.",
    )
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    config = load_source_acquisition_config(arguments.config)
    seeds = [
        seed
        for seed in load_candidate_seeds(arguments.candidates)
        if seed.target_class == "field"
        and seed.case_type == "negative"
        and (arguments.group is None or seed.group == arguments.group)
    ]
    done: dict[str, dict[str, Any]] = {}
    if arguments.state and arguments.state.is_file():
        done = {row["candidate_id"]: row for row in read_json(arguments.state)["results"]}
    pending = [seed for seed in seeds if seed.candidate_id not in done]
    if arguments.limit is not None:
        pending = pending[: arguments.limit]
    print(f"{len(seeds)} field negatives; {len(done)} cached; checking {len(pending)}", flush=True)
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        futures = {
            pool.submit(_check, seed, config=config, year=arguments.year): seed for seed in pending
        }
        for index, future in enumerate(futures, start=1):
            seed = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # recorded, never silently dropped
                failures.append({"candidate_id": seed.candidate_id, "error": repr(exc)})
                print(f"[{index}/{len(pending)}] failed {seed.candidate_id}", flush=True)
                continue
            done[row["candidate_id"]] = row
            verdict = "shortlisted" if row["cultivated_px"] == 0 else "rejected"
            print(
                f"[{index}/{len(pending)}] {verdict} {row['candidate_id']} "
                f"cultivated_px={row['cultivated_px']}",
                flush=True,
            )
            if arguments.state and index % 25 == 0:
                write_json(arguments.state, {"results": sorted(done.values(), key=str)})
    if arguments.state:
        write_json(arguments.state, {"results": sorted(done.values(), key=str)})
    shortlisted = [
        seed
        for seed in seeds
        if seed.candidate_id in done and done[seed.candidate_id]["cultivated_px"] == 0
    ]
    write_candidate_manifest(arguments.output, shortlisted)
    by_group: dict[str, int] = {}
    for seed in shortlisted:
        by_group[seed.group] = by_group.get(seed.group, 0) + 1
    checked = len(done)
    write_json(
        arguments.report,
        {
            "schema_version": "4.0.0",
            "authority": "pre-filter only; screen_field_negatives.py remains authoritative",
            "prefilter_year": arguments.year,
            "screened_half_extent_m": config.negative_screen_half_extent_m + 30,
            "candidates": len(seeds),
            "checked": checked,
            "shortlisted": len(shortlisted),
            "shortlisted_by_group": by_group,
            "shortlist_rate": round(len(shortlisted) / checked, 4) if checked else 0.0,
            "failures": failures,
        },
    )
    print(
        f"shortlisted {len(shortlisted)}/{checked} -> {arguments.output} "
        f"({len(failures)} failures)",
        flush=True,
    )
    print("Confirm every shortlisted seed with screen_field_negatives.py before acquisition.")


if __name__ == "__main__":
    main()
