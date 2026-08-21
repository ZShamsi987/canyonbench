#!/usr/bin/env python3
"""Shortlist water-positive seeds that can actually clear G2 and G4.

WHY THIS EXISTS.  G2 requires two-source agreement at or above
``minimum_consensus_fraction`` and G4 requires the independent detector to find
the feature.  Measured across the acquired cohort, both hold for large water and
neither holds for small water: accepted sites carry 12-531 km2 of NHD water and
score IoU 0.92-0.98, while rejected sites mostly carry under 1 km2 and score
0.00-0.16.  NHD and TIGER agree almost perfectly on a reservoir and barely at all
on a scattered stock pond.  Regional and cross-biome strata are arid by
construction, so they returned 0 of 12 and 0 of 7 accepted water positives.

Two corrections follow, and this stage applies both before any imagery is fetched.

DRY FEATURES ARE NOT SURFACE WATER.  NHD files Playa, Wash, and Swamp/Marsh as
hydrographic features, and in drylands they dominate: one acquired site's water
mask is 100 percent playa -- 187 km2 of dry lake bed carrying a positive water
label.  Neither TIGER nor the land-cover detector calls those water, which is
why G2 and G4 rejected them together.  They are excluded here by FTYPE.

SMALL WATER CANNOT CLEAR CONSENSUS.  A candidate whose remaining wet area falls
below ``--minimum-area-km2`` is not shortlisted, because the registered gates
will reject it on arrival and the acquisition cost is wasted.

This is a pre-filter, never an authority: the registered gates still decide every
site after acquisition.  It only stops the pipeline spending 20-45 minutes of
imagery download on a candidate whose own vector geometry already predicts a
rejection.

    python scripts/adroit/prefilter_water_positives.py \\
        configs/trace_sources.yaml \\
        "$CANYONBENCH_DATA/manifests/trace_candidates_expanded_v2.yaml" \\
        "$CANYONBENCH_DATA/manifests/water_positive_shortlist.yaml" \\
        "$CANYONBENCH_DATA/reports/water-positive-prefilter.json" \\
        --state "$CANYONBENCH_DATA/cache/water-positive-prefilter.json"
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from canyonbench.io import read_json, write_json
from canyonbench.trace.acquisition import (
    NHD_SERVICE,
    WATER_LAYERS,
    _Grid,
    _http_client,
    write_candidate_manifest,
)
from canyonbench.trace.config import load_candidate_seeds, load_source_acquisition_config
from canyonbench.trace.schemas import CandidateSeed

# NHD FTYPE codes that are hydrographic features but not open surface water.
# 361 Playa, 484 Wash, 466 Swamp/Marsh, 403 Inundation Area, 378 Ice Mass.
DRY_FTYPES = frozenset({361, 484, 466, 403, 378})
# Below this the two water sources stop agreeing and G2 rejects the site.
DEFAULT_MINIMUM_AREA_KM2 = 2.0


def _water_area_km2(
    client: httpx.Client, grid: _Grid
) -> tuple[float, float, dict[int, float]]:
    """Return (wet area, dry-feature area, per-FTYPE area) over one footprint."""

    west, south, east, north = grid.wgs84_bounds
    areas: Counter[int] = Counter()
    for layer in WATER_LAYERS:
        response = client.get(
            f"{NHD_SERVICE}/{layer}/query",
            params={
                "f": "json",
                "where": "1=1",
                "geometryType": "esriGeometryEnvelope",
                "geometry": f"{west},{south},{east},{north}",
                "inSR": "4326",
                "outFields": "FTYPE,AREASQKM",
                "returnGeometry": "false",
                "resultRecordCount": 4000,
            },
            timeout=httpx.Timeout(60, connect=20),
        )
        response.raise_for_status()
        for feature in response.json().get("features", []):
            attributes = feature.get("attributes") or {}
            areas[attributes.get("FTYPE")] += float(attributes.get("AREASQKM") or 0.0)
    dry = sum(value for key, value in areas.items() if key in DRY_FTYPES)
    wet = sum(value for key, value in areas.items() if key not in DRY_FTYPES)
    return wet, dry, dict(areas)


def _check(seed: CandidateSeed, *, config: Any) -> dict[str, Any]:
    grid = _Grid(
        seed.longitude,
        seed.latitude,
        config.working_resolution_m,
        config.negative_screen_half_extent_m + 30,
    )
    with _http_client() as client:
        wet, dry, areas = _water_area_km2(client, grid)
    return {
        "candidate_id": seed.candidate_id,
        "region_id": seed.region_id,
        "group": seed.group,
        "wet_area_km2": round(wet, 4),
        "dry_feature_area_km2": round(dry, 4),
        "ftype_area_km2": {str(k): round(v, 4) for k, v in sorted(areas.items(), key=str)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--group", choices=("flight_corridor", "regional_ood", "cross_biome"))
    parser.add_argument("--minimum-area-km2", type=float, default=DEFAULT_MINIMUM_AREA_KM2)
    parser.add_argument("--max-per-group", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()

    config = load_source_acquisition_config(arguments.config)
    seeds = [
        seed
        for seed in load_candidate_seeds(arguments.candidates)
        if seed.target_class == "water"
        and seed.case_type == "positive"
        and (arguments.group is None or seed.group == arguments.group)
    ]

    done: dict[str, dict[str, Any]] = {}
    if arguments.state and arguments.state.is_file():
        done = {row["candidate_id"]: row for row in read_json(arguments.state)["results"]}
    pending = [seed for seed in seeds if seed.candidate_id not in done]
    if arguments.limit is not None:
        pending = pending[: arguments.limit]
    print(f"{len(seeds)} water positives; {len(done)} cached; checking {len(pending)}", flush=True)

    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        futures = {pool.submit(_check, seed, config=config): seed for seed in pending}
        for index, future in enumerate(futures, start=1):
            seed = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # recorded, never silently dropped
                failures.append({"candidate_id": seed.candidate_id, "error": repr(exc)})
                continue
            done[row["candidate_id"]] = row
            if index % 50 == 0:
                print(f"[{index}/{len(pending)}] checked", flush=True)
                if arguments.state:
                    write_json(arguments.state, {"results": sorted(done.values(), key=str)})
    if arguments.state:
        write_json(arguments.state, {"results": sorted(done.values(), key=str)})

    shortlisted = [
        seed
        for seed in seeds
        if seed.candidate_id in done
        and done[seed.candidate_id]["wet_area_km2"] >= arguments.minimum_area_km2
    ]
    capped: list[CandidateSeed] = []
    if arguments.max_per_group:
        kept: Counter[str] = Counter()
        for seed in shortlisted:
            if kept[seed.group] < arguments.max_per_group:
                kept[seed.group] += 1
                capped.append(seed)
    else:
        capped = shortlisted
    write_candidate_manifest(arguments.output, capped)

    by_group = Counter(seed.group for seed in capped)
    write_json(
        arguments.report,
        {
            "schema_version": "4.0.0",
            "authority": "pre-filter only; the registered gates still decide every site",
            "minimum_area_km2": arguments.minimum_area_km2,
            "excluded_ftypes": sorted(DRY_FTYPES),
            "screened_half_extent_m": config.negative_screen_half_extent_m + 30,
            "candidates": len(seeds),
            "checked": len(done),
            "shortlisted": len(shortlisted),
            "written": len(capped),
            "written_by_group": dict(by_group),
            "failures": failures,
        },
    )
    print(
        f"shortlisted {len(shortlisted)}/{len(done)}, wrote {len(capped)} -> {arguments.output} "
        f"({len(failures)} failures)",
        flush=True,
    )
    print("by group:", dict(by_group), flush=True)


if __name__ == "__main__":
    main()
