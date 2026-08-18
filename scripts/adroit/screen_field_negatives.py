#!/usr/bin/env python3
"""Pre-screen field-negative seeds against their exact CDL and NLCD support."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import rasterio  # type: ignore[import-untyped]
from rasterio.warp import transform_bounds  # type: ignore[import-untyped]
from rasterio.windows import Window, from_bounds  # type: ignore[import-untyped]

from canyonbench.io import write_json
from canyonbench.trace.acquisition import (
    CULTIVATED_CDL_CODES,
    _fetch_cdl,
    _fetch_nlcd,
    _Grid,
    _http_client,
    _naip_items,
    write_candidate_manifest,
)
from canyonbench.trace.config import load_candidate_seeds, load_source_acquisition_config
from canyonbench.trace.schemas import CandidateSeed


def _contains(
    path: str | Path,
    values: set[int],
    *,
    bounds_wgs84: tuple[float, float, float, float] | None = None,
) -> bool:
    """Check a small geographic window without reading a national CDL archive."""

    with rasterio.open(path) as dataset:
        window: Window | None = None
        if bounds_wgs84 is not None:
            if dataset.crs is None:
                raise RuntimeError(f"Raster has no CRS: {path}")
            bounds = transform_bounds("EPSG:4326", dataset.crs, *bounds_wgs84, densify_pts=21)
            window = from_bounds(*bounds, transform=dataset.transform)
            window = window.round_offsets().round_lengths().intersection(
                Window(0, 0, dataset.width, dataset.height)
            )
        return bool(np.isin(dataset.read(1, window=window), list(values)).any())


def _screen(
    seed: CandidateSeed,
    *,
    cache_dir: Path,
    config: object,
) -> tuple[bool, int]:
    """Return whether both field authorities are clear over the source support."""

    # Type annotations on the generated Pydantic schema deliberately remain
    # local to the acquisition library.  The script only needs these frozen
    # configuration attributes.
    grid = _Grid(
        seed.longitude,
        seed.latitude,
        config.working_resolution_m,  # type: ignore[attr-defined]
        config.source_half_extent_m + 30,  # type: ignore[attr-defined]
    )
    directory = cache_dir / seed.candidate_id
    directory.mkdir(parents=True, exist_ok=True)
    with _http_client() as client:
        year, _ = _naip_items(client, grid, config.preferred_naip_years)  # type: ignore[attr-defined]
        cdl = _fetch_cdl(
            client,
            grid.wgs84_bounds,
            year,
            directory / f"cdl_{year}.tif",
            use_cached_archive=bool(os.environ.get("CANYONBENCH_CDL_CACHE_DIR")),
        )
        nlcd = _fetch_nlcd(
            client,
            "Land-Cover-Native",
            grid.wgs84_bounds,
            year,
            directory / f"annual_nlcd_landcover_{year}.tif",
        )
    clear = not _contains(
        cdl,
        set(CULTIVATED_CDL_CODES),
        bounds_wgs84=grid.wgs84_bounds,
    ) and not _contains(nlcd, {82})
    return clear, year


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--group", choices=("flight_corridor", "regional_ood", "cross_biome"))
    parser.add_argument("--limit", type=int, default=None)
    arguments = parser.parse_args()

    config = load_source_acquisition_config(arguments.config)
    candidates = [
        seed
        for seed in load_candidate_seeds(arguments.candidates)
        if seed.target_class == "field"
        and seed.case_type == "negative"
        and (arguments.group is None or seed.group == arguments.group)
        and not (
            arguments.source_root / seed.candidate_id.replace("candidate", "site") / "COMPLETE"
        ).is_file()
    ]
    if arguments.limit is not None:
        candidates = candidates[: arguments.limit]

    accepted: list[CandidateSeed] = []
    rejected: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for index, seed in enumerate(candidates, start=1):
        try:
            clear, year = _screen(seed, cache_dir=arguments.cache_dir, config=config)
            if clear:
                accepted.append(seed.model_copy(update={"discovery_source": "cdl+annual-nlcd"}))
                print(
                    f"[{index}/{len(candidates)}] accepted {seed.candidate_id} ({year})",
                    flush=True,
                )
            else:
                rejected.append({"candidate_id": seed.candidate_id, "naip_year": year})
                print(
                    f"[{index}/{len(candidates)}] rejected {seed.candidate_id} ({year})",
                    flush=True,
                )
        except Exception as exc:
            failures.append(
                {
                    "candidate_id": seed.candidate_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(candidates)}] failed {seed.candidate_id}: {exc}", flush=True)
        write_candidate_manifest(arguments.output, accepted)
        write_json(
            arguments.report,
            {
                "requested": len(candidates),
                "processed": index,
                "accepted": len(accepted),
                "rejected": rejected,
                "failures": failures,
            },
        )

    print(f"Wrote {len(accepted)} field-negative seeds to {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
