#!/usr/bin/env python3
"""Append a larger deterministic discovery pool without changing frozen seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from canyonbench.trace.acquisition import discover_candidates
from canyonbench.trace.config import load_candidate_seeds, load_source_acquisition_config
from canyonbench.trace.schemas import CandidateSeed


def _identifier(seed: CandidateSeed) -> int:
    return int(seed.candidate_id.rsplit("_", 1)[1])


def _fingerprint(seed: CandidateSeed) -> tuple[str, str, str, str, float, float]:
    return (
        seed.region_id,
        seed.group,
        seed.target_class,
        seed.case_type,
        round(seed.longitude, 7),
        round(seed.latitude, 7),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("existing", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--multiplier", type=float, default=5.0)
    arguments = parser.parse_args()

    policy = load_source_acquisition_config(arguments.config).model_copy(
        update={"candidate_multiplier": arguments.multiplier}
    )
    existing = load_candidate_seeds(arguments.existing)
    discovered = discover_candidates(policy, arguments.cache_dir)
    known = {_fingerprint(seed) for seed in existing}
    extras = [seed for seed in discovered if _fingerprint(seed) not in known]
    next_identifier = max(_identifier(seed) for seed in existing) + 1
    appended = [
        seed.model_copy(update={"candidate_id": f"candidate_{next_identifier + index:04d}"})
        for index, seed in enumerate(extras)
    ]
    values = [*existing, *appended]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        yaml.safe_dump(
            {
                "schema_version": "4.0.0",
                "candidate_count": len(values),
                "candidates": [seed.model_dump(mode="json") for seed in values],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(
        f"Preserved {len(existing)} frozen candidates and appended {len(appended)} "
        f"new candidates to {arguments.output}"
    )


if __name__ == "__main__":
    main()
