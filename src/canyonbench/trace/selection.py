"""Gate candidates and freeze an exact independent 120-site cohort."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp  # type: ignore[import-untyped]
from scipy.sparse import lil_matrix  # type: ignore[import-untyped]

from canyonbench.exceptions import DataValidationError
from canyonbench.io import atomic_write_text, write_json
from canyonbench.trace.render import evaluate_site
from canyonbench.trace.schemas import DatasetConfig, GateResult, SiteSpec
from canyonbench.trace.splits import (
    distance_m,
    independence_issues,
    validate_quota,
)


def _bucket(site: SiteSpec) -> str:
    return "negative" if site.case_type == "negative" else "positive"


def _conflict(first: SiteSpec, second: SiteSpec, config: DatasetConfig) -> bool:
    return bool(
        set(first.source_tile_ids).intersection(second.source_tile_ids)
        or set(first.feature_ids).intersection(second.feature_ids)
        or distance_m(first, second) < config.minimum_site_separation_m
    )


def _split_requirements(
    config: DatasetConfig,
) -> dict[tuple[str, str, str, str], int]:
    requirements: dict[tuple[str, str, str, str], int] = {}
    per_group = {row.group: row.per_class // 2 for row in config.quotas}
    for group, total in per_group.items():
        development = round(total * config.split_fractions["development"])
        validation = round(total * config.split_fractions["validation"])
        test = total - development - validation
        for feature in ("water", "road", "field"):
            for presence in ("positive", "negative"):
                requirements[(group, feature, presence, "development")] = development
                requirements[(group, feature, presence, "validation")] = validation
                requirements[(group, feature, presence, "test")] = test
    return requirements


def _solve_assignment(
    candidates: list[SiteSpec],
    config: DatasetConfig,
) -> list[SiteSpec] | None:
    """Jointly select sites and splits under exact quota/conflict constraints."""

    splits = ("development", "validation", "test")
    requirements = _split_requirements(config)
    variable_count = len(candidates) * len(splits)
    conflicts = [
        (first, second)
        for first in range(len(candidates))
        for second in range(first + 1, len(candidates))
        if _conflict(candidates[first], candidates[second], config)
    ]
    constraint_count = len(candidates) + len(requirements) + len(conflicts) * 6
    matrix = lil_matrix((constraint_count, variable_count), dtype=float)
    lower = np.full(constraint_count, -np.inf)
    upper = np.ones(constraint_count)
    row = 0
    for candidate_index in range(len(candidates)):
        for split_index in range(len(splits)):
            matrix[row, candidate_index * len(splits) + split_index] = 1
        row += 1
    for key, required in sorted(requirements.items()):
        group, feature, presence, split = key
        split_index = splits.index(split)
        for candidate_index, candidate in enumerate(candidates):
            if (
                candidate.group == group
                and candidate.target_class == feature
                and _bucket(candidate) == presence
            ):
                matrix[row, candidate_index * len(splits) + split_index] = 1
        lower[row] = required
        upper[row] = required
        row += 1
    for first, second in conflicts:
        for first_split in range(len(splits)):
            for second_split in range(len(splits)):
                if first_split == second_split:
                    continue
                matrix[row, first * len(splits) + first_split] = 1
                matrix[row, second * len(splits) + second_split] = 1
                row += 1
    assert row == constraint_count
    generator = np.random.default_rng(config.seed)
    objective = generator.random(variable_count) * 1e-6
    result = milp(
        objective,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"presolve": True, "time_limit": 300},
    )
    if not result.success or result.x is None:
        return None
    selected: list[SiteSpec] = []
    for candidate_index, candidate in enumerate(candidates):
        for split_index, split in enumerate(splits):
            if result.x[candidate_index * len(splits) + split_index] > 0.5:
                selected.append(candidate.model_copy(update={"split": split}))
    return sorted(selected, key=lambda site: site.site_id)


def select_sites(
    candidates: list[SiteSpec],
    config: DatasetConfig,
    *,
    attempts: int = 1000,
) -> tuple[list[SiteSpec], list[GateResult]]:
    """Run target gates, then repeated deterministic greedy independent selection."""

    passed: list[SiteSpec] = []
    gate_results: list[GateResult] = []
    ordered = sorted(candidates, key=lambda site: site.site_id)
    # Each evaluation opens independent, read-only rasters and spends most of
    # its time in GDAL/OpenCV. Bounded concurrency cuts cohort freeze time
    # without changing the deterministic candidate or MILP ordering.
    with ThreadPoolExecutor(max_workers=4) as executor:
        evaluations = {
            candidate.site_id: executor.submit(evaluate_site, candidate, config)
            for candidate in ordered
        }
    for candidate in ordered:
        results = evaluations[candidate.site_id].result()
        gate_results.extend(results)
        target = next(result for result in results if result.feature == candidate.target_class)
        if target.accepted:
            passed.append(candidate)
    strata: dict[tuple[str, str, str], list[SiteSpec]] = defaultdict(list)
    for site in passed:
        strata[(site.group, site.target_class, _bucket(site))].append(site)
    quota = {row.group: row.per_class // 2 for row in config.quotas}
    required = {
        (group, feature, presence): count
        for group, count in quota.items()
        for feature in ("water", "road", "field")
        for presence in ("positive", "negative")
    }
    shortages = {
        key: (count, len(strata[key]))
        for key, count in required.items()
        if len(strata[key]) < count
    }
    if shortages:
        raise DataValidationError(f"Insufficient gate-passing candidates: {shortages}")

    del attempts
    selected = _solve_assignment(passed, config)
    if selected is not None:
        validate_quota(selected, config)
        issues = independence_issues(
            selected,
            minimum_site_separation_m=config.minimum_site_separation_m,
        )
        if issues:
            raise AssertionError(f"MILP returned an invalid split assignment: {issues[:5]}")
        return selected, gate_results
    raise DataValidationError(
        "No exact 120-site 20/20/60 assignment satisfies all quota, source-ID, "
        "and cross-split footprint constraints; add candidates without changing gates"
    )


def merge_site_manifests(output: Path, sources: list[Path]) -> dict[str, Any]:
    """Combine per-chunk acquisition manifests into one cohort manifest.

    Acquisition is chunked into array tasks so it fits under any wall-clock
    limit, and each task writes its own manifest. A site materialized twice must
    be byte-identical, because acquisition is deterministic given the same seed
    and sources; a disagreement means two chunks wrote different artifacts for
    the same identifier and must not be silently reconciled.
    """

    if not sources:
        raise DataValidationError("at least one chunk manifest is required")
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise DataValidationError(f"missing chunk manifests: {missing}")

    merged: dict[str, dict[str, Any]] = {}
    origin: dict[str, Path] = {}
    for source in sorted(sources):
        value = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        rows = value.get("sites", value) if isinstance(value, dict) else value
        if not isinstance(rows, list):
            raise DataValidationError(f"chunk manifest is not a site list: {source}")
        for row in rows:
            identifier = str(row["site_id"])
            if identifier in merged and merged[identifier] != row:
                raise DataValidationError(
                    f"{identifier} differs between {origin[identifier]} and {source}"
                )
            merged[identifier] = row
            origin[identifier] = source

    ordered = [merged[identifier] for identifier in sorted(merged)]
    atomic_write_text(output, yaml.safe_dump({"sites": ordered}, sort_keys=False))
    return {
        "schema_version": "4.2.0",
        "output": str(output),
        "chunks": [str(path) for path in sorted(sources)],
        "sites": len(ordered),
        "by_chunk": {
            str(path): sum(1 for value in origin.values() if value == path)
            for path in sorted(sources)
        },
    }


def write_selection(
    output: Path,
    selected: list[SiteSpec],
    gate_results: list[GateResult],
) -> None:
    """Write a loadable YAML manifest and complete candidate gate report."""

    value = {"sites": [site.model_dump(mode="json", exclude_none=True) for site in selected]}
    atomic_write_text(output, yaml.safe_dump(value, sort_keys=False))
    write_json(
        output.with_suffix(".gates.json"),
        [result.model_dump(mode="json") for result in gate_results],
    )
