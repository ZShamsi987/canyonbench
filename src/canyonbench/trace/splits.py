"""Site-independent quotas, split assignment, and leakage audits."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import cast

import numpy as np

from canyonbench.exceptions import DataValidationError
from canyonbench.trace.schemas import DatasetConfig, SiteSpec, SplitName


def distance_m(first: SiteSpec, second: SiteSpec) -> float:
    """Great-circle centre distance for conservative footprint separation."""

    radius = 6_371_008.8
    first_latitude, second_latitude = np.radians([first.latitude, second.latitude])
    delta_latitude = second_latitude - first_latitude
    delta_longitude = np.radians(second.longitude - first.longitude)
    value = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(first_latitude) * np.cos(second_latitude) * np.sin(delta_longitude / 2) ** 2
    )
    return float(2 * radius * np.arcsin(np.sqrt(value)))


def assign_splits(sites: list[SiteSpec], *, seed: int = 2026) -> list[SiteSpec]:
    """Stratify 20/20/60 splits by group, class, and case without cutting sites."""

    groups: dict[tuple[str, str, str], list[SiteSpec]] = defaultdict(list)
    for site in sites:
        groups[(site.group, site.target_class, site.case_type)].append(site)
    generator = np.random.default_rng(seed)
    output: list[SiteSpec] = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda site: site.site_id)
        order = generator.permutation(len(rows))
        development = round(len(rows) * 0.2)
        validation = round(len(rows) * 0.2)
        assignments = cast(
            list[SplitName],
            ["development"] * development
            + ["validation"] * validation
            + ["test"] * (len(rows) - development - validation),
        )
        for position, index in enumerate(order):
            output.append(rows[int(index)].model_copy(update={"split": assignments[position]}))
    return sorted(output, key=lambda site: site.site_id)


def validate_quota(sites: list[SiteSpec], config: DatasetConfig) -> None:
    """Validate exact v4 group/class totals and balanced positive/negative sites."""

    if len(sites) != config.site_count:
        raise DataValidationError(f"Expected {config.site_count} sites; found {len(sites)}")
    quota = {row.group: row.per_class for row in config.quotas}
    counts = Counter((site.group, site.target_class) for site in sites)
    for group, per_class in quota.items():
        for feature in ("water", "road", "field"):
            if counts[(group, feature)] != per_class:
                raise DataValidationError(
                    f"Expected {per_class} {group}/{feature} sites; "
                    f"found {counts[(group, feature)]}"
                )
    for key in counts:
        relevant = [site for site in sites if (site.group, site.target_class) == key]
        positives = sum(site.case_type in {"positive", "extinction"} for site in relevant)
        negatives = sum(site.case_type == "negative" for site in relevant)
        if positives != negatives:
            raise DataValidationError(
                f"{key[0]}/{key[1]} is not balanced: {positives} positive vs {negatives} negative"
            )


def leakage_issues(sites: list[SiteSpec]) -> list[str]:
    """Report identifiers shared across splits, including overlapping source assets."""

    issues: list[str] = []
    for field in ("source_tile_ids", "feature_ids"):
        owners: dict[str, set[str]] = defaultdict(set)
        site_owners: dict[str, set[str]] = defaultdict(set)
        for site in sites:
            if site.split is None:
                continue
            for identifier in getattr(site, field):
                owners[identifier].add(site.split)
                site_owners[identifier].add(site.site_id)
        for identifier, splits in owners.items():
            if len(splits) > 1:
                issues.append(
                    f"{field}:{identifier} crosses {sorted(splits)} at "
                    f"{sorted(site_owners[identifier])}"
                )
    coordinates: dict[tuple[float, float], set[str]] = defaultdict(set)
    for site in sites:
        if site.split:
            coordinates[(round(site.latitude, 5), round(site.longitude, 5))].add(site.split)
    for coordinate, splits in coordinates.items():
        if len(splits) > 1:
            issues.append(f"coordinate:{coordinate} crosses {sorted(splits)}")
    return sorted(issues)


def independence_issues(sites: list[SiteSpec], *, minimum_site_separation_m: float) -> list[str]:
    """Reject source sharing or footprint overlap across different splits."""

    issues: list[str] = []
    owners: dict[tuple[str, str], list[SiteSpec]] = defaultdict(list)
    for site in sites:
        for identifier in site.source_tile_ids:
            owners[("source_tile", identifier)].append(site)
        for identifier in site.feature_ids:
            owners[("feature", identifier)].append(site)
    for (kind, identifier), relevant in owners.items():
        splits = {site.split for site in relevant if site.split is not None}
        if len(splits) > 1:
            issues.append(
                f"{kind}:{identifier} crosses {sorted(splits)} at "
                f"{sorted(site.site_id for site in relevant)}"
            )
    ordered = sorted(sites, key=lambda site: site.site_id)
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            if first.split is None or second.split is None or first.split == second.split:
                continue
            separation = distance_m(first, second)
            if separation < minimum_site_separation_m:
                issues.append(
                    f"footprints:{first.site_id}({first.split})/"
                    f"{second.site_id}({second.split}) centres are "
                    f"{separation:.1f} m apart (< {minimum_site_separation_m:.1f} m)"
                )
    return sorted(issues)
