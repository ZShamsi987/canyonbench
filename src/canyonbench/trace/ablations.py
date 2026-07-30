"""Registered gate and causal-instrument ablation utilities."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from canyonbench.io import write_json
from canyonbench.trace.render import evaluate_site
from canyonbench.trace.schemas import DatasetConfig, GateResult, SiteSpec


def gate_ablation_report(
    sites: list[SiteSpec],
    config: DatasetConfig,
) -> dict[str, Any]:
    """Re-run the registered source/date/road-width/detector gate variants."""

    scenarios: dict[str, tuple[list[SiteSpec], DatasetConfig]] = {
        "registered_consensus_strict_road2_detector": (sites, config),
        "single_source": (
            [
                site.model_copy(update={"secondary_mask_paths": site.primary_mask_paths.copy()})
                for site in sites
            ],
            config,
        ),
        "relaxed_dates": (
            sites,
            config.model_copy(
                update={
                    "gates": config.gates.model_copy(
                        update={
                            "maximum_date_gap_days": config.gates.maximum_date_gap_days * 2,
                            "field_maximum_date_gap_days": (
                                config.gates.field_maximum_date_gap_days * 2
                            ),
                        }
                    )
                }
            ),
        ),
        "road_3px": (
            sites,
            config.model_copy(
                update={
                    "gates": config.gates.model_copy(update={"road_minimum_apparent_width_px": 3.0})
                }
            ),
        ),
        "without_exclusion_detector": (
            [site.model_copy(update={"detector_score_paths": {}}) for site in sites],
            config,
        ),
    }
    output: dict[str, Any] = {}
    for scenario, (scenario_sites, scenario_config) in scenarios.items():
        target_results: list[GateResult] = []
        by_stratum: dict[str, list[bool]] = defaultdict(list)
        for site in scenario_sites:
            result = next(
                row
                for row in evaluate_site(site, scenario_config)
                if row.feature == site.target_class
            )
            target_results.append(result)
            by_stratum[f"{site.group}/{site.target_class}/{site.case_type}"].append(result.accepted)
        output[scenario] = {
            "accepted": sum(row.accepted for row in target_results),
            "total": len(target_results),
            "acceptance_rate": (
                sum(row.accepted for row in target_results) / len(target_results)
                if target_results
                else 0
            ),
            "by_stratum": {
                name: {
                    "accepted": sum(values),
                    "total": len(values),
                    "acceptance_rate": sum(values) / len(values),
                }
                for name, values in sorted(by_stratum.items())
            },
            "results": [row.model_dump(mode="json") for row in target_results],
        }
    return output


def write_gate_ablation_report(
    sites: list[SiteSpec],
    config: DatasetConfig,
    output: Path,
) -> None:
    write_json(output, gate_ablation_report(sites, config))
