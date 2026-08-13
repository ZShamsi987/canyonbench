#!/usr/bin/env python3
"""Write target-class gate outcomes for every prepared CanyonBench site."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from canyonbench.trace.config import load_project_config, load_sites
from canyonbench.trace.render import evaluate_site


def _evaluate_target(site: object, project: object) -> dict[str, object]:
    """Evaluate one source site; raster I/O and OpenCV release the GIL."""

    results = evaluate_site(site, project.dataset)  # type: ignore[attr-defined]
    target = next(result for result in results if result.feature == site.target_class)  # type: ignore[attr-defined]
    return target.model_dump(mode="json") | {  # type: ignore[no-any-return]
        "group": site.group,  # type: ignore[attr-defined]
        "target_class": site.target_class,  # type: ignore[attr-defined]
    }


def main() -> None:
    root = Path("/scratch/network/pu9340/canyonbench-trace-data")
    project = load_project_config("configs/trace.yaml")
    sites = load_sites(root / "manifests/trace_prepared_candidates.yaml")
    rows: list[dict[str, object]] = []
    ordered = sorted(sites, key=lambda item: item.site_id)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_evaluate_target, site, project): site.site_id for site in ordered
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            reasons = ",".join(str(reason) for reason in row["reasons"]) or "PASS"
            print(
                f"[{index}/{len(ordered)}] {row['site_id']} {row['accepted']} {reasons}", flush=True
            )

    summary: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = (str(row["group"]), str(row["target_class"]), str(row["case_type"]))
        summary[key]["total"] += 1
        summary[key]["accepted"] += int(bool(row["accepted"]))
        for reason in row["reasons"]:  # type: ignore[index]
            summary[key][str(reason)] += 1
    output = {
        "rows": rows,
        "summary": {"/".join(key): dict(value) for key, value in summary.items()},
    }
    path = root / "reports/gate-diagnostic.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"WROTE {path}", flush=True)


if __name__ == "__main__":
    main()
