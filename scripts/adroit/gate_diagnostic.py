#!/usr/bin/env python3
"""Write target-class gate outcomes for every prepared CanyonBench site."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from canyonbench.trace.config import load_project_config, load_sites
from canyonbench.trace.render import evaluate_site


def main() -> None:
    root = Path("/scratch/network/pu9340/canyonbench-trace-data")
    project = load_project_config("configs/trace.yaml")
    sites = load_sites(root / "manifests/trace_prepared_candidates.yaml")
    rows: list[dict[str, object]] = []
    for index, site in enumerate(sorted(sites, key=lambda item: item.site_id), start=1):
        target = next(
            result
            for result in evaluate_site(site, project.dataset)
            if result.feature == site.target_class
        )
        row = target.model_dump(mode="json") | {
            "group": site.group,
            "target_class": site.target_class,
        }
        rows.append(row)
        reasons = ",".join(target.reasons) or "PASS"
        print(f"[{index}/{len(sites)}] {site.site_id} {target.accepted} {reasons}", flush=True)

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
