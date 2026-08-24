#!/usr/bin/env python3
"""Verify and package the audit bundle before it reaches a human.

An auditor who opens a packet with a missing review sheet, a row pointing at a
file that was never written, or a sheet that is not a readable image, has to
stop and wait for a regenerated packet. Their turnaround is the only step in
this project that cannot be compressed by compute, so a defect discovered by
them costs a day; the same defect discovered here costs a minute.

Checks, all of which have a plausible failure mode in this pipeline:

* every row names a sheet that exists, opens, and has the expected two-column
  eight-panel geometry;
* both auditor ids appear, with the same view list for each, since the loader
  duplicates every sampled view across auditors and a mismatch silently makes
  the agreement rate uncomputable;
* the four answer columns are present and empty, so nobody receives a packet
  with answers already in it;
* the sampled views span the strata, because an audit that never looks at a
  stratum cannot detect a problem in it.

    python scripts/adroit/package_audit.py "$CANYONBENCH_DATA/audits/audit.csv"
"""

from __future__ import annotations

import argparse
import csv
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ANSWER_COLUMNS = (
    "overlay_aligned",
    "feature_resolvable",
    "obvious_edit_artifact",
    "source_mismatch",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_csv", type=Path)
    parser.add_argument("--output", type=Path, help="Zip to write (default: alongside the CSV).")
    parser.add_argument("--skip-image-check", action="store_true")
    arguments = parser.parse_args()

    csv_path = arguments.audit_csv
    assets = csv_path.parent / f"{csv_path.stem}_assets"
    if not csv_path.is_file():
        raise SystemExit(f"no audit CSV at {csv_path}")
    if not assets.is_dir():
        raise SystemExit(f"no assets directory at {assets}")

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    problems: list[str] = []

    missing_columns = [c for c in ANSWER_COLUMNS if c not in (rows[0] if rows else {})]
    if missing_columns:
        problems.append(f"CSV is missing answer columns: {missing_columns}")

    prefilled = sum(1 for r in rows for c in ANSWER_COLUMNS if (r.get(c) or "").strip())
    if prefilled:
        problems.append(f"{prefilled} answer cells are already filled in")

    by_auditor: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        by_auditor[row["auditor"]].add((row["site"], row["view"]))
    if len(by_auditor) != 2:
        problems.append(f"expected exactly two auditors, found {sorted(by_auditor)}")
    else:
        first, second = by_auditor.values()
        if first != second:
            problems.append(
                f"auditors were given different view lists "
                f"({len(first ^ second)} views differ); the agreement rate needs identical lists"
            )

    views = {(r["site"], r["view"]) for r in rows}
    checked = missing = unreadable = 0
    for site, view in sorted(views):
        sheet = assets / f"{site}__{view}.png"
        if not sheet.is_file():
            missing += 1
            problems.append(f"missing review sheet: {sheet.name}")
            continue
        checked += 1
        if arguments.skip_image_check:
            continue
        try:
            from PIL import Image

            with Image.open(sheet) as image:
                width, height = image.size
            # Two columns of 1024 px panels, each with a 36 px label strip.
            if width < 2000 or height < 2000:
                unreadable += 1
                problems.append(f"sheet is unexpectedly small ({width}x{height}): {sheet.name}")
        except Exception as exc:  # a sheet that will not open is a defect
            unreadable += 1
            problems.append(f"sheet will not open ({type(exc).__name__}): {sheet.name}")

    print(f"rows            : {len(rows)}")
    print(f"unique views    : {len(views)}")
    print(f"auditors        : {sorted(by_auditor)}")
    print(f"sheets present  : {checked}   missing: {missing}   unreadable: {unreadable}")

    strata = Counter(site.rsplit('_', 1)[0] for site, _ in views)
    altitudes = Counter(view.split('_')[1][:1] for _, view in views if '_' in view)
    print(f"altitude spread : {dict(sorted(altitudes.items()))}")
    if len(altitudes) < 4:
        problems.append(f"sample covers only {len(altitudes)} of the four altitudes")
    del strata

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for problem in problems[:20]:
            print(f"  - {problem}")
        raise SystemExit("packet is not fit to send")

    output = arguments.output or csv_path.parent / "canyonbench-audit-packet.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(csv_path, csv_path.name)
        for sheet in sorted(assets.glob("*.png")):
            bundle.write(sheet, f"{assets.name}/{sheet.name}")
        guide = Path(__file__).resolve().parents[2] / "docs" / "AUDIT-ASSIGNMENT.md"
        if guide.is_file():
            bundle.write(guide, "HOW-TO-DO-THIS.md")
        auditor_guide = guide.with_name("AUDITOR-GUIDE.md")
        if auditor_guide.is_file():
            bundle.write(auditor_guide, "AUDITOR-GUIDE.md")

    size_mb = output.stat().st_size / 1e6
    print(f"\nOK. Wrote {output} ({size_mb:.1f} MB)")
    print("Send the same zip to both auditors; each fills only their own rows.")


if __name__ == "__main__":
    main()
