#!/bin/bash
# Close the field-negative quota gap in one resumable pass.
#
# THE PROBLEM THIS SOLVES.  The registered field-negative rule admits a site
# only when no cultivated pixel falls inside the buffered maximum camera
# footprint.  Measured against the frozen pool, that rule accepts roughly one
# candidate in ten, so the quota is reached by searching a large pool -- not by
# relaxing the rule.  The authoritative screen cannot run that search: it costs
# one CropScape request or one national-archive window per candidate, CropScape
# has been answering 503, and reading whole archives is what the login node
# kills for memory.  Screening a few dozen candidates a day against a need for
# several hundred is why this stage stalled.
#
# THE ORDER HERE.  Discovery enlarges the pool.  The Planetary Computer
# pre-filter then discards roughly nine candidates in ten at a windowed range
# read each, so the authoritative screen -- unchanged, still the only thing that
# can vet a seed -- runs over dozens of candidates instead of thousands.  Only
# vetted seeds reach acquisition.
#
# Every stage is resumable and writes atomically; stop and rerun at any point.
#
#   bash scripts/adroit/recover_field_negatives.sh
#   MULTIPLIER=20 WORKERS=3 bash scripts/adroit/recover_field_negatives.sh
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../slurm/_common.sh
source "$SCRIPT_DIR/../../slurm/_common.sh"

export CANYONBENCH_CDL_CACHE_DIR="${CANYONBENCH_CDL_CACHE_DIR:-$CANYONBENCH_DATA/cache/cdl}"
# One download at a time in the authoritative stages; this must never look like
# a parallel scraper.  The pre-filter's own concurrency is set by --workers.
export GDAL_NUM_THREADS=1

SOURCES_CONFIG="${SOURCES_CONFIG:-$CANYONBENCH_HOME/configs/trace_sources.yaml}"
MULTIPLIER="${MULTIPLIER:-20}"
WORKERS="${WORKERS:-3}"
PREFILTER_WORKERS="${PREFILTER_WORKERS:-8}"
# Field-negative quotas are 20 flight, 12 regional, 8 cross-biome. The
# authoritative screen costs about 22 s per candidate, so shortlisting far past
# the quota only buys screen time; three times the largest quota leaves margin
# for authoritative rejections and for seeds the 22 km rule cannot place.
MAX_PER_GROUP="${MAX_PER_GROUP:-60}"

MANIFESTS="$CANYONBENCH_DATA/manifests"
REPORTS="$CANYONBENCH_DATA/reports"
LOGS="$CANYONBENCH_DATA/logs"
CACHE="$CANYONBENCH_DATA/cache"
CANDIDATES="$MANIFESTS/trace_candidates.yaml"
EXPANDED="$MANIFESTS/trace_candidates_expanded.yaml"
SHORTLIST="$MANIFESTS/field_negative_shortlist.yaml"
VETTED="$MANIFESTS/field_negative_flight_vetted.yaml"
mkdir -p "$MANIFESTS" "$REPORTS" "$LOGS" "$CACHE/site-discovery"

cd "$CANYONBENCH_HOME"
run() { nice -n 19 "$CANYONBENCH_HOME/.venv/bin/python" "$@"; }

echo "== 1/4 expanding the deterministic candidate pool (${MULTIPLIER}x) =="
if [ -f "$EXPANDED" ]; then
  echo "   reusing $EXPANDED"
else
  run scripts/adroit/extend_candidates.py \
    "$SOURCES_CONFIG" "$CANDIDATES" "$EXPANDED" \
    --cache-dir "$CACHE/site-discovery" \
    --multiplier "$MULTIPLIER"
fi

echo "== 2/4 pre-filtering field negatives against CDL Cultivated Layer COGs =="
# Cheap, resumable, and never authoritative: the state file means a rerun costs
# no requests, and the shortlist only decides who is worth an authoritative check.
run scripts/adroit/prefilter_field_negatives.py \
  "$SOURCES_CONFIG" "$EXPANDED" "$SHORTLIST" \
  "$REPORTS/field-negative-prefilter.json" \
  --state "$CACHE/field-negative-prefilter.json" \
  --workers "$PREFILTER_WORKERS" \
  --max-per-group "$MAX_PER_GROUP"

echo "== 3/4 authoritative CDL + Annual NLCD screen of the shortlist =="
# Restarting every eight candidates clears GIS process state, which is what the
# login-node memory enforcement killed the previous single long run for.
shortlisted="$(grep -Ec '^[[:space:]]*-[[:space:]]+candidate_id:' "$SHORTLIST" || true)"
echo "   $shortlisted shortlisted candidates to vet"
: > "$LOGS/field-negative-screen.log"
for ((start = 0; start < shortlisted; start += 8)); do
  run scripts/adroit/screen_field_negatives.py \
    "$SOURCES_CONFIG" "$SHORTLIST" \
    "$MANIFESTS/field_negative_vetted-slice-${start}.yaml" \
    "$REPORTS/field-negative-screen-slice-${start}.json" \
    --cache-dir "$CACHE/field-negative-screen" \
    --source-root "$CANYONBENCH_DATA/sources" \
    --start "$start" --limit 8 2>&1 | tee -a "$LOGS/field-negative-screen.log"
done

run - "$MANIFESTS" "$VETTED" <<'PY'
"""Merge the completed slices into one vetted manifest, dropping duplicates."""
import sys
from pathlib import Path

import yaml

manifests, output = Path(sys.argv[1]), Path(sys.argv[2])
seen, merged = set(), []
for path in sorted(manifests.glob("field_negative_vetted-slice-*.yaml")):
    for seed in (yaml.safe_load(path.read_text()) or {}).get("candidates", []) or []:
        if seed["candidate_id"] not in seen:
            seen.add(seed["candidate_id"])
            merged.append(seed)
output.write_text(
    yaml.safe_dump(
        {"schema_version": "4.0.0", "candidate_count": len(merged), "candidates": merged},
        sort_keys=False,
    ),
    encoding="utf-8",
)
print(f"vetted field negatives: {len(merged)} -> {output}")
PY

echo "== 4/4 acquiring every vetted field negative =="
VETTED_CANDIDATES="$VETTED" WORKERS="$WORKERS" \
  bash "$SCRIPT_DIR/acquire_vetted_parallel.sh"

completed="$(find "$CANYONBENCH_DATA/sources" -mindepth 2 -maxdepth 2 -name COMPLETE | wc -l)"
echo "total complete source bundles: $completed"
echo "Next: sbatch slurm/adroit_freeze.sbatch"
