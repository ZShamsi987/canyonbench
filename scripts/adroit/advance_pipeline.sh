#!/bin/bash
# Advance the Adroit-only portion of CanyonBench without fragile process-name
# polling.  Run this detached on the login node: acquisition needs its network
# access; the CPU-heavy gates, build, and instruments run through Slurm.
#
# The first pass uses the already frozen discovery pool.  If selection cannot
# produce the registered cohort, no cohort has been frozen, so it is safe to
# append the deterministic 5x discovery pool and try once more.  The script
# never starts paid model inference and stops at the required human audit gate.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../slurm/_common.sh
source "$SCRIPT_DIR/../../slurm/_common.sh"

export CANYONBENCH_CDL_CACHE_DIR="${CANYONBENCH_CDL_CACHE_DIR:-$CANYONBENCH_DATA/cache/cdl}"
# This process is intentionally a single niced, I/O-bound client on the login
# node.  The Slurm jobs below set their own CPU threading from their allocation.
export GDAL_NUM_THREADS=1

SOURCES_CONFIG="${SOURCES_CONFIG:-configs/trace_sources.yaml}"
CANDIDATES="$CANYONBENCH_DATA/manifests/trace_candidates.yaml"
PREPARED="$CANYONBENCH_DATA/manifests/trace_prepared_candidates.yaml"
REPORTS="$CANYONBENCH_DATA/reports"
LOGS="$CANYONBENCH_DATA/logs"
# Public WCS/COG services occasionally leave one request open indefinitely.
# Isolate each candidate in a child process: a timed-out tile is recorded in
# the log and omitted from the prepared manifest, while every later candidate
# continues.  This is intentionally sequential; it is not a parallel scraper.
ACQUIRE_TIMEOUT_SECONDS="${ACQUIRE_TIMEOUT_SECONDS:-900}"
ACQUIRE_START="${CANYONBENCH_ACQUIRE_START:-0}"
mkdir -p "$REPORTS" "$LOGS" "$CANYONBENCH_DATA/cache/site-discovery"

wait_for_job() {
  local job_id="${1%%;*}" state
  echo "Waiting for Slurm job $job_id"
  while squeue -h -j "$job_id" | grep -q .; do
    sleep 60
  done
  # Accounting may lag queue departure very briefly.
  for _ in $(seq 1 10); do
    state="$(sacct -X -n -P -j "$job_id" --format=State | awk -F'|' 'NF {print $1; exit}')"
    case "$state" in
      COMPLETED) return 0 ;;
      FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)
        echo "Slurm job $job_id ended with $state" >&2
        return 1
        ;;
      *) sleep 15 ;;
    esac
  done
  echo "Could not obtain a terminal accounting state for Slurm job $job_id" >&2
  return 1
}

submit_gate_diagnostic() {
  sbatch --parsable \
    --job-name=cb-gate-diagnostic \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --time=06:00:00 --nice=5000 \
    --output="$CANYONBENCH_HOME/logs/%x-%j.out" \
    --error="$CANYONBENCH_HOME/logs/%x-%j.err" \
    --wrap="cd '$CANYONBENCH_HOME' && '$CANYONBENCH_HOME/.venv/bin/python' scripts/adroit/gate_diagnostic.py"
}

acquire_pool() {
  local pass="$1" start="${2:-$ACQUIRE_START}" count index report
  count="$("$CANYONBENCH_HOME/.venv/bin/python" - "$CANDIDATES" <<'PY'
from pathlib import Path
import sys
import yaml
print(len(yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))["candidates"]))
PY
)"
  echo "== acquisition pass $pass: candidates $start through $((count - 1)), $(date -Is) =="
  for ((index = start; index < count; index += 1)); do
    report="$REPORTS/source-acquisition-supervised-pass-${pass}-index-${index}.json"
    echo "== acquisition pass $pass, candidate index $index/$((count - 1)) =="
    if ! timeout --kill-after=60s "${ACQUIRE_TIMEOUT_SECONDS}s" \
      nice -n 19 uv run --frozen canyonbench trace acquire-sources \
        "$SOURCES_CONFIG" "$CANDIDATES" "$CANYONBENCH_DATA/sources" "$PREPARED" \
        "$report" --start "$index" --limit 1; then
      echo "Candidate index $index exceeded ${ACQUIRE_TIMEOUT_SECONDS}s or exited unexpectedly; continuing."
    fi
  done
}

refresh_prepared_manifest() {
  "$CANYONBENCH_HOME/.venv/bin/python" - "$SOURCES_CONFIG" "$CANYONBENCH_DATA" <<'PY'
from pathlib import Path
import sys

from canyonbench.trace.acquisition import acquire_candidates
from canyonbench.trace.config import load_source_acquisition_config

config = Path(sys.argv[1])
root = Path(sys.argv[2])
sites = acquire_candidates(
    [], load_source_acquisition_config(config), root / "sources",
    root / "manifests/trace_prepared_candidates.yaml",
    root / "reports/source-acquisition-manifest-refresh.json",
)
print(f"Refreshed prepared manifest with {len(sites)} invocation sites")
PY
}

run_cohort() {
  local gate_job freeze_job
  refresh_prepared_manifest
  gate_job="$(submit_gate_diagnostic)"
  wait_for_job "$gate_job" || return 1
  freeze_job="$(sbatch --parsable "$CANYONBENCH_HOME/slurm/adroit_freeze.sbatch")"
  wait_for_job "$freeze_job"
}

acquire_pool 1
if ! run_cohort; then
  echo "== first cohort did not freeze; appending deterministic expansion =="
  original_count="$("$CANYONBENCH_HOME/.venv/bin/python" - "$CANDIDATES" <<'PY'
from pathlib import Path
import sys
import yaml
print(len(yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))["candidates"]))
PY
)"
  expanded="$CANYONBENCH_DATA/manifests/trace_candidates.expanded.yaml"
  "$CANYONBENCH_HOME/.venv/bin/python" scripts/adroit/extend_candidates.py \
    "$SOURCES_CONFIG" "$CANDIDATES" "$expanded" \
    --cache-dir "$CANYONBENCH_DATA/cache/site-discovery" --multiplier 5.0
  mv "$expanded" "$CANDIDATES"
  # The old pool has already been attempted.  Only materialize the appended
  # deterministic candidates, including their corrected field-negative screen.
  acquire_pool 2 "$original_count"
  run_cohort
fi

echo "== frozen cohort ready; submitting generation =="
build_job="$(sbatch --parsable "$CANYONBENCH_HOME/slurm/adroit_build.sbatch")"
wait_for_job "$build_job"

"$CANYONBENCH_HOME/.venv/bin/python" - "$REPORTS/dataset-validation.json" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("passed") is not True:
    raise SystemExit(f"Dataset validation did not pass: {report}")
PY

# The selected cohort always contains negatives; select one from the frozen
# manifest rather than assuming a particular site identifier survives selection.
negative_site="$("$CANYONBENCH_HOME/.venv/bin/python" - "$CANYONBENCH_DATA/manifests/sites.yaml" <<'PY'
from pathlib import Path
import sys
from canyonbench.trace.config import load_sites

for site in load_sites(Path(sys.argv[1])):
    if site.case_type == "negative":
        print(site.site_id)
        break
else:
    raise SystemExit("Frozen cohort contains no negative site for V1")
PY
)"
negative_view="$CANYONBENCH_DATA/generated/$negative_site/view_a3km_nadir/rgb.png"
test -f "$negative_view"
instruments_job="$(sbatch --parsable "$CANYONBENCH_HOME/slurm/adroit_instruments.sbatch" "$negative_view")"
wait_for_job "$instruments_job"

echo "STAGE 1 COMPLETE: audit packet is ready at $CANYONBENCH_DATA/audits/audit.csv"
echo "The remaining gate is the two independent human audits; no model calls or API spend were started."
