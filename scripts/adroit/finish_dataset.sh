#!/bin/bash
# Carry the dataset from the end of acquisition to the audit packet, unattended.
#
# WHY THIS EXISTS.  Freeze, build, and instruments are three Slurm jobs with a
# strict dependency, and each one finishing at an unattended hour has repeatedly
# cost this project half a day of wall clock waiting for a human to submit the
# next one. Nothing here makes a decision: every stage still stops on its own
# failure, and the registered human audit gate is still a hard stop.
#
# The one judgement encoded here is refusing to build on an unvalidated freeze.
# A build reads sites.yaml and writes 960 clean views plus 240 degraded ones; if
# selection produced no manifest, submitting the build wastes eight hours of
# allocation and produces nothing.
#
#   nohup setsid bash scripts/adroit/finish_dataset.sh > "$CANYONBENCH_DATA/logs/finish.log" 2>&1 &
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../slurm/_common.sh
source "$SCRIPT_DIR/../../slurm/_common.sh"

cd "$CANYONBENCH_HOME"
SITES="$CANYONBENCH_DATA/manifests/sites.yaml"
DATASET_DIR="${CANYONBENCH_DATASET_DIR:-$CANYONBENCH_DATA/generated}"
VALIDATION="$CANYONBENCH_DATA/reports/dataset-validation.json"

say() { echo "[$(date -Is)] $*"; }

wait_for_acquisition() {
  local waited=0
  while pgrep -f "trace acquire-sources" > /dev/null 2>&1; do
    sleep 120
    waited=$((waited + 2))
    if (( waited % 20 == 0 )); then
      say "still acquiring after ${waited}m: $(find "$CANYONBENCH_DATA/sources" \
        -mindepth 2 -maxdepth 2 -name COMPLETE | wc -l) bundles"
    fi
    # Acquisition is bounded by its own per-candidate timeout; this is only a
    # guard against a wedged process holding the chain open indefinitely.
    if (( waited > 480 )); then
      say "acquisition still running after 8h; continuing anyway"
      return 0
    fi
  done
}

wait_for_job() {
  local job_id="$1" name="$2" state
  say "submitted $name as $job_id"
  while squeue -h -j "$job_id" 2>/dev/null | grep -q .; do sleep 60; done
  for _ in $(seq 1 10); do
    state="$(sacct -X -n -P -j "$job_id" --format=State | awk -F'|' 'NF {print $1; exit}')"
    case "$state" in
      COMPLETED) say "$name COMPLETED"; return 0 ;;
      FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)
        say "$name ended $state; see logs/${name}-${job_id}.err"; return 1 ;;
      *) sleep 15 ;;
    esac
  done
  say "no terminal accounting state for $job_id"; return 1
}

say "waiting for acquisition to drain"
wait_for_acquisition
say "$(find "$CANYONBENCH_DATA/sources" -mindepth 2 -maxdepth 2 -name COMPLETE | wc -l) bundles acquired"

say "stage 1/3 freeze"
if ! wait_for_job "$(sbatch --parsable slurm/adroit_freeze.sbatch)" cb-freeze; then
  say "FREEZE FAILED - the shortage list is in the job's stderr. Stopping."
  exit 1
fi
if [ ! -s "$SITES" ]; then
  say "freeze reported success but wrote no $SITES. Stopping rather than building on nothing."
  exit 1
fi
say "frozen: $(grep -Ec '^[[:space:]]*-[[:space:]]+site_id:' "$SITES") sites"

say "stage 2/3 build"
if ! wait_for_job "$(sbatch --parsable slurm/adroit_build.sbatch)" cb-build; then
  say "BUILD FAILED. Stopping."
  exit 1
fi
if [ -f "$VALIDATION" ]; then
  say "dataset-validation passed: $(grep -o '"passed"[^,}]*' "$VALIDATION" | head -1)"
fi

say "stage 3/3 instruments and audit packet"
# V1 inserts a synthetic feature into an otherwise NEGATIVE scene: inserting a
# road into a view that already contains one measures nothing. Pick the view
# from a frozen negative site rather than whichever rgb.png sorts first.
NEGATIVE_VIEW="$("$CANYONBENCH_HOME/.venv/bin/python" - "$SITES" "$DATASET_DIR" <<'PYEOF'
import sys
from pathlib import Path

import yaml

sites_path, dataset_dir = Path(sys.argv[1]), Path(sys.argv[2])
document = yaml.safe_load(sites_path.read_text()) or {}
sites = document.get("sites") or document.get("candidates") or []
for site in sites:
    if site.get("case_type") != "negative":
        continue
    directory = dataset_dir / site["site_id"]
    # Prefer the lowest altitude, where an inserted feature is most resolvable.
    for view in sorted(directory.glob("view_a*nadir")) or sorted(directory.glob("view_*")):
        candidate = view / "rgb.png"
        if candidate.is_file():
            print(candidate)
            raise SystemExit(0)
raise SystemExit("no negative-site view found")
PYEOF
)"
if [ -z "$NEGATIVE_VIEW" ]; then
  say "no negative-site rendered view found under $DATASET_DIR; cannot run V1."
  exit 1
fi
say "V1 negative control view: $NEGATIVE_VIEW"
if ! wait_for_job "$(sbatch --parsable slurm/adroit_instruments.sbatch "$NEGATIVE_VIEW")" cb-instruments; then
  say "INSTRUMENTS FAILED. Stopping."
  exit 1
fi

say "DATASET READY. Audit packet: $CANYONBENCH_DATA/audits/audit.csv"
say "Send that CSV, its audit.csv_assets/ folder, and docs/AUDITOR-GUIDE.md to both auditors."
say "Nothing further runs until both audit CSVs come back - that gate is registered."
