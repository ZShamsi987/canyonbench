#!/usr/bin/env bash
# Acquire a small, already-vetted candidate set with bounded parallelism.
#
# This is deliberately separate from advance_pipeline.sh: it is for repairing a
# known cohort without re-entering the full discovery/acquisition loop.  Each
# worker writes its own report and prepared-manifest file, so workers never race
# on output files; after all workers stop, a zero-candidate refresh produces the
# canonical prepared manifest from completed source directories.
set -euo pipefail

CANYONBENCH_HOME="${CANYONBENCH_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CANYONBENCH_DATA="${CANYONBENCH_DATA:-/scratch/network/$USER/canyonbench-trace-data}"
SOURCES_CONFIG="${SOURCES_CONFIG:-$CANYONBENCH_HOME/configs/trace_sources.yaml}"
VETTED_CANDIDATES="${VETTED_CANDIDATES:-$CANYONBENCH_DATA/manifests/field_negative_flight_vetted.yaml}"
WORKERS="${WORKERS:-3}"
ACQUIRE_TIMEOUT_SECONDS="${ACQUIRE_TIMEOUT_SECONDS:-3600}"
RUN_LABEL="${RUN_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)}"

COMMAND="$CANYONBENCH_HOME/.venv/bin/canyonbench"
SOURCE_ROOT="$CANYONBENCH_DATA/sources"
MANIFEST_DIR="$CANYONBENCH_DATA/manifests"
REPORT_DIR="$CANYONBENCH_DATA/reports"
LOG_DIR="$CANYONBENCH_DATA/logs"
CANONICAL_PREPARED="$MANIFEST_DIR/trace_prepared_candidates.yaml"

if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS must be a positive integer, got: $WORKERS" >&2
  exit 2
fi
if ! [[ "$ACQUIRE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ACQUIRE_TIMEOUT_SECONDS must be a positive integer, got: $ACQUIRE_TIMEOUT_SECONDS" >&2
  exit 2
fi
for path in "$COMMAND" "$SOURCES_CONFIG" "$VETTED_CANDIDATES"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 2
  fi
done

mkdir -p "$SOURCE_ROOT" "$MANIFEST_DIR" "$REPORT_DIR" "$LOG_DIR"
candidate_count="$(grep -Ec '^[[:space:]]*-[[:space:]]+candidate_id:' "$VETTED_CANDIDATES" || true)"
if (( candidate_count == 0 )); then
  echo "No vetted candidates in $VETTED_CANDIDATES" >&2
  exit 2
fi

echo "[$(date -Is)] starting $candidate_count vetted candidates with $WORKERS workers"
echo "candidate manifest: $VETTED_CANDIDATES"

declare -a worker_pids=()
failures=0

reap_one() {
  local pid
  pid="${worker_pids[0]}"
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
  worker_pids=("${worker_pids[@]:1}")
}

for ((index = 0; index < candidate_count; index++)); do
  while (( ${#worker_pids[@]} >= WORKERS )); do
    reap_one
  done
  report="$REPORT_DIR/source-acquisition-vetted-${RUN_LABEL}-index-${index}.json"
  prepared="$MANIFEST_DIR/trace_prepared_vetted-${RUN_LABEL}-index-${index}.yaml"
  log="$LOG_DIR/acquire-vetted-${RUN_LABEL}-index-${index}.log"
  (
    set -o pipefail
    echo "[$(date -Is)] start index=$index"
    timeout --signal=KILL "${ACQUIRE_TIMEOUT_SECONDS}s" nice -n 19 "$COMMAND" trace acquire-sources \
      "$SOURCES_CONFIG" "$VETTED_CANDIDATES" "$SOURCE_ROOT" "$prepared" "$report" \
      --start "$index" --limit 1
    echo "[$(date -Is)] complete index=$index"
  ) >"$log" 2>&1 &
  worker_pids+=("$!")
  echo "[$(date -Is)] launched index=$index pid=${worker_pids[-1]} log=$log"
done

while (( ${#worker_pids[@]} > 0 )); do
  reap_one
done

# The empty slice makes acquire-sources perform no network work.  It only scans
# complete source directories and atomically rebuilds the shared manifest.
refresh_report="$REPORT_DIR/source-acquisition-vetted-${RUN_LABEL}-refresh.json"
"$COMMAND" trace acquire-sources \
  "$SOURCES_CONFIG" "$VETTED_CANDIDATES" "$SOURCE_ROOT" "$CANONICAL_PREPARED" "$refresh_report" \
  --start "$candidate_count" --limit 1

completed="$(find "$SOURCE_ROOT" -mindepth 2 -maxdepth 2 -name COMPLETE -print | wc -l)"
echo "[$(date -Is)] refresh complete; total completed source bundles: $completed"
if (( failures > 0 )); then
  echo "[$(date -Is)] $failures worker(s) exited unsuccessfully; inspect $LOG_DIR/acquire-vetted-${RUN_LABEL}-index-*.log" >&2
  exit 1
fi
echo "[$(date -Is)] vetted acquisition batch complete"
