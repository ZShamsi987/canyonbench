#!/bin/bash
#SBATCH --job-name=cb-acquire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:45:00
#SBATCH --array=0-9%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# D3-D4: discover candidate sites and materialize every aligned source layer.
# Network- and I/O-bound with no GPU utility. Source tiles (100-300 GB) stay on
# Adroit for the whole project and are never transferred to Lambda.
#
# Chunked into ten array tasks of under four hours each, two at a time, so the
# job fits under any plausible wall-clock limit and stays polite to the upstream
# services. Every artifact is written atomically and skipped when it already
# matches the site grid, so a killed task is resumed by resubmitting.
#
# Raise --time and drop --array once `qos` on this account confirms a longer
# limit; the commands are identical either way.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SOURCES_CONFIG="${1:-configs/trace_sources.yaml}"
CANDIDATES="${2:-$CANYONBENCH_DATA/manifests/trace_candidates.yaml}"
SITES="${3:-$CANYONBENCH_DATA/manifests/sites.yaml}"
PREPARED="$CANYONBENCH_DATA/manifests/trace_prepared_candidates.yaml"

mkdir -p "$(dirname "$CANDIDATES")" "$CANYONBENCH_DATA/reports" "$CANYONBENCH_DATA/cache"

CHUNK="${SLURM_ARRAY_TASK_ID:-0}"
CHUNKS="${CANYONBENCH_ACQUIRE_CHUNKS:-10}"
PER_CHUNK="${CANYONBENCH_SITES_PER_CHUNK:-25}"
START=$((CHUNK * PER_CHUNK))

# Task 0 owns discovery; the rest wait for the shared candidate manifest so the
# upstream services are queried once rather than ten times.
if [ "$CHUNK" -eq 0 ]; then
  if [ ! -f "$CANDIDATES" ]; then
    canyonbench trace discover-sites "$SOURCES_CONFIG" "$CANDIDATES" \
      --cache-dir "$CANYONBENCH_DATA/cache/site-discovery"
  fi
else
  for _ in $(seq 1 120); do
    [ -f "$CANDIDATES" ] && break
    sleep 30
  done
  if [ ! -f "$CANDIDATES" ]; then
    echo "candidate manifest never appeared; check the task 0 log" >&2
    exit 75
  fi
fi

echo "== chunk $CHUNK/$CHUNKS: sites [$START, $((START + PER_CHUNK))) =="
canyonbench trace acquire-sources \
  "$SOURCES_CONFIG" \
  "$CANDIDATES" \
  "$CANYONBENCH_DATA/sources" \
  "$PREPARED.chunk$CHUNK" \
  "$CANYONBENCH_DATA/reports/source-acquisition.chunk$CHUNK.json" \
  --start "$START" \
  --limit "$PER_CHUNK" \
  ${WORLD10_PATH:+--flight-source "$WORLD10_PATH"}

# Array tasks finish in no guaranteed order, so the merge and the cohort freeze
# are a separate dependent job rather than a race inside the last chunk:
#   sbatch --dependency=afterok:<this-array-job-id> slurm/adroit_freeze.sbatch
echo "CHUNK $CHUNK OK -> $PREPARED.chunk$CHUNK"
