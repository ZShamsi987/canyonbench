#!/bin/bash
#SBATCH --job-name=cb-acquire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# D3-D4: discover candidate sites and materialize every aligned source layer.
# Network- and I/O-bound with no GPU utility. Source tiles (100-300 GB) stay on
# Adroit for the whole project and are never transferred to Lambda.
#
# Resumable: every artifact is written atomically and skipped when it already
# matches the site grid, so a job that hits the wall clock can be resubmitted.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SOURCES_CONFIG="${1:-configs/trace_sources.yaml}"
CANDIDATES="${2:-$CANYONBENCH_DATA/manifests/trace_candidates.yaml}"
SITES="${3:-$CANYONBENCH_DATA/manifests/sites.yaml}"
PREPARED="$CANYONBENCH_DATA/manifests/trace_prepared_candidates.yaml"

mkdir -p "$(dirname "$CANDIDATES")" "$CANYONBENCH_DATA/reports" "$CANYONBENCH_DATA/cache"

if [ ! -f "$CANDIDATES" ]; then
  canyonbench trace discover-sites "$SOURCES_CONFIG" "$CANDIDATES" \
    --cache-dir "$CANYONBENCH_DATA/cache/site-discovery"
fi

# Atomic and restart-safe: an artifact that already matches the site grid is
# skipped, so a job killed at the wall clock is resumed by resubmitting.
canyonbench trace acquire-sources \
  "$SOURCES_CONFIG" \
  "$CANDIDATES" \
  "$CANYONBENCH_DATA/sources" \
  "$PREPARED" \
  "$CANYONBENCH_DATA/reports/source-acquisition.json" \
  ${WORLD10_PATH:+--flight-source "$WORLD10_PATH"}

canyonbench trace select-sites "$PREPARED" configs/trace.yaml "$SITES"

canyonbench trace gate-ablations configs/trace.yaml \
  "$CANYONBENCH_DATA/reports/gate-ablations.json"

echo "ACQUIRE OK -> $SITES"
