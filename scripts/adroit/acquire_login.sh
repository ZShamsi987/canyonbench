#!/bin/bash
# Source acquisition on the Adroit LOGIN node.
#
# WHY NOT A BATCH JOB. Adroit compute nodes have no outbound network: DNS does
# not resolve and TCP 443 is blocked, with no proxy available. Verified directly:
#
#   srun ... curl https://api.github.com   ->  000, DNS fails, TCP 443 blocked
#   login node                             ->  200
#
# Every other stage stays in Slurm, because only this one needs the network.
# Generation, gates, interventions, scoring, and figures are all CPU-bound and
# run as batch jobs with no internet requirement.
#
# LOGIN-NODE ETIQUETTE. This is I/O-bound, not CPU-bound: one process, niced to
# the floor, so it cannot compete with anyone's interactive work. It is fully
# resumable — every artifact is written atomically and skipped when it already
# matches the site grid — so it can be stopped and restarted at any point.
#
#   bash scripts/adroit/acquire_login.sh              # full run, detached
#   CANYONBENCH_LIMIT=1 bash scripts/adroit/acquire_login.sh   # smoke test
set -euo pipefail

CANYONBENCH_HOME="${CANYONBENCH_HOME:-/scratch/network/$USER/canyonbench-trace}"
CANYONBENCH_DATA="${CANYONBENCH_DATA:-/scratch/network/$USER/canyonbench-trace-data}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CANYONBENCH_DATA/.uv-cache}"
export PATH="$HOME/.local/bin:$PATH"
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif,.tiff
# One download at a time; this must never look like a parallel scraper.
export GDAL_NUM_THREADS=1

SOURCES_CONFIG="${SOURCES_CONFIG:-configs/trace_sources.yaml}"
CANDIDATES="$CANYONBENCH_DATA/manifests/trace_candidates.yaml"
PREPARED="$CANYONBENCH_DATA/manifests/trace_prepared_candidates.yaml"
LOGS="$CANYONBENCH_DATA/logs"
mkdir -p "$(dirname "$CANDIDATES")" "$LOGS" "$CANYONBENCH_DATA/reports" "$CANYONBENCH_DATA/cache"
cd "$CANYONBENCH_HOME"

run() { nice -n 19 uv run --frozen canyonbench "$@"; }

if [ ! -f "$CANDIDATES" ]; then
  echo "== discovering candidate sites =="
  run trace discover-sites "$SOURCES_CONFIG" "$CANDIDATES" \
    --cache-dir "$CANYONBENCH_DATA/cache/site-discovery"
else
  echo "== candidates already frozen: $CANDIDATES =="
fi

echo "== acquiring source layers =="
run trace acquire-sources \
  "$SOURCES_CONFIG" \
  "$CANDIDATES" \
  "$CANYONBENCH_DATA/sources" \
  "$PREPARED" \
  "$CANYONBENCH_DATA/reports/source-acquisition.json" \
  ${CANYONBENCH_START:+--start "$CANYONBENCH_START"} \
  ${CANYONBENCH_LIMIT:+--limit "$CANYONBENCH_LIMIT"}

echo "ACQUIRE OK -> $PREPARED"
echo "Next (batch, no network needed):"
echo "  sbatch slurm/adroit_freeze.sbatch"
