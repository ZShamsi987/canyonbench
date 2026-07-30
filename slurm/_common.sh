#!/bin/bash
# Shared environment for every Adroit CPU job.
#
# Adroit runs only the CPU-bound half of the project: source acquisition, view
# generation, gates, interventions, validation, scoring, and figures. No job in
# this directory requests a GPU. Model inference happens on Lambda (open-weight)
# or over the OpenRouter API (proprietary).
set -euo pipefail

# Project checkout and the data root that holds sources, bundles, and results.
CANYONBENCH_HOME="${CANYONBENCH_HOME:-$HOME/CanyonBench}"
CANYONBENCH_DATA="${CANYONBENCH_DATA:-/scratch/network/$USER/CanyonBench-data}"

export CANYONBENCH_HOME CANYONBENCH_DATA
export GDAL_CACHEMAX="${GDAL_CACHEMAX:-512}"
export GDAL_NUM_THREADS="${GDAL_NUM_THREADS:-ALL_CPUS}"
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif,.tiff
# Keep numeric libraries from oversubscribing the allocated cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p "$CANYONBENCH_HOME/logs" "$CANYONBENCH_DATA"
cd "$CANYONBENCH_HOME"

# uv keeps the pinned lockfile authoritative; no conda environment to drift.
if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || {
  echo "uv is not installed. Run scripts/adroit/bootstrap.sh first." >&2
  exit 127
}

canyonbench() { uv run --frozen canyonbench "$@"; }
