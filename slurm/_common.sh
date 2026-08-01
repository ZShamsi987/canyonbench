#!/bin/bash
# Shared environment for every Adroit CPU job.
#
# Adroit runs only the CPU-bound half of the project: source acquisition, view
# generation, gates, interventions, validation, scoring, and figures. No job in
# this directory requests a GPU. Model inference happens on Lambda (open-weight)
# or over the OpenRouter API (proprietary).
#
# ISOLATION CONTRACT. These jobs write to exactly two directories and nowhere
# else on Adroit:
#
#   $CANYONBENCH_HOME   the project checkout (code, .venv, logs/)
#   $CANYONBENCH_DATA   sources, generated views, runs, reports, results
#
# Nothing here edits ~/.bashrc, installs into a shared prefix, touches another
# checkout, or writes outside those two roots. Both are overridable, and the
# guard below refuses to run if $CANYONBENCH_HOME is not this project.
set -euo pipefail

CANYONBENCH_HOME="${CANYONBENCH_HOME:-$HOME/canyonbench-trace}"
CANYONBENCH_DATA="${CANYONBENCH_DATA:-/scratch/network/$USER/canyonbench-trace-data}"

export CANYONBENCH_HOME CANYONBENCH_DATA
export GDAL_CACHEMAX="${GDAL_CACHEMAX:-512}"
export GDAL_NUM_THREADS="${GDAL_NUM_THREADS:-ALL_CPUS}"
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif,.tiff
# Keep numeric libraries from oversubscribing the allocated cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Refuse to operate on a directory that is not this project: a stray
# CANYONBENCH_HOME must never cause a job to write into unrelated work.
if [ ! -f "$CANYONBENCH_HOME/pyproject.toml" ] ||
   ! grep -q '^name = "canyonbench"' "$CANYONBENCH_HOME/pyproject.toml" 2>/dev/null; then
  echo "CANYONBENCH_HOME=$CANYONBENCH_HOME is not a CanyonBench checkout." >&2
  echo "Run scripts/adroit/bootstrap.sh, or set CANYONBENCH_HOME correctly." >&2
  exit 78
fi

mkdir -p "$CANYONBENCH_HOME/logs" "$CANYONBENCH_DATA"
cd "$CANYONBENCH_HOME"

# uv keeps the pinned lockfile authoritative; the environment lives in
# $CANYONBENCH_HOME/.venv and no conda module is loaded or modified.
if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || {
  echo "uv is not installed. Run scripts/adroit/bootstrap.sh first." >&2
  exit 127
}

canyonbench() { uv run --frozen canyonbench "$@"; }
