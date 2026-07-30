#!/bin/bash
# One-time Adroit setup. Run once on a login node, then never again.
#
# Adroit does the CPU half of the project. Nothing here requests a GPU.
set -euo pipefail

CANYONBENCH_HOME="${CANYONBENCH_HOME:-$HOME/CanyonBench}"
CANYONBENCH_DATA="${CANYONBENCH_DATA:-/scratch/network/$USER/CanyonBench-data}"
REPO_URL="${REPO_URL:-https://github.com/ZShamsi987/canyonbench.git}"

echo "== installing uv (user-local, no module needed) =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "== checking out the project =="
if [ ! -d "$CANYONBENCH_HOME/.git" ]; then
  git clone "$REPO_URL" "$CANYONBENCH_HOME"
fi
cd "$CANYONBENCH_HOME"
git pull --ff-only || true

echo "== resolving the pinned environment =="
# `trace` carries rasterio, GDAL bindings, OpenCV, shapely, matplotlib,
# statsmodels, and scikit-learn: the whole CPU half of the project.
uv sync --frozen --extra trace

echo "== creating the data root =="
mkdir -p \
  "$CANYONBENCH_DATA"/{sources,generated,manifests,reports,audits,calibration} \
  "$CANYONBENCH_DATA"/runs/{openrouter,lambda,price-pilot} \
  "$CANYONBENCH_DATA"/results/final \
  "$CANYONBENCH_HOME/logs"

cat <<PROFILE

Add these to ~/.bashrc so every job and login shell agrees:

  export PATH="\$HOME/.local/bin:\$PATH"
  export CANYONBENCH_HOME="$CANYONBENCH_HOME"
  export CANYONBENCH_DATA="$CANYONBENCH_DATA"
  export CANYONBENCH_DATASET_DIR="$CANYONBENCH_DATA/generated"
  # Paste the key here only in your private shell profile, never in the repo:
  # export OPENROUTER_API_KEY=sk-or-...

Then submit the ten-minute verification job:

  cd "$CANYONBENCH_HOME" && sbatch slurm/adroit_preflight.sbatch

PROFILE
