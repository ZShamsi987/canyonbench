#!/bin/bash
# One-time Adroit setup. Run once on a login node, then never again.
#
# Adroit does the CPU half of the project. Nothing here requests a GPU.
#
# ISOLATION CONTRACT. This script creates or updates exactly two directories:
#
#   /scratch/network/$USER/canyonbench-trace     the project checkout and venv
#   /scratch/network/$USER/canyonbench-trace-data   all data and results
#
# It does NOT edit ~/.bashrc, load or modify any module, touch any conda
# environment, write to any shared prefix, or go near an existing checkout. If a
# directory is already present and is not this project, the script stops rather
# than writing into it. Both paths are overridable:
#
#   CANYONBENCH_HOME=... CANYONBENCH_DATA=... bash scripts/adroit/bootstrap.sh
set -euo pipefail

CANYONBENCH_HOME="${CANYONBENCH_HOME:-/scratch/network/$USER/canyonbench-trace}"
CANYONBENCH_DATA="${CANYONBENCH_DATA:-/scratch/network/$USER/canyonbench-trace-data}"
REPO_URL="${REPO_URL:-https://github.com/ZShamsi987/canyonbench.git}"

is_this_project() {
  [ -f "$1/pyproject.toml" ] && grep -q '^name = "canyonbench"' "$1/pyproject.toml" 2>/dev/null
}

echo "== paths this script will touch, and nothing else =="
echo "  checkout : $CANYONBENCH_HOME"
echo "  data     : $CANYONBENCH_DATA"
echo "  uv binary: $HOME/.local/bin/uv (user-local)"
echo

if [ -e "$CANYONBENCH_HOME" ] && ! is_this_project "$CANYONBENCH_HOME"; then
  echo "REFUSING TO CONTINUE: $CANYONBENCH_HOME already exists and is not a" >&2
  echo "CanyonBench checkout. Nothing has been modified. Re-run with a different" >&2
  echo "location, for example:" >&2
  echo "  CANYONBENCH_HOME=\$HOME/canyonbench-trace2 bash \$0" >&2
  exit 78
fi

# /home is a 10 GiB quota shared with every other project on the account, so the
# environment and the package cache both live on scratch.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CANYONBENCH_DATA/.uv-cache}"

echo "== installing uv (user-local, no module needed) =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "== checking out the project =="
if [ ! -d "$CANYONBENCH_HOME/.git" ]; then
  git clone "$REPO_URL" "$CANYONBENCH_HOME"
else
  # Only ever pull a checkout whose origin really is this project.
  origin="$(git -C "$CANYONBENCH_HOME" remote get-url origin 2>/dev/null || echo '')"
  case "$origin" in
    *canyonbench*) git -C "$CANYONBENCH_HOME" pull --ff-only || true ;;
    *) echo "Skipping pull: unexpected origin '$origin'" >&2 ;;
  esac
fi
cd "$CANYONBENCH_HOME"

echo "== resolving the pinned environment into $CANYONBENCH_HOME/.venv =="
# `trace` carries rasterio, GDAL bindings, OpenCV, shapely, matplotlib,
# statsmodels, and scikit-learn: the whole CPU half of the project. `dev`
# provides the ruff, mypy, and pytest commands used by scripts/check.sh.
uv sync --frozen --extra trace --extra dev

echo "== creating the data root =="
mkdir -p \
  "$CANYONBENCH_DATA"/{sources,generated,manifests,reports,audits,calibration,cache} \
  "$CANYONBENCH_DATA"/runs/{openrouter,lambda,price-pilot} \
  "$CANYONBENCH_DATA"/results/final \
  "$CANYONBENCH_HOME/logs"

cat <<PROFILE

Done. Nothing outside those two directories was modified, and ~/.bashrc was NOT
edited. Add these lines to your shell profile yourself if you want them
persistent:

  export PATH="\$HOME/.local/bin:\$PATH"
  export CANYONBENCH_HOME="$CANYONBENCH_HOME"
  export CANYONBENCH_DATA="$CANYONBENCH_DATA"
  export CANYONBENCH_DATASET_DIR="$CANYONBENCH_DATA/generated"
  # Paste the key only in your private profile, never in the repo:
  # export OPENROUTER_API_KEY=sk-or-...

Then submit the ten-minute verification job:

  cd "$CANYONBENCH_HOME" && sbatch slurm/adroit_preflight.sbatch

To remove every trace of this project later:

  rm -rf "$CANYONBENCH_HOME" "$CANYONBENCH_DATA"

PROFILE
