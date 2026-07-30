#!/bin/bash
# One-time Lambda setup, plus the D1 twenty-minute verification.
#
# Lambda does the GPU half only: open-weight and RS-specialized VLM inference,
# and the incidental edit-detector work. Everything it needs lives on the
# persistent filesystem, so the instance type may change between sessions.
set -euo pipefail

CANYONBENCH_ROOT="${CANYONBENCH_ROOT:-/lambda/canyonbench}"
CANYONBENCH_HOME="${CANYONBENCH_HOME:-$CANYONBENCH_ROOT/CanyonBench}"
REPO_URL="${REPO_URL:-https://github.com/ZShamsi987/canyonbench.git}"

echo "== persistent filesystem layout =="
mkdir -p "$CANYONBENCH_ROOT"/{hf,dataset,results,logs}
export HF_HOME="$CANYONBENCH_ROOT/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "== installing uv =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "== checking out the project onto persistent storage =="
if [ ! -d "$CANYONBENCH_HOME/.git" ]; then
  git clone "$REPO_URL" "$CANYONBENCH_HOME"
fi
cd "$CANYONBENCH_HOME"
git pull --ff-only || true

echo "== resolving the pinned environment =="
uv sync --frozen --extra trace

# vLLM ships platform-specific CUDA wheels and is deliberately absent from
# uv.lock, which stays cross-platform so the same lockfile resolves on a laptop,
# on Adroit, and here. It is installed into the same environment instead.
echo "== installing the serving stack (not in the lockfile, by design) =="
uv pip install "vllm>=0.6" "huggingface_hub>=0.24" hf_transfer

cat > "$CANYONBENCH_ROOT/env.sh" <<PROFILE
export PATH="\$HOME/.local/bin:\$PATH"
export CANYONBENCH_ROOT="$CANYONBENCH_ROOT"
export CANYONBENCH_HOME="$CANYONBENCH_HOME"
export CANYONBENCH_DATASET_DIR="$CANYONBENCH_ROOT/dataset"
export CANYONBENCH_RESULTS_DIR="$CANYONBENCH_ROOT/results"
export HF_HOME="$CANYONBENCH_ROOT/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1
PROFILE
echo "wrote $CANYONBENCH_ROOT/env.sh (source it in every later session)"

echo "== D1 verification: filesystem, GPU, precision, vLLM =="
uv run --frozen canyonbench trace compute-check --role lambda \
  --storage-root "$CANYONBENCH_ROOT" \
  --output "$CANYONBENCH_ROOT/logs/compute-check.json"

echo
echo "Serving profile selected for this device:"
uv run --frozen canyonbench trace vllm-profile

cat <<'NEXT'

Next, and while still in this session:
  1. bash scripts/lambda/prefetch_weights.sh      # C1, before any inference
  2. (from Adroit) bash scripts/lambda/sync_dataset.sh
  3. bash scripts/lambda/run_open_weight.sh configs/trace_run.lambda.yaml

NEXT
