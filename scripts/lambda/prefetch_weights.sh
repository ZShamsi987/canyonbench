#!/bin/bash
# C1 weight pre-retrieval: download every served model to the persistent
# filesystem before any inference begins, so no GPU time is spent waiting on
# network transfer and later sessions incur no download cost.
set -euo pipefail

CANYONBENCH_ROOT="${CANYONBENCH_ROOT:-/lambda/canyonbench}"
CANYONBENCH_HOME="${CANYONBENCH_HOME:-$CANYONBENCH_ROOT/CanyonBench}"
CONFIG="${1:-$CANYONBENCH_HOME/configs/trace_run.lambda.yaml}"

export HF_HOME="${HF_HOME:-$CANYONBENCH_ROOT/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$CANYONBENCH_HOME"

# Served model IDs are the `model` field of every non-detector entry.
mapfile -t MODELS < <(
  uv run --frozen python - "$CONFIG" <<'PY'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for model in config["models"]:
    if model.get("benchmark_role") != "detector":
        print(model.get("served_model_id") or model["id"])
PY
)

echo "== pre-retrieving ${#MODELS[@]} model(s) into $HF_HOME =="
for model in "${MODELS[@]}"; do
  echo "-- $model"
  uv run --frozen python - "$model" <<'PY'
import sys
from huggingface_hub import snapshot_download

path = snapshot_download(
    sys.argv[1],
    # Safetensors only: no .bin duplicates, no consolidated checkpoints.
    allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py"],
    max_workers=8,
)
print(f"   -> {path}")
PY
done

du -sh "$HF_HOME"
echo "PREFETCH OK"
