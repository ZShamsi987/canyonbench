#!/bin/bash
#SBATCH --job-name=cb-openrouter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# D9-D13 (paid half): every model reached over the OpenRouter API. No local
# compute is required, so this runs on an Adroit CPU node. The run is
# append-only and resumable; resubmitting after a wall-clock kill continues from
# the last recorded request instead of re-paying for it.
#
# Export OPENROUTER_API_KEY in the submitting shell; sbatch propagates it by
# default. Never commit the key.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CONFIG="${1:-configs/trace_run.frozen.yaml}"
DATASET="${CANYONBENCH_DATASET_DIR:-$CANYONBENCH_DATA/generated}"
RUN_DIR="$CANYONBENCH_DATA/runs/openrouter"
REPORTS="$CANYONBENCH_DATA/reports"
mkdir -p "$RUN_DIR" "$REPORTS"

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY is not set; export it before sbatch." >&2
  exit 78
fi

canyonbench trace compute-check --role openrouter --dataset-dir "$DATASET"

# The metered slice of the roster: everything billed through the API.
mapfile -t PAID < <(
  uv run --frozen python - "$CONFIG" <<'PY'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for model in config["models"]:
    if model.get("metered", True):
        print(model["id"])
PY
)
ONLY=()
for model in "${PAID[@]}"; do ONLY+=(--only-model "$model"); done
echo "== paid roster slice: ${PAID[*]} =="

# Authorization gate: never spend against the registered token estimate alone.
if [ ! -f "$REPORTS/price-pilot.json" ]; then
  canyonbench trace price-pilot "$CONFIG" --calls 50 \
    --dataset-dir "$DATASET" \
    --output-dir "$CANYONBENCH_DATA/runs/price-pilot"
  cp "$CANYONBENCH_DATA/runs/price-pilot/price_pilot.json" "$REPORTS/price-pilot.json"
fi

canyonbench trace plan-run "$CONFIG" \
  --dataset-dir "$DATASET" \
  --price-pilot "$REPORTS/price-pilot.json" \
  --output "$REPORTS/call-plan-measured.json"

canyonbench trace run "$CONFIG" \
  --dataset-dir "$DATASET" \
  --output-dir "$RUN_DIR" \
  "${ONLY[@]}"

echo "OPENROUTER RUN OK -> $RUN_DIR/predictions.jsonl"
