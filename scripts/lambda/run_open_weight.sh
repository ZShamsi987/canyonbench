#!/bin/bash
# C3 sequential model execution inside a single GPU session.
#
# Each locally served model is started on its own port, traced, then released,
# so instance startup and filesystem mount are amortized across the whole roster
# instead of being paid once per model. The trace log is append-only and keyed by
# a content hash of the request, so an interrupted session resumes at the point
# of failure (C2) without re-running or re-serving completed work.
set -euo pipefail

CANYONBENCH_ROOT="${CANYONBENCH_ROOT:-/lambda/canyonbench}"
CANYONBENCH_HOME="${CANYONBENCH_HOME:-$CANYONBENCH_ROOT/CanyonBench}"
CONFIG="${1:-$CANYONBENCH_HOME/configs/trace_run.frozen.yaml}"
DATASET="${CANYONBENCH_DATASET_DIR:-$CANYONBENCH_ROOT/dataset}"
RESULTS="${CANYONBENCH_RESULTS_DIR:-$CANYONBENCH_ROOT/results}"
LOGS="$CANYONBENCH_ROOT/logs"

export HF_HOME="${HF_HOME:-$CANYONBENCH_ROOT/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_API_KEY="${VLLM_API_KEY:-canyonbench-local}"
mkdir -p "$LOGS" "$RESULTS"
cd "$CANYONBENCH_HOME"

# C4 capability-adaptive configuration: one script, every admissible instance.
read -r -a PROFILE_ARGS <<<"$(uv run --frozen canyonbench trace vllm-profile --server-args)"
echo "== serving profile: ${PROFILE_ARGS[*]} =="

SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "-- releasing VRAM (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT INT TERM

# Locally served models are the unmetered, non-detector entries: id, weights,
# and the port their adapter expects.
mapfile -t SERVED < <(
  uv run --frozen python - "$CONFIG" <<'PY'
import sys, urllib.parse, yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for model in config["models"]:
    if model.get("metered", True) or model.get("benchmark_role") == "detector":
        continue
    url = urllib.parse.urlparse(model["adapter"]["base_url"])
    if url.hostname not in {"127.0.0.1", "localhost"}:
        continue
    print(f"{model['id']}\t{model.get('served_model_id') or model['id']}\t{url.port or 8000}")
PY
)

if [ "${#SERVED[@]}" -eq 0 ]; then
  echo "No locally served models in $CONFIG." >&2
  exit 1
fi

for entry in "${SERVED[@]}"; do
  IFS=$'\t' read -r benchmark_id served_id port <<<"$entry"
  safe_name="${benchmark_id//\//_}"
  echo "=== $benchmark_id  (weights: $served_id, port: $port) ==="

  uv run --frozen python -m vllm.entrypoints.openai.api_server \
    --model "$served_id" \
    --served-model-name "$benchmark_id" \
    --port "$port" \
    --api-key "$VLLM_API_KEY" \
    --download-dir "$HF_HOME" \
    "${PROFILE_ARGS[@]}" \
    >"$LOGS/vllm-$safe_name.log" 2>&1 &
  SERVER_PID=$!

  echo "-- waiting for readiness on :$port"
  ready=0
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null; then ready=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    sleep 5
  done
  if [ "$ready" -ne 1 ]; then
    echo "$benchmark_id did not become ready; see $LOGS/vllm-$safe_name.log" >&2
    echo "If this model needs a bespoke wrapper rather than plain vLLM, start it" >&2
    echo "manually on :$port and rerun with --only-model $benchmark_id." >&2
    exit 1
  fi

  uv run --frozen canyonbench trace run "$CONFIG" \
    --only-model "$benchmark_id" \
    --dataset-dir "$DATASET" \
    --output-dir "$RESULTS"

  cleanup
done

echo "LAMBDA RUN OK -> $RESULTS/predictions.jsonl"
echo "Now run scripts/lambda/fetch_results.sh from Adroit, then terminate the instance."
