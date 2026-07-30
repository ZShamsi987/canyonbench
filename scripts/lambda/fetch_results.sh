#!/bin/bash
# Pull the Lambda prediction log back to Adroit for merging and scoring.
#
# Run this FROM ADROIT once the Lambda roster has finished. Results are small;
# nothing else needs to come back.
set -euo pipefail

CANYONBENCH_DATA="${CANYONBENCH_DATA:-/scratch/network/$USER/CanyonBench-data}"
LAMBDA_HOST="${LAMBDA_HOST:?set LAMBDA_HOST=ubuntu@<lambda-ip>}"
LAMBDA_ROOT="${LAMBDA_ROOT:-/lambda/canyonbench}"
DESTINATION="$CANYONBENCH_DATA/runs/lambda"

mkdir -p "$DESTINATION"
rsync -avh --partial --info=progress2 \
  "$LAMBDA_HOST:$LAMBDA_ROOT/results/" "$DESTINATION/"

if [ ! -f "$DESTINATION/predictions.jsonl" ]; then
  echo "No predictions.jsonl arrived; check $LAMBDA_ROOT/logs on the instance." >&2
  exit 1
fi

echo "$(wc -l < "$DESTINATION/predictions.jsonl") prediction rows in $DESTINATION"
echo "FETCH OK. The Lambda instance can now be terminated."
