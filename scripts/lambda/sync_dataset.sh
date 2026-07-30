#!/bin/bash
# Push the frozen dataset bundle from Adroit to the Lambda filesystem.
#
# Run this FROM ADROIT, after the dataset freeze. Source tiles (100-300 GB) stay
# on Adroit and are never transferred; only the rendered views, masks, and
# intervention images move (20-40 GB).
set -euo pipefail

CANYONBENCH_DATA="${CANYONBENCH_DATA:-/scratch/network/$USER/CanyonBench-data}"
DATASET="${CANYONBENCH_DATASET_DIR:-$CANYONBENCH_DATA/generated}"
LAMBDA_HOST="${LAMBDA_HOST:?set LAMBDA_HOST=ubuntu@<lambda-ip>}"
LAMBDA_ROOT="${LAMBDA_ROOT:-/lambda/canyonbench}"

if [ ! -f "$DATASET/index.json" ]; then
  echo "No frozen dataset at $DATASET; run the build job first." >&2
  exit 1
fi

echo "== transferring $(du -sh "$DATASET" | cut -f1) to $LAMBDA_HOST:$LAMBDA_ROOT/dataset =="
ssh "$LAMBDA_HOST" "mkdir -p $LAMBDA_ROOT/dataset"

# Depth rasters and source manifests are not read during inference; excluding
# them keeps the transfer to the images the models actually see.
rsync -avh --partial --info=progress2 \
  --exclude 'depth.tif' \
  "$DATASET/" "$LAMBDA_HOST:$LAMBDA_ROOT/dataset/"

echo "== verifying the far side =="
ssh "$LAMBDA_HOST" "cd $LAMBDA_ROOT/CanyonBench && uv run --frozen canyonbench trace compute-check \
  --role lambda --storage-root $LAMBDA_ROOT --dataset-dir $LAMBDA_ROOT/dataset"

echo "SYNC OK"
