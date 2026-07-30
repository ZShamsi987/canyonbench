#!/bin/bash
#SBATCH --job-name=cb-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# D5-D6: generate 960 clean views, the 240 singly degraded counterparts, and
# every intervention, then validate the frozen bundles. GDAL, rasterio, and
# OpenCV parallelize across cores; a GPU instance would consume credited time
# for no benefit.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PROJECT_CONFIG="${1:-configs/trace.yaml}"
DATASET="${CANYONBENCH_DATASET_DIR:-$CANYONBENCH_DATA/generated}"
REPORTS="$CANYONBENCH_DATA/reports"
mkdir -p "$REPORTS"

canyonbench trace build "$PROJECT_CONFIG"

canyonbench trace validate "$DATASET" \
  --config "$PROJECT_CONFIG" \
  --output "$REPORTS/dataset-validation.json"

# Instrument products of the frozen bundles, both required before the freeze.
canyonbench trace fidelity-report "$DATASET" \
  --output "$REPORTS/relief-displacement.json"
canyonbench trace extinction-figure "$DATASET" "$REPORTS/figures"

echo "BUILD OK -> $DATASET"
