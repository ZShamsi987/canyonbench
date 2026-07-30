#!/bin/sh
set -eu

DATASET_DIR="${1:?usage: scripts/release.sh DATASET_DIR OUTPUT_DIR [CONFIG]}"
OUTPUT_DIR="${2:?usage: scripts/release.sh DATASET_DIR OUTPUT_DIR [CONFIG]}"
CONFIG="${3:-configs/trace.yaml}"
canyonbench trace validate "$DATASET_DIR" --config "$CONFIG"
canyonbench trace release "$DATASET_DIR" "$OUTPUT_DIR"
python -m build
python -m twine check dist/*
