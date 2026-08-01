#!/bin/bash
#SBATCH --job-name=cb-analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --nice=5000
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# D15-D17: merge the Lambda and OpenRouter logs, tune and apply CAVE, compute
# every registered metric with site/group bootstrap, mixed effects, and
# Benjamini-Hochberg, then write the figures and the main table. CPU-bound.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATASET="${CANYONBENCH_DATASET_DIR:-$CANYONBENCH_DATA/generated}"
RESULTS="$CANYONBENCH_DATA/results/final"
RUNS="$CANYONBENCH_DATA/runs"
mkdir -p "$RESULTS" "$CANYONBENCH_DATA/reports/final"

canyonbench trace merge-runs \
  "$RUNS/lambda/predictions.jsonl" \
  "$RUNS/openrouter/predictions.jsonl" \
  --output "$RESULTS/predictions.jsonl"

canyonbench trace reference-baselines "$DATASET" "$RESULTS/reference-baselines.csv"

canyonbench trace cave-tune "$DATASET" "$RESULTS/predictions.jsonl" \
  "$RESULTS/cave-thresholds.json"
canyonbench trace cave-apply "$DATASET" "$RESULTS/predictions.jsonl" \
  "$RESULTS/cave-thresholds.json" "$RESULTS/cave-decisions.jsonl"
canyonbench trace cave-ablations "$DATASET" "$RESULTS/predictions.jsonl" \
  "$RESULTS/cave-thresholds.json" "$RESULTS/cave-ablations.jsonl"

canyonbench trace score "$DATASET" "$RESULTS/predictions.jsonl" "$RESULTS/metrics.json" \
  --cave-decisions "$RESULTS/cave-decisions.jsonl" \
  --cave-ablations "$RESULTS/cave-ablations.jsonl" \
  --cave-frontier "$RESULTS/cave-thresholds.frontier.json"

canyonbench trace report "$RESULTS/metrics.json" "$RESULTS/metrics.rows.csv" \
  "$CANYONBENCH_DATA/reports/final" --dataset-dir "$DATASET"

echo "ANALYZE OK -> $CANYONBENCH_DATA/reports/final"
