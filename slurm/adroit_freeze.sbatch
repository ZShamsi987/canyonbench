#!/bin/bash
#SBATCH --job-name=cb-freeze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --nice=5000
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Runs after the acquisition array completes:
#   sbatch --dependency=afterok:<array-job-id> slurm/adroit_freeze.sbatch
#
# Merges the per-chunk manifests, solves the exact 120-site 20/20/60 cohort, and
# records the registered gate ablations. Array tasks finish out of order, so this
# is a dependent job rather than a race inside the last chunk.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PREPARED="$CANYONBENCH_DATA/manifests/trace_prepared_candidates.yaml"
SITES="${1:-$CANYONBENCH_DATA/manifests/sites.yaml}"

shopt -s nullglob
CHUNKS=("$PREPARED".chunk*)
shopt -u nullglob
if [ "${#CHUNKS[@]}" -eq 0 ]; then
  echo "No chunk manifests beside $PREPARED; did the acquisition array run?" >&2
  exit 75
fi

canyonbench trace merge-sites "$PREPARED" "${CHUNKS[@]}"
canyonbench trace select-sites "$PREPARED" configs/trace.yaml "$SITES"
canyonbench trace gate-ablations configs/trace.yaml \
  "$CANYONBENCH_DATA/reports/gate-ablations.json"

echo "FREEZE OK -> $SITES"
