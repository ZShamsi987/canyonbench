#!/bin/bash
#SBATCH --job-name=cb-freeze
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Gate evaluation reads independent source rasters; the worker count follows
# SLURM_CPUS_PER_TASK, and the cohort is identical at any width.
#SBATCH --cpus-per-task=16
# Each gate evaluation holds a full-resolution site in memory: a 12530x12530
# three-band image, six binary masks, and three float32 detector layers, roughly
# 4 GB. That is why memory scales with cpus-per-task here - 16 workers against
# the old 64 GB request was OOM-killed at 67 GB.
#SBATCH --mem=300G
#SBATCH --time=02:00:00
#SBATCH --nice=5000
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Runs after source acquisition completes.  The historical array workflow writes
# one manifest per chunk; the Adroit login-node workflow writes one complete,
# resumable manifest.  Accept both forms so the downstream freeze is identical
# regardless of where egress was available.

# Slurm stages this script under /var/spool, so BASH_SOURCE cannot locate the
# checkout.  SLURM_SUBMIT_DIR is the repository root at submission time.
source "${CANYONBENCH_HOME:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/_common.sh"

PREPARED="$CANYONBENCH_DATA/manifests/trace_prepared_candidates.yaml"
SITES="${1:-$CANYONBENCH_DATA/manifests/sites.yaml}"

shopt -s nullglob
CHUNKS=("$PREPARED".chunk*)
shopt -u nullglob
if [ "${#CHUNKS[@]}" -gt 0 ]; then
  echo "Merging ${#CHUNKS[@]} acquisition chunk manifest(s)."
  canyonbench trace merge-sites "$PREPARED" "${CHUNKS[@]}"
elif [ -f "$PREPARED" ]; then
  echo "Using complete manifest from login-node acquisition: $PREPARED"
else
  echo "No prepared-site manifest found at $PREPARED; acquisition has not completed." >&2
  exit 75
fi

canyonbench trace select-sites "$PREPARED" configs/trace.yaml "$SITES"
canyonbench trace gate-ablations configs/trace.yaml \
  "$CANYONBENCH_DATA/reports/gate-ablations.json"

echo "FREEZE OK -> $SITES"
