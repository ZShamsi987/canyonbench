#!/bin/bash
#SBATCH --job-name=cb-instruments
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nice=5000
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# D7: instrument validation V1 (synthetic inserts), V2 (edit detectability), and
# the audit packet. V6 suppression efficacy is computed by `trace score`.
# DATASET FREEZE follows this job: no generator change afterwards without
# restarting model runs.

# Slurm stages this script under /var/spool, so BASH_SOURCE cannot locate the
# checkout.  SLURM_SUBMIT_DIR is the repository root at submission time.
source "${CANYONBENCH_HOME:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/_common.sh"

DATASET="${CANYONBENCH_DATASET_DIR:-$CANYONBENCH_DATA/generated}"
REPORTS="$CANYONBENCH_DATA/reports"
NEGATIVE_VIEW="${1:?usage: sbatch slurm/adroit_instruments.sbatch NEGATIVE_RGB.png}"
mkdir -p "$REPORTS" "$CANYONBENCH_DATA/audits"

canyonbench trace instrument-v1 "$NEGATIVE_VIEW" "$REPORTS/v1-controls"
canyonbench trace instrument-v2 "$DATASET" "$REPORTS/v2-edit-detectability.json"

canyonbench trace audit-sample "$DATASET" "$CANYONBENCH_DATA/audits/audit.csv" \
  --auditor-1 AUD-KUNSH --auditor-2 AUD-ATHARVA --fraction 0.1

echo "INSTRUMENTS OK; ship $CANYONBENCH_DATA/audits/audit.csv to both auditors"
