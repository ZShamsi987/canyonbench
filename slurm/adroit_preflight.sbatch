#!/bin/bash
#SBATCH --job-name=cb-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# D1 verification job. Confirms the allocation is live, records the partitions,
# GPU types, and maximum wall time visible to this account, and checks that the
# pinned environment resolves. Costs ten minutes and removes the most disruptive
# class of mid-project failure.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

echo "=== allocation ==="
id -un; hostname; date -Is
echo "cpus-per-task=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}"

echo "=== partitions, limits, and GPU types visible to this account ==="
sinfo -o "%20P %5a %10l %6D %10G %N" || true
scontrol show config 2>/dev/null | grep -iE "MaxJobCount|MaxArraySize|DefaultTime" || true

echo "=== storage ==="
df -h "$CANYONBENCH_DATA" "$HOME" || true

echo "=== pinned environment ==="
uv run --frozen canyonbench --version
uv run --frozen canyonbench doctor
uv run --frozen canyonbench trace compute-check --role adroit \
  ${CANYONBENCH_DATASET_DIR:+--dataset-dir "$CANYONBENCH_DATASET_DIR"}

echo "PREFLIGHT OK"
