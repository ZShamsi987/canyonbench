#!/bin/bash
#SBATCH --job-name=cb-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --nice=5000
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# D1 verification job. Confirms the allocation is live, records the partitions,
# GPU types, and maximum wall time visible to this account, and checks that the
# pinned environment resolves. Costs ten minutes and removes the most disruptive
# class of mid-project failure.

# Slurm stages this script under /var/spool, so BASH_SOURCE cannot locate the
# checkout.  SLURM_SUBMIT_DIR is the repository root at submission time.
source "${CANYONBENCH_HOME:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/_common.sh"

echo "=== allocation ==="
id -un; hostname; date -Is
echo "cpus-per-task=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}"

echo "=== partitions, limits, and GPU types visible to this account ==="
# %l is the partition wall-time limit: this is the number the schedule depends
# on. Princeton routes jobs by QOS, so `qos` is the authoritative view and the
# published docs deliberately do not fix the values.
sinfo -o "%20P %5a %10l %10L %6D %10G %N" || true
echo "--- qos (authoritative limits for this account) ---"
qos 2>/dev/null || sacctmgr -p show qos format=Name,MaxWall,MaxJobsPU,MaxTRESPU 2>/dev/null || true
echo "--- account associations ---"
sacctmgr -p show assoc user="$USER" format=Cluster,Account,Partition,QOS,MaxWall 2>/dev/null || true
scontrol show config 2>/dev/null | grep -iE "MaxJobCount|MaxArraySize|DefaultTime" || true

echo "=== storage ==="
df -h "$CANYONBENCH_DATA" "$HOME" || true

echo "=== pinned environment ==="
uv run --frozen canyonbench --version
uv run --frozen canyonbench doctor
uv run --frozen canyonbench trace compute-check --role adroit \
  ${CANYONBENCH_DATASET_DIR:+--dataset-dir "$CANYONBENCH_DATASET_DIR"}

echo "PREFLIGHT OK"
