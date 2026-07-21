#!/bin/bash
#SBATCH --job-name=canyonbench
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load anaconda3/2024.6
conda activate canyonbench

CONFIG_PATH="${1:?usage: sbatch slurm/run_open_weight.sbatch CONFIG.yaml}"
mkdir -p logs
canyonbench run "$CONFIG_PATH"

