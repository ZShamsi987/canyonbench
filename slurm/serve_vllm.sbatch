#!/bin/bash
#SBATCH --job-name=canyonbench-vllm
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load anaconda3/2024.6
conda activate canyonbench

MODEL_ID="${1:?usage: sbatch slurm/serve_vllm.sbatch MODEL_ID}"
PORT="${VLLM_PORT:-8000}"
mkdir -p logs
python -m vllm.entrypoints.openai.api_server --model "$MODEL_ID" --port "$PORT"

