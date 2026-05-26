#!/usr/bin/env bash
#SBATCH --job-name=sam3-bench
#SBATCH --partition=gpu-l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load python/3.12.5 cuda/12.6.1

source .venv/bin/activate
mkdir -p logs results

sam-env-probe
sam-benchmark \
  --backend "${SAM_BACKEND:-null}" \
  --checkpoint-path "${SAM_CHECKPOINT_PATH:-}" \
  --device cuda \
  --synthetic-frames "${SAM_SYNTHETIC_FRAMES:-16}" \
  --prompt "${SAM_PROMPT:-person}" \
  --warmup 5 \
  --runs 50 \
  --output "results/${SAM_BACKEND:-null}-${SLURM_JOB_ID}.json"
