#!/usr/bin/env bash
#SBATCH --job-name=esam3-prof
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load python/3.12.5 cuda/12.6.1
source .venv/bin/activate

mkdir -p logs results overlays checkpoints

sam-env-probe
python -m sam_backend.variant_runner \
  --videos ${SAM_VIDEOS:-videos/test1.mov videos/test2.mov} \
  --prompt "${SAM_PROMPT:-monitor}" \
  --max-frames "${SAM_MAX_FRAMES:-30}" \
  --frame-stride "${SAM_FRAME_STRIDE:-1}" \
  --checkpoint-dir checkpoints \
  --output-dir "results/efficientsam3_variants/${SLURM_JOB_ID}" \
  --overlay-dir "overlays/efficientsam3_variants/${SLURM_JOB_ID}"
