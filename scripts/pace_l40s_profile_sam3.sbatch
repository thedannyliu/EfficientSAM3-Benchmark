#!/usr/bin/env bash
#SBATCH --job-name=sam3-prof
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load python/3.12.5 cuda/12.6.1
source .venv/bin/activate

mkdir -p logs results overlays

sam-env-probe

VIDEOS=(${SAM_VIDEOS:-videos/test1.mov videos/test2.mov})
for video in "${VIDEOS[@]}"; do
  stem="$(basename "${video}")"
  stem="${stem%.*}"
  python -m sam_backend.profile_video \
    --model-id sam3 \
    --backend sam3 \
    --device cuda \
    --prompt "${SAM_PROMPT:-monitor}" \
    --video "${video}" \
    --max-frames "${SAM_MAX_FRAMES:-30}" \
    --frame-stride "${SAM_FRAME_STRIDE:-1}" \
    --csv-output "results/sam3-${SLURM_JOB_ID}-${stem}.csv" \
    --overlay-output "overlays/sam3-${SLURM_JOB_ID}-${stem}.mp4"
done
