#!/usr/bin/env bash
#SBATCH --job-name=tv-trt-aux
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

module load python/3.12.5 cuda/12.6.1
source .venv/bin/activate

MODEL="${MODEL:-tv5}"
EXPORTER=dynamo
if [[ "${MODEL}" == "tv21" ]]; then
  EXPORTER=legacy
fi

python scripts/pace_tinyvit_trt_encoder_smoke.py \
  --checkpoint "checkpoints/distill/${MODEL}.pt" \
  --distill-root /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/SAM2-Distillation-Pipeline \
  --output-dir "results/pace/tinyvit_trt_aux_streams/${SLURM_ARRAY_JOB_ID}/${MODEL}/aux-${SLURM_ARRAY_TASK_ID}" \
  --precision fp16 \
  --exporter "${EXPORTER}" \
  --builder-optimization-level 5 \
  --max-aux-streams "${SLURM_ARRAY_TASK_ID}" \
  --workspace-gib 8 \
  --warmup 20 \
  --runs 100
