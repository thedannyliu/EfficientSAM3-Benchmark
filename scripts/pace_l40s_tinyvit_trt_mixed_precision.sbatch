#!/usr/bin/env bash
#SBATCH --job-name=tv-trt-mixed
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-20
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

module load python/3.12.5 cuda/12.6.1
source .venv/bin/activate

MODELS=(tv5 tv11 tv21)
PROFILES=(
  fp16
  projection_fp32
  norm_fp32
  matmul_fp32
  conv_fp32
  matmul_projection_fp32
  conv_matmul_fp32
)

MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / ${#PROFILES[@]}))
PROFILE_INDEX=$((SLURM_ARRAY_TASK_ID % ${#PROFILES[@]}))
MODEL="${MODELS[${MODEL_INDEX}]}"
PROFILE="${PROFILES[${PROFILE_INDEX}]}"

EXPORTER=dynamo
if [[ "${MODEL}" == "tv21" ]]; then
  EXPORTER=legacy
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/distill}"
DISTILL_ROOT="${DISTILL_ROOT:-/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/SAM2-Distillation-Pipeline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/pace/tinyvit_trt_mixed_precision/${SLURM_ARRAY_JOB_ID}}"

python scripts/pace_tinyvit_trt_encoder_smoke.py \
  --checkpoint "${CHECKPOINT_ROOT}/${MODEL}.pt" \
  --distill-root "${DISTILL_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/${MODEL}/${PROFILE}" \
  --precision fp32 \
  --mixed-precision-profile "${PROFILE}" \
  --exporter "${EXPORTER}" \
  --builder-optimization-level 5 \
  --workspace-gib "${WORKSPACE_GIB:-8}" \
  --warmup "${WARMUP:-10}" \
  --runs "${RUNS:-50}"
