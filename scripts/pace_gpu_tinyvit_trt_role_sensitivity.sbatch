#!/usr/bin/env bash
#SBATCH --job-name=tv-trt-role
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

module load python/3.12.5 cuda/12.6.1
source .venv/bin/activate

LABELS=(attn_qkv attn_projection mlp_fc1 mlp_fc2 local_conv)
SCOPES=(
  '\.attn\.qkv$'
  '\.attn\.proj$'
  '\.mlp\.fc1$'
  '\.mlp\.fc2$'
  '\.local_conv\.'
)

MODEL="${MODEL:-tv5}"
EXPORTER=dynamo
if [[ "${MODEL}" == "tv21" ]]; then
  EXPORTER=legacy
fi
LABEL="${LABELS[${SLURM_ARRAY_TASK_ID}]}"
SCOPE="${SCOPES[${SLURM_ARRAY_TASK_ID}]}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/pace/tinyvit_trt_role_sensitivity/${SLURM_ARRAY_JOB_ID}}"

python scripts/pace_tinyvit_trt_encoder_smoke.py \
  --checkpoint "checkpoints/distill/${MODEL}.pt" \
  --distill-root /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/SAM2-Distillation-Pipeline \
  --output-dir "${OUTPUT_ROOT}/${MODEL}/${LABEL}" \
  --precision fp32 \
  --quantization-mode fp8 \
  --quantization-op-set conv_matmul \
  --quantization-scope-regex "${SCOPE}" \
  --calibration-video "${CALIBRATION_VIDEO:-videos/test1.mov}" \
  --calibration-samples "${CALIBRATION_SAMPLES:-32}" \
  --calibration-method max \
  --exporter "${EXPORTER}" \
  --builder-optimization-level 5 \
  --workspace-gib 8 \
  --warmup 10 \
  --runs 50
