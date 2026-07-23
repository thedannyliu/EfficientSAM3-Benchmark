#!/usr/bin/env bash
#SBATCH --job-name=tv-trt-layer
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-15
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

module load python/3.12.5 "${CUDA_MODULE:-cuda/12.6.1}"
source .venv/bin/activate

LABELS=(
  patch_embed
  stage0_block0 stage0_block1
  stage1_downsample stage1_block0 stage1_block1
  stage2_downsample
  stage2_block0 stage2_block1 stage2_block2 stage2_block3 stage2_block4 stage2_block5
  projection_high_res_s0 projection_high_res_s1 projection_image_embed
)
SCOPES=(
  'student\.backbone\.body\.patch_embed\.'
  'student\.backbone\.body\.stages_0\.blocks\.0\.'
  'student\.backbone\.body\.stages_0\.blocks\.1\.'
  'student\.backbone\.body\.stages_1\.downsample\.'
  'student\.backbone\.body\.stages_1\.blocks\.0\.'
  'student\.backbone\.body\.stages_1\.blocks\.1\.'
  'student\.backbone\.body\.stages_2\.downsample\.'
  'student\.backbone\.body\.stages_2\.blocks\.0\.'
  'student\.backbone\.body\.stages_2\.blocks\.1\.'
  'student\.backbone\.body\.stages_2\.blocks\.2\.'
  'student\.backbone\.body\.stages_2\.blocks\.3\.'
  'student\.backbone\.body\.stages_2\.blocks\.4\.'
  'student\.backbone\.body\.stages_2\.blocks\.5\.'
  '^student\.projections\.high_res_s0$'
  '^student\.projections\.high_res_s1$'
  '^student\.projections\.image_embed$'
)

MODEL="${MODEL:-tv5}"
EXPORTER=dynamo
if [[ "${MODEL}" == "tv21" ]]; then
  EXPORTER=legacy
fi
LABEL="${LABELS[${SLURM_ARRAY_TASK_ID}]}"
SCOPE="${SCOPES[${SLURM_ARRAY_TASK_ID}]}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/pace/tinyvit_trt_layer_sensitivity/${SLURM_ARRAY_JOB_ID}}"

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
