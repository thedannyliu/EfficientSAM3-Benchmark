#!/usr/bin/env bash
#SBATCH --job-name=tv21-patch-fp8-cal
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

module load python/3.12.5 "${CUDA_MODULE:-cuda/12.6.1}"
source .venv/bin/activate

METHODS=(max max entropy entropy)
SAMPLES=(64 128 64 128)
METHOD="${METHODS[${SLURM_ARRAY_TASK_ID}]}"
COUNT="${SAMPLES[${SLURM_ARRAY_TASK_ID}]}"
LABEL="${METHOD}-${COUNT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/pace/tv21_patch_fp8_calibration/${SLURM_ARRAY_JOB_ID}}"

python scripts/pace_tinyvit_trt_encoder_smoke.py \
  --checkpoint checkpoints/distill/tv21.pt \
  --distill-root /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/SAM2-Distillation-Pipeline \
  --output-dir "${OUTPUT_ROOT}/${LABEL}" \
  --precision fp32 \
  --quantization-mode fp8 \
  --quantization-op-set conv_matmul \
  --quantization-scope-regex 'student\.backbone\.body\.patch_embed\.' \
  --calibration-video videos/test1.mov \
  --calibration-samples "${COUNT}" \
  --calibration-method "${METHOD}" \
  --exporter legacy \
  --builder-optimization-level 5 \
  --workspace-gib 8 \
  --warmup 10 \
  --runs 50
