#!/usr/bin/env bash
#SBATCH --job-name=tv-trt-attn-block
#SBATCH --partition=gpu-h200
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --gres=gpu:h200:1
#SBATCH --array=0-7%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

module load python/3.12.5 "${CUDA_MODULE:-cuda/12.6.1}"
source .venv/bin/activate

STAGES=(1 1 2 2 2 2 2 2)
BLOCKS=(0 1 0 1 2 3 4 5)
STAGE="${STAGES[${SLURM_ARRAY_TASK_ID}]}"
BLOCK="${BLOCKS[${SLURM_ARRAY_TASK_ID}]}"
LABEL="stage${STAGE}_block${BLOCK}_attention"
SCOPE="student\\.backbone\\.body\\.stages_${STAGE}\\.blocks\\.${BLOCK}\\.attn$"
MODEL="${MODEL:-tv5}"
EXPORTER=dynamo
if [[ "${MODEL}" == "tv21" ]]; then
  EXPORTER=legacy
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-results/pace/tinyvit_trt_attention_block_sensitivity/${SLURM_ARRAY_JOB_ID}}"

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
