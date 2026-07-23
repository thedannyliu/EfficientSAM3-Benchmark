#!/usr/bin/env bash
#SBATCH --job-name=tv-trt-quant
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

module load python/3.12.5 cuda/12.6.1
source .venv/bin/activate

MODELS=(tv5 tv11 tv21)
MODES=(fp8 int8)
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / ${#MODES[@]}))
MODE_INDEX=$((SLURM_ARRAY_TASK_ID % ${#MODES[@]}))
MODEL="${MODELS[${MODEL_INDEX}]}"
MODE="${MODES[${MODE_INDEX}]}"
MODE="${MODE_OVERRIDE:-${MODE}}"
OP_SET="${OP_SET:-conv_matmul}"

EXPORTER=dynamo
if [[ "${MODEL}" == "tv21" ]]; then
  EXPORTER=legacy
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/distill}"
DISTILL_ROOT="${DISTILL_ROOT:-/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/SAM2-Distillation-Pipeline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/pace/tinyvit_trt_quantization/${SLURM_ARRAY_JOB_ID}}"

python scripts/pace_tinyvit_trt_encoder_smoke.py \
  --checkpoint "${CHECKPOINT_ROOT}/${MODEL}.pt" \
  --distill-root "${DISTILL_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/${MODEL}/${MODE}-${OP_SET}" \
  --precision fp32 \
  --quantization-mode "${MODE}" \
  --quantization-op-set "${OP_SET}" \
  --calibration-video "${CALIBRATION_VIDEO:-videos/test1.mov}" \
  --calibration-samples "${CALIBRATION_SAMPLES:-32}" \
  --calibration-method "${CALIBRATION_METHOD:-max}" \
  --exporter "${EXPORTER}" \
  --builder-optimization-level 5 \
  --workspace-gib "${WORKSPACE_GIB:-8}" \
  --warmup "${WARMUP:-10}" \
  --runs "${RUNS:-50}"
