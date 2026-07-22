#!/usr/bin/env bash
#SBATCH --job-name=tv-trt-mask
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

module load python/3.12.5 cuda/12.6.1
source .venv/bin/activate

MODELS=(tv5 tv11 tv21)
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
TIER="${TIER:-fp16}"
case "${TIER}:${MODEL}" in
  fp16:tv5|fp16:tv11)
    ENGINE_ROOT="${ENGINE_ROOT:-results/pace/tinyvit_trt_optimization/11367310}"
    ENGINE_VARIANT="fp16-dynamo-tf320-opt5"
    ENGINE_FILE="encoder.fp16.engine"
    ;;
  fp16:tv21)
    ENGINE_ROOT="${ENGINE_ROOT:-results/pace/tinyvit_trt_optimization/11367310}"
    ENGINE_VARIANT="fp16-legacy-tf320-opt5"
    ENGINE_FILE="encoder.fp16.engine"
    ;;
  tf32:tv5|tf32:tv11)
    ENGINE_ROOT="${ENGINE_ROOT:-results/pace/tinyvit_trt_optimization/11367310}"
    ENGINE_VARIANT="fp32-dynamo-tf321-opt5"
    ENGINE_FILE="encoder.fp32.engine"
    ;;
  tf32:tv21)
    ENGINE_ROOT="${ENGINE_ROOT:-results/pace/tinyvit_trt_optimization/11368732}"
    ENGINE_VARIANT="fp32-legacy-tf321-opt5"
    ENGINE_FILE="encoder.fp32.engine"
    ;;
  fp32:tv5)
    ENGINE_ROOT="${ENGINE_ROOT:-results/pace/tinyvit_trt_optimization/11367310}"
    ENGINE_VARIANT="fp32-dynamo-tf320-opt5"
    ENGINE_FILE="encoder.fp32.engine"
    ;;
  fp32:tv11)
    ENGINE_ROOT="${ENGINE_ROOT:-results/pace/tinyvit_trt_optimization/11368507}"
    ENGINE_VARIANT="fp32-legacy-tf320-opt5"
    ENGINE_FILE="encoder.fp32.engine"
    ;;
  fp32:tv21)
    ENGINE_ROOT="${ENGINE_ROOT:-results/pace/tinyvit_trt_optimization/11367310}"
    ENGINE_VARIANT="fp32-legacy-tf320-opt5"
    ENGINE_FILE="encoder.fp32.engine"
    ;;
  *)
    echo "Unsupported TIER:MODEL combination: ${TIER}:${MODEL}" >&2
    exit 2
    ;;
esac

OUTPUT_ROOT="${OUTPUT_ROOT:-results/pace/tinyvit_trt_mask_parity/${SLURM_ARRAY_JOB_ID}/${TIER}}"

python scripts/pace_tinyvit_trt_mask_parity.py \
  --student-checkpoint "checkpoints/distill/${MODEL}.pt" \
  --engine "${ENGINE_ROOT}/${MODEL}/${ENGINE_VARIANT}/${ENGINE_FILE}" \
  --sam2-checkpoint checkpoints/sam2/sam2.1_hiera_large.pt \
  --sam2-root external/sam2 \
  --distill-root /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/SAM2-Distillation-Pipeline \
  --video videos/test1.mov \
  --frames "${FRAMES:-8}" \
  --minimum-mask-iou "${MINIMUM_MASK_IOU:-0.999}" \
  --output "${OUTPUT_ROOT}/${MODEL}/report.json"
