#!/usr/bin/env bash
#SBATCH --job-name=tv-trt-mask-candidate
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

: "${MODEL:?set MODEL to tv11 or tv21}"
: "${ENGINE:?set ENGINE to the candidate TensorRT engine path}"

module load python/3.12.5 "${CUDA_MODULE:-cuda/12.6.1}"
source .venv/bin/activate

LABEL="${LABEL:-candidate}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/pace/tinyvit_trt_candidate_mask_parity/${SLURM_JOB_ID}}"

python scripts/pace_tinyvit_trt_mask_parity.py \
  --student-checkpoint "checkpoints/distill/${MODEL}.pt" \
  --engine "${ENGINE}" \
  --sam2-checkpoint checkpoints/sam2/sam2.1_hiera_large.pt \
  --sam2-root external/sam2 \
  --distill-root /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/SAM2-Distillation-Pipeline \
  --video "${VIDEO:-videos/test1.mov}" \
  --frames "${FRAMES:-16}" \
  --minimum-mask-iou "${MINIMUM_MASK_IOU:-0.80}" \
  --minimum-mean-mask-iou "${MINIMUM_MEAN_MASK_IOU:-0.95}" \
  --output "${OUTPUT_ROOT}/${MODEL}/${LABEL}/report.json"
