#!/usr/bin/env bash
#SBATCH --job-name=tv-trt-pipe
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

module load python/3.12.5 cuda/12.6.1
source .venv/bin/activate

MODEL="${MODEL:-tv5}"
: "${ENGINE_PATH:?set ENGINE_PATH to the TensorRT encoder engine}"

LABEL=""
OPTIONS=()
case "${SLURM_ARRAY_TASK_ID}" in
  0) LABEL="baseline" ;;
  1) LABEL="shape-only"; OPTIONS+=(--trt-position-mode shape-only) ;;
  2) LABEL="binary-only"; OPTIONS+=(--mask-transfer binary-only) ;;
  3) LABEL="gpu-preprocess"; OPTIONS+=(--trt-gpu-preprocess) ;;
  4) LABEL="decoder-fp16"; OPTIONS+=(--decoder-autocast-fp16 --trt-native-outputs) ;;
  5) LABEL="compile"; OPTIONS+=(--compile-components reduce-overhead) ;;
  6) LABEL="shape-binary"; OPTIONS+=(--trt-position-mode shape-only --mask-transfer binary-only) ;;
  7) LABEL="shape-gpu-binary"; OPTIONS+=(--trt-position-mode shape-only --trt-gpu-preprocess --mask-transfer binary-only) ;;
  8) LABEL="all-no-compile"; OPTIONS+=(--trt-position-mode shape-only --trt-gpu-preprocess --mask-transfer binary-only --decoder-autocast-fp16 --trt-native-outputs) ;;
  9) LABEL="all-compile"; OPTIONS+=(--trt-position-mode shape-only --trt-gpu-preprocess --mask-transfer binary-only --decoder-autocast-fp16 --trt-native-outputs --compile-components reduce-overhead) ;;
  12) LABEL="safe-compile"; OPTIONS+=(--trt-position-mode shape-only --trt-gpu-preprocess --mask-transfer binary-only --compile-components reduce-overhead) ;;
  13) LABEL="safe-compile-max"; OPTIONS+=(--trt-position-mode shape-only --trt-gpu-preprocess --mask-transfer binary-only --compile-components max-autotune) ;;
  14) LABEL="all-compile-max"; OPTIONS+=(--trt-position-mode shape-only --trt-gpu-preprocess --mask-transfer binary-only --decoder-autocast-fp16 --trt-native-outputs --compile-components max-autotune) ;;
  15) LABEL="all-compile-pre-fp16"; OPTIONS+=(--trt-position-mode shape-only --trt-gpu-preprocess --trt-gpu-preprocess-fp16 --mask-transfer binary-only --decoder-autocast-fp16 --trt-native-outputs --compile-components reduce-overhead) ;;
  16) LABEL="all-compile-max-pre-fp16"; OPTIONS+=(--trt-position-mode shape-only --trt-gpu-preprocess --trt-gpu-preprocess-fp16 --mask-transfer binary-only --decoder-autocast-fp16 --trt-native-outputs --compile-components max-autotune) ;;
  17) LABEL="all-compile-cache"; OPTIONS+=(--trt-position-mode shape-only --trt-gpu-preprocess --mask-transfer binary-only --decoder-autocast-fp16 --trt-native-outputs --compile-components reduce-overhead --cache-prompts) ;;
  18) LABEL="all-compile-pre-fp16-cache"; OPTIONS+=(--trt-position-mode shape-only --trt-gpu-preprocess --trt-gpu-preprocess-fp16 --mask-transfer binary-only --decoder-autocast-fp16 --trt-native-outputs --compile-components reduce-overhead --cache-prompts) ;;
  19) LABEL="all-compile-max-pre-fp16-cache"; OPTIONS+=(--trt-position-mode shape-only --trt-gpu-preprocess --trt-gpu-preprocess-fp16 --mask-transfer binary-only --decoder-autocast-fp16 --trt-native-outputs --compile-components max-autotune --cache-prompts) ;;
  *) echo "unsupported array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

python scripts/pace_tinyvit_trt_mask_parity.py \
  --student-checkpoint "checkpoints/distill/${MODEL}.pt" \
  --engine "${ENGINE_PATH}" \
  --sam2-checkpoint checkpoints/sam2/sam2.1_hiera_large.pt \
  --sam2-root external/sam2 \
  --distill-root /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/SAM2-Distillation-Pipeline \
  --video "${VIDEO:-videos/test1.mov}" \
  --frames "${FRAMES:-16}" \
  --warmup "${WARMUP:-3}" \
  --sequential-frames \
  --decode-warmup "${DECODE_WARMUP:-0}" \
  --stream-throughput-frames "${STREAM_THROUGHPUT_FRAMES:-0}" \
  --minimum-mask-iou 0.0 \
  --minimum-mean-mask-iou "${MINIMUM_MEAN_MASK_IOU:-0.95}" \
  --output "results/pace/tinyvit_trt_pipeline_sweep/${SLURM_ARRAY_JOB_ID}/${LABEL}/report.json" \
  "${OPTIONS[@]}"
