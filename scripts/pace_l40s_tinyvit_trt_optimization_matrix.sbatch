#!/usr/bin/env bash
#SBATCH --job-name=tv-trt-opt
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-18
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

module load python/3.12.5 cuda/12.6.1
source .venv/bin/activate

MATRIX=(
  "tv5 fp32 dynamo 0 5"
  "tv5 fp32 dynamo 1 5"
  "tv5 fp16 dynamo 0 5"
  "tv5 bf16 dynamo 0 5"
  "tv11 fp32 dynamo 0 5"
  "tv11 fp32 dynamo 1 5"
  "tv11 fp16 dynamo 0 5"
  "tv11 bf16 dynamo 0 5"
  "tv21 fp32 dynamo 0 5"
  "tv21 fp32 dynamo 1 5"
  "tv21 fp16 dynamo 0 5"
  "tv21 bf16 dynamo 0 5"
  "tv21 fp32 legacy 0 5"
  "tv21 fp16 legacy 0 5"
  "tv5 fp16 legacy 0 5"
  "tv11 fp16 legacy 0 5"
  "tv5 fp32 legacy 0 5"
  "tv11 fp32 legacy 0 5"
  "tv21 fp32 legacy 1 5"
)
read -r MODEL PRECISION EXPORTER TF32 OPTIMIZATION_LEVEL <<< "${MATRIX[${SLURM_ARRAY_TASK_ID}]}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/distill}"
DISTILL_ROOT="${DISTILL_ROOT:-/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/SAM2-Distillation-Pipeline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/pace/tinyvit_trt_optimization/${SLURM_ARRAY_JOB_ID}}"
VARIANT="${PRECISION}-${EXPORTER}-tf32${TF32}-opt${OPTIMIZATION_LEVEL}"
TF32_ARGS=()
if [[ "${TF32}" == "1" ]]; then
  TF32_ARGS+=(--allow-tf32)
fi

python scripts/pace_tinyvit_trt_encoder_smoke.py \
  --checkpoint "${CHECKPOINT_ROOT}/${MODEL}.pt" \
  --distill-root "${DISTILL_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/${MODEL}/${VARIANT}" \
  --precision "${PRECISION}" \
  --exporter "${EXPORTER}" \
  --builder-optimization-level "${OPTIMIZATION_LEVEL}" \
  --workspace-gib "${WORKSPACE_GIB:-8}" \
  --warmup "${WARMUP:-20}" \
  --runs "${RUNS:-100}" \
  "${TF32_ARGS[@]}"
