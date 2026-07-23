#!/usr/bin/env bash
#SBATCH --job-name=tv-trt-final
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

: "${ENGINE_PATH:?set ENGINE_PATH to the TensorRT encoder engine}"
export FRAMES="${FRAMES:-32}"
export WARMUP="${WARMUP:-5}"
export DECODE_WARMUP="${DECODE_WARMUP:-3}"
export STREAM_THROUGHPUT_FRAMES="${STREAM_THROUGHPUT_FRAMES:-64}"

for task in 0 17; do
  SLURM_ARRAY_JOB_ID="${SLURM_JOB_ID}" \
  SLURM_ARRAY_TASK_ID="${task}" \
  bash scripts/pace_l40s_tinyvit_trt_pipeline_sweep.sbatch
done
