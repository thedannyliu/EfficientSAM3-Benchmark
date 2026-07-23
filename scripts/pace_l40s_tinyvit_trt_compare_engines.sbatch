#!/usr/bin/env bash
#SBATCH --job-name=tv-trt-compare
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

: "${ENGINE_ARGUMENTS:?set ENGINE_ARGUMENTS to space-separated LABEL=PATH entries}"

module load python/3.12.5 cuda/12.6.1
source .venv/bin/activate

arguments=()
for engine in ${ENGINE_ARGUMENTS}; do
  arguments+=(--engine "${engine}")
done

python scripts/pace_tinyvit_trt_compare_engines.py \
  "${arguments[@]}" \
  --warmup "${WARMUP:-50}" \
  --rounds "${ROUNDS:-200}" \
  --output "${OUTPUT:-results/pace/tinyvit_trt_compare/${SLURM_JOB_ID}/report.json}"
