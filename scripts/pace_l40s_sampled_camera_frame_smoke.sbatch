#!/usr/bin/env bash
#SBATCH --job-name=sampled-camera-smoke
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

mkdir -p logs results overlays

SOURCE_INPUT="${SOURCE_INPUT:-videos/test1.mov}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID}}"
SAM_MAX_FRAMES="${SAM_MAX_FRAMES:-1}"

export SOURCE_INPUT RUN_ID SAM_MAX_FRAMES

bash scripts/run_sampled_camera_frame_smoke.sh "${SOURCE_INPUT}"
