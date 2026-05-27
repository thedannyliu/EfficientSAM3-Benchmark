#!/usr/bin/env bash
#SBATCH --job-name=yoloe-edgetam-poc
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load python/3.12.5 cuda/12.6.1
source .venv/bin/activate

mkdir -p logs results overlays

TEXT_PROMPT="${TEXT_PROMPT:-monitor}"
YOLOE_SOURCE="${YOLOE_SOURCE:-videos/test1.mov}"
RUN_ID="${SLURM_JOB_ID:-manual}"

if [[ "${DOWNLOAD_YOLOE_EDGETAM_MOBILESAM:-0}" == "1" ]]; then
  bash scripts/download_yoloe_edgetam_mobilesam_assets.sh
fi

python -m sam_backend.profile_yoloe_edgetam \
  --video-path "${YOLOE_SOURCE}" \
  --source-id "$(basename "${YOLOE_SOURCE%.*}")" \
  --text-prompt "${TEXT_PROMPT}" \
  --device cuda \
  --max-frames "${MAX_FRAMES:-240}" \
  --yoloe-interval "${YOLOE_INTERVAL:-20}" \
  --autocast-bfloat16 \
  --csv-output "results/yoloe_edgetam/${RUN_ID}/frames.csv" \
  --summary-output "results/yoloe_edgetam/${RUN_ID}/summary.json" \
  --overlay-root "overlays/yoloe_edgetam/${RUN_ID}" \
  --work-dir "results/yoloe_edgetam/${RUN_ID}/work"
