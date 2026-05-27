#!/usr/bin/env bash
#SBATCH --job-name=mobilesam-coco
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load python/3.12.5 cuda/12.6.1
source .venv/bin/activate

mkdir -p logs results overlays data/manifests

RUN_ID="${SLURM_JOB_ID:-manual}"

if [[ ! -f data/manifests/coco_val2017_fixed10.jsonl ]]; then
  bash scripts/prepare_coco_fixed_subset.sh 10
fi

if [[ "${DOWNLOAD_YOLOE_EDGETAM_MOBILESAM:-0}" == "1" ]]; then
  bash scripts/download_yoloe_edgetam_mobilesam_assets.sh
fi

python -m sam_backend.profile_coco \
  --backend mobilesam \
  --model-id mobilesam_vit_t \
  --external-repo external/MobileSAM \
  --checkpoint-path checkpoints/mobilesam/mobile_sam.pt \
  --mobile-sam-model-type vit_t \
  --device cuda \
  --manifest data/manifests/coco_val2017_fixed10.jsonl \
  --prompt-mode point \
  --eval-mode both \
  --csv-output "results/coco/mobilesam/${RUN_ID}/profile.csv" \
  --summary-output "results/coco/mobilesam/${RUN_ID}/summary.json" \
  --overlay-dir "overlays/coco/mobilesam/${RUN_ID}"
