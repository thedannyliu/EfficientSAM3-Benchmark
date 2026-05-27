#!/usr/bin/env bash
#SBATCH --job-name=sam-coco-suite
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load python/3.12.5 cuda/12.6.1
source .venv/bin/activate

mkdir -p logs results overlays data/manifests

sam-env-probe

if [[ ! -f data/manifests/coco_val2017_fixed10.jsonl ]]; then
  bash scripts/prepare_coco_fixed_subset.sh 10
fi

if [[ ! -f checkpoints/sam3/sam3.pt ]]; then
  bash scripts/download_sam3_checkpoint.sh
fi

python -m sam_backend.coco_suite \
  --manifest data/manifests/coco_val2017_fixed10.jsonl \
  --device cuda \
  --skip-missing \
  --output-dir "results/coco/suite/${SLURM_JOB_ID}" \
  --overlay-dir "overlays/coco/suite/${SLURM_JOB_ID}"
