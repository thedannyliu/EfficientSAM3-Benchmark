#!/usr/bin/env bash
#SBATCH --job-name=sav-sam2-family
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load python/3.12.5 cuda/12.6.1
source .venv/bin/activate

mkdir -p logs results overlays data/manifests

sam-env-probe

SAV_SPLIT="${SAV_SPLIT:-val}"
SAV_SUBSET_COUNT="${SAV_SUBSET_COUNT:-3}"
SAV_ROOT="${SAV_ROOT:-data/sa-v/sav_${SAV_SPLIT}_fixed${SAV_SUBSET_COUNT}}"
MANIFEST="${SAV_MANIFEST:-data/manifests/sav_${SAV_SPLIT}_fixed${SAV_SUBSET_COUNT}.jsonl}"

if [[ ! -f "${MANIFEST}" ]]; then
  bash scripts/download_sav_valtest_subset.sh "${SAV_SPLIT}" "${SAV_SUBSET_COUNT}" "${SAV_ROOT}"
fi

if [[ "${DOWNLOAD_SAM2_FAMILY_CHECKPOINTS:-0}" == "1" ]]; then
  bash scripts/download_sam2_family_checkpoints.sh
fi

run_video_profile() {
  local model_id="$1"
  local backend="$2"
  local external_repo="$3"
  local model_config="$4"
  local checkpoint="$5"

  if [[ ! -f "${checkpoint}" || ! -d "${external_repo}" ]]; then
    echo "skip ${model_id}: missing checkpoint or external repo"
    return
  fi

  if ! python -m sam_backend.profile_sav_video \
    --model-id "${model_id}" \
    --backend "${backend}" \
    --external-repo "${external_repo}" \
    --model-config "${model_config}" \
    --checkpoint-path "${checkpoint}" \
    --manifest "${MANIFEST}" \
    --device cuda \
    --max-frames "${SAM_MAX_FRAMES:-96}" \
    --autocast-bfloat16 \
    --csv-output "results/sav/video/${SLURM_JOB_ID}/${model_id}/frames.csv" \
    --summary-output "results/sav/video/${SLURM_JOB_ID}/${model_id}/summary.json" \
    --pred-root "results/sav/video/${SLURM_JOB_ID}/${model_id}/pred" \
    --overlay-root "overlays/sav/video/${SLURM_JOB_ID}/${model_id}"; then
    echo "failed ${model_id}; continuing with remaining models" >&2
  fi
}

run_video_profile \
  sam2p1_hiera_tiny \
  sam2 \
  external/sam2 \
  configs/sam2.1/sam2.1_hiera_t.yaml \
  checkpoints/sam2/sam2.1_hiera_tiny.pt

run_video_profile \
  efficient_sam2p1_hiera_tiny \
  efficient-sam2 \
  external/Efficient-SAM2 \
  configs/sam2.1/sam2.1_hiera_t.yaml \
  checkpoints/efficient-sam2/sam2.1_hiera_tiny.pt

run_video_profile \
  efficienttam_ti \
  efficienttam \
  external/EfficientTAM \
  configs/efficienttam/efficienttam_ti.yaml \
  checkpoints/efficienttam/efficienttam_ti.pt

run_video_profile \
  efficienttam_s \
  efficienttam \
  external/EfficientTAM \
  configs/efficienttam/efficienttam_s.yaml \
  checkpoints/efficienttam/efficienttam_s.pt

python -m sam_backend.sav_video_report \
  --root "results/sav/video/${SLURM_JOB_ID}" \
  --output "results/sav/video/${SLURM_JOB_ID}/sav_video_suite_summary.csv"
