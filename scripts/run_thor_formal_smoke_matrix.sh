#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/thor/formal_smoke/${RUN_ID}}"
OVERLAY_ROOT="${OVERLAY_ROOT:-overlays/thor/formal_smoke/${RUN_ID}}"
SAM2D_PIPELINE="${SAM2D_PIPELINE:-/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/SAM2-Distillation-Pipeline}"
SAV_ROOT="${SAV_ROOT:-data/sa-v/sav_test}"
SA1B_ROOT="${SA1B_ROOT:-data/sa1b/extracted_two_tar}"
SA1B_IMAGE_ROOT="${SA1B_IMAGE_ROOT:-${SA1B_ROOT}}"
DEVICE="${DEVICE:-cuda}"
SKIP_MISSING="${SKIP_MISSING:-1}"
SKIP_DONE="${SKIP_DONE:-1}"
NUM_EVAL_PROCESSES="${NUM_EVAL_PROCESSES:-4}"
SAV_MAX_FRAMES="${SAV_MAX_FRAMES:-120}"
MAX_IMAGE_OBJECTS="${MAX_IMAGE_OBJECTS:-20}"
SA1B_MODELS="${SA1B_MODELS:-sam3 es3p1_weak_image_weak_text es3p1_strong_image_weak_text es3_weak_image_strong_available_text es3_strong_image_strong_available_text instinctsam_vitb sam2p1_hiera_tiny sam2p1_hiera_small sam2p1_hiera_base_plus sam2p1_hiera_large efficient_sam2p1_hiera_tiny efficient_sam2p1_hiera_small efficient_sam2p1_hiera_base_plus efficient_sam2p1_hiera_large official_edgetam efficienttam_ti efficienttam_s mobilesam_vit_t mobilesam_vit_b mobilesam_vit_l mobilesam_vit_h sam1_vit_h}"

case "${OUTPUT_ROOT}" in
  /*) ;;
  *) OUTPUT_ROOT="$(pwd)/${OUTPUT_ROOT}" ;;
esac
case "${OVERLAY_ROOT}" in
  /*) ;;
  *) OVERLAY_ROOT="$(pwd)/${OVERLAY_ROOT}" ;;
esac
case "${SAV_ROOT}" in
  /*) ;;
  *) SAV_ROOT="$(pwd)/${SAV_ROOT}" ;;
esac
case "${SA1B_ROOT}" in
  /*) ;;
  *) SA1B_ROOT="$(pwd)/${SA1B_ROOT}" ;;
esac
case "${SA1B_IMAGE_ROOT}" in
  /*) ;;
  *) SA1B_IMAGE_ROOT="$(pwd)/${SA1B_IMAGE_ROOT}" ;;
esac

mkdir -p "${OUTPUT_ROOT}/manifests" "${OUTPUT_ROOT}/prepared" "${OVERLAY_ROOT}"

echo "RUN_ID=${RUN_ID}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "OVERLAY_ROOT=${OVERLAY_ROOT}"
echo "SAV_ROOT=${SAV_ROOT}"
echo "SA1B_ROOT=${SA1B_ROOT}"
echo "SA1B_IMAGE_ROOT=${SA1B_IMAGE_ROOT}"
echo "SAM2D_PIPELINE=${SAM2D_PIPELINE}"

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "${path}" ]]; then
    echo "missing ${label}: ${path}" >&2
    exit 1
  fi
}

maybe_run_efficienttam_sav() {
  local model_id="$1"
  local checkpoint="$2"
  local config="$3"
  local manifest="$4"
  local out_dir="${OUTPUT_ROOT}/sav_efficienttam/${model_id}"
  local overlay_dir="${OVERLAY_ROOT}/sav_efficienttam/${model_id}"

  if [[ ! -f "${checkpoint}" || ! -d external/EfficientTAM ]]; then
    if [[ "${SKIP_MISSING}" == "1" ]]; then
      echo "skip ${model_id} SA-V: missing ${checkpoint} or external/EfficientTAM" >&2
      return 0
    fi
    require_path "${model_id} checkpoint" "${checkpoint}"
    require_path "EfficientTAM repo" "external/EfficientTAM"
  fi

  python -m sam_backend.profile_sav_video \
    --model-id "${model_id}" \
    --backend efficienttam \
    --external-repo external/EfficientTAM \
    --checkpoint-path "${checkpoint}" \
    --model-config "${config}" \
    --device "${DEVICE}" \
    --manifest "${manifest}" \
    --eval-mode both \
    --max-frames "${SAV_MAX_FRAMES}" \
    --autocast-bfloat16 \
    --csv-output "${out_dir}/frames.csv" \
    --summary-output "${out_dir}/summary.json" \
    --overlay-root "${overlay_dir}"
}

run_sav_smoke() {
  require_path "SA-V root" "${SAV_ROOT}"
  require_path "SAM2-Distillation-Pipeline" "${SAM2D_PIPELINE}"

  MAX_VIDEOS=1 \
  MAX_IMAGE_OBJECTS="${MAX_IMAGE_OBJECTS}" \
  OUT_ROOT="${OUTPUT_ROOT}/sav_sam2d" \
  PREP_ROOT="${OUTPUT_ROOT}/prepared/sav_one_video" \
  SAM2D_PIPELINE="${SAM2D_PIPELINE}" \
  SAV_ROOT="${SAV_ROOT}" \
  DEVICE="${DEVICE}" \
  SKIP_MISSING="${SKIP_MISSING}" \
  SKIP_DONE="${SKIP_DONE}" \
  NUM_EVAL_PROCESSES="${NUM_EVAL_PROCESSES}" \
  bash scripts/run_thor_sam2_distill_sav_suite.sh all

  local sav_manifest="${OUTPUT_ROOT}/manifests/sav_one_video.jsonl"
  python -m sam_backend.sav_manifest \
    --sav-root "${SAV_ROOT}" \
    --count 1 \
    --output "${sav_manifest}"

  maybe_run_efficienttam_sav efficienttam_ti checkpoints/efficienttam/efficienttam_ti.pt configs/efficienttam/efficienttam_ti.yaml "${sav_manifest}"
  maybe_run_efficienttam_sav efficienttam_s checkpoints/efficienttam/efficienttam_s.pt configs/efficienttam/efficienttam_s.yaml "${sav_manifest}"

  if compgen -G "${OUTPUT_ROOT}/sav_efficienttam/*/frames_summary.csv" >/dev/null; then
    python -m sam_backend.sav_video_report \
      --root "${OUTPUT_ROOT}/sav_efficienttam" \
      --output "${OUTPUT_ROOT}/sav_efficienttam/sav_video_suite_summary.csv"
  fi
}

run_sa1b_smoke() {
  require_path "SA1B annotation root" "${SA1B_ROOT}"
  require_path "SAM2-Distillation-Pipeline" "${SAM2D_PIPELINE}"

  local sa1b_manifest="${OUTPUT_ROOT}/manifests/sa1b_one_image.jsonl"
  SA1B_ROOT="${SA1B_ROOT}" \
  SA1B_IMAGE_ROOT="${SA1B_IMAGE_ROOT}" \
  SA1B_COUNT=1 \
  MANIFEST="${sa1b_manifest}" \
  bash scripts/prepare_sa1b_fixed_subset.sh

  python -m sam_backend.manifest_mask_layout \
    --manifest "${sa1b_manifest}" \
    --output-root "${OUTPUT_ROOT}/prepared/sa1b_one_image_mask_layout"

  MODELS="${SA1B_MODELS}" \
  MANIFEST="${sa1b_manifest}" \
  RUN_ID="${RUN_ID}" \
  DEVICE="${DEVICE}" \
  LIMIT=1 \
  EVAL_MODE=both \
  SKIP_MISSING="${SKIP_MISSING}" \
  OUTPUT_DIR="${OUTPUT_ROOT}/sa1b_sam_family" \
  OVERLAY_DIR="${OVERLAY_ROOT}/sa1b_sam_family" \
  bash scripts/run_thor_sa1b_image_benchmarks.sh

  (
    cd "${SAM2D_PIPELINE}"
    PREP_ROOT="${OUTPUT_ROOT}/prepared/sa1b_one_image_mask_layout" \
    OUT_ROOT="${OUTPUT_ROOT}/sa1b_sam2d/sam2_stage1" \
    MAX_IMAGE_OBJECTS=1 \
    IMAGE_ARTIFACT_VIDEOS=1 \
    DEVICE="${DEVICE}" \
    SKIP_MISSING="${SKIP_MISSING}" \
    SKIP_DONE="${SKIP_DONE}" \
    scripts/company/15_benchmark_raw_sav_shard_suite.sh image
    OUT_ROOT="${OUTPUT_ROOT}/sa1b_sam2d/sam2_stage1" \
      scripts/company/15_benchmark_raw_sav_shard_suite.sh summarize
  )

  (
    cd "${SAM2D_PIPELINE}"
    PREP_ROOT="${OUTPUT_ROOT}/prepared/sa1b_one_image_mask_layout" \
    OUT_ROOT="${OUTPUT_ROOT}/sa1b_sam2d/edgetam" \
    MAX_IMAGE_OBJECTS=1 \
    IMAGE_ARTIFACT_VIDEOS=1 \
    DEVICE="${DEVICE}" \
    SKIP_DONE="${SKIP_DONE}" \
    scripts/company/16_benchmark_edgetam_bridge_raw_sav.sh image
    OUT_ROOT="${OUTPUT_ROOT}/sa1b_sam2d/edgetam" \
      scripts/company/16_benchmark_edgetam_bridge_raw_sav.sh summarize
  )
}

run_sav_smoke
run_sa1b_smoke

python -m sam_backend.thor_smoke_summary \
  --root "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/thor_formal_smoke_summary.csv"

echo "summary=${OUTPUT_ROOT}/thor_formal_smoke_summary.csv"
