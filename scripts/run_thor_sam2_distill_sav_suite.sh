#!/usr/bin/env bash
set -euo pipefail

SAM2D_PIPELINE="${SAM2D_PIPELINE:-/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/SAM2-Distillation-Pipeline}"
SAV_ROOT="${SAV_ROOT:-data/sa-v/sav_test}"
SAV_SPLIT_FILE="${SAV_SPLIT_FILE:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-results/thor/offline/sav_test_sam2_distill/${RUN_ID}}"
PREP_ROOT="${PREP_ROOT:-${OUT_ROOT}/prepared_sav_test_links}"
STAGE="${1:-all}"

PROJECT_ROOT="$(pwd)"
SAM2_DISTILL_CHECKPOINT_ROOT="${SAM2_DISTILL_CHECKPOINT_ROOT:-checkpoints/sam2_distill}"
SAM2_STAGE1_ROOT="${SAM2_STAGE1_ROOT:-${SAM2_DISTILL_CHECKPOINT_ROOT}/stage1}"
SAM2_TINYVIT_ROOT="${SAM2_TINYVIT_ROOT:-${SAM2_DISTILL_CHECKPOINT_ROOT}/tinyvit}"

SAM2_ROOT="${SAM2_ROOT:-external/sam2}"
EDGETAM_ROOT="${EDGETAM_ROOT:-external/EdgeTAM}"
SAM2L_CKPT="${SAM2L_CKPT:-checkpoints/sam2/sam2.1_hiera_large.pt}"
SAM2B_CKPT="${SAM2B_CKPT:-checkpoints/sam2/sam2.1_hiera_base_plus.pt}"
EDGETAM_CHECKPOINT="${EDGETAM_CHECKPOINT:-checkpoints/edgetam/edgetam.pt}"

TINYVIT21_CKPT="${TINYVIT21_CKPT:-${SAM2_TINYVIT_ROOT}/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"
TINYVIT11_CKPT="${TINYVIT11_CKPT:-${SAM2_TINYVIT_ROOT}/tiny_vit_11m_224.dist_in22k_ft_in1k.safetensors}"
TINYVIT5_CKPT="${TINYVIT5_CKPT:-${SAM2_TINYVIT_ROOT}/tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors}"

TV21_MSE="${TV21_MSE:-${SAM2_STAGE1_ROOT}/tv21m_mse.pt}"
TV21_MSE_COS="${TV21_MSE_COS:-${SAM2_STAGE1_ROOT}/tv21m_mse_cos.pt}"
TV21_HIGHRES="${TV21_HIGHRES:-${SAM2_STAGE1_ROOT}/tv21m_highres.pt}"
TV11_MSE="${TV11_MSE:-${SAM2_STAGE1_ROOT}/tv11m_mse.pt}"
TV11_MSE_COS="${TV11_MSE_COS:-${SAM2_STAGE1_ROOT}/tv11m_mse_cos.pt}"
TV5_MSE="${TV5_MSE:-${SAM2_STAGE1_ROOT}/tv5m_mse.pt}"
TV5_MSE_COS="${TV5_MSE_COS:-${SAM2_STAGE1_ROOT}/tv5m_mse_cos.pt}"

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "${PROJECT_ROOT}/$1" ;;
  esac
}

SAV_ROOT="$(abs_path "${SAV_ROOT}")"
OUT_ROOT="$(abs_path "${OUT_ROOT}")"
PREP_ROOT="$(abs_path "${PREP_ROOT}")"
SAM2_DISTILL_CHECKPOINT_ROOT="$(abs_path "${SAM2_DISTILL_CHECKPOINT_ROOT}")"
SAM2_STAGE1_ROOT="$(abs_path "${SAM2_STAGE1_ROOT}")"
SAM2_TINYVIT_ROOT="$(abs_path "${SAM2_TINYVIT_ROOT}")"
SAM2_ROOT="$(abs_path "${SAM2_ROOT}")"
EDGETAM_ROOT="$(abs_path "${EDGETAM_ROOT}")"
SAM2L_CKPT="$(abs_path "${SAM2L_CKPT}")"
SAM2B_CKPT="$(abs_path "${SAM2B_CKPT}")"
EDGETAM_CHECKPOINT="$(abs_path "${EDGETAM_CHECKPOINT}")"
TINYVIT21_CKPT="$(abs_path "${TINYVIT21_CKPT}")"
TINYVIT11_CKPT="$(abs_path "${TINYVIT11_CKPT}")"
TINYVIT5_CKPT="$(abs_path "${TINYVIT5_CKPT}")"
TV21_MSE="$(abs_path "${TV21_MSE}")"
TV21_MSE_COS="$(abs_path "${TV21_MSE_COS}")"
TV21_HIGHRES="$(abs_path "${TV21_HIGHRES}")"
TV11_MSE="$(abs_path "${TV11_MSE}")"
TV11_MSE_COS="$(abs_path "${TV11_MSE_COS}")"
TV5_MSE="$(abs_path "${TV5_MSE}")"
TV5_MSE_COS="$(abs_path "${TV5_MSE_COS}")"

MAX_VIDEOS="${MAX_VIDEOS:-0}"
MAX_IMAGE_OBJECTS="${MAX_IMAGE_OBJECTS:-0}"
IMAGE_ARTIFACT_VIDEOS="${IMAGE_ARTIFACT_VIDEOS:-3}"
VOS_OVERLAY_VIDEOS="${VOS_OVERLAY_VIDEOS:-3}"
VOS_OVERLAY_FRAMES="${VOS_OVERLAY_FRAMES:-0}"
NUM_EVAL_PROCESSES="${NUM_EVAL_PROCESSES:-4}"
DEVICE="${DEVICE:-cuda}"
SKIP_MISSING="${SKIP_MISSING:-1}"
SKIP_DONE="${SKIP_DONE:-1}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_thor_sam2_distill_sav_suite.sh prepare-links
  scripts/run_thor_sam2_distill_sav_suite.sh sam2
  scripts/run_thor_sam2_distill_sav_suite.sh edgetam
  scripts/run_thor_sam2_distill_sav_suite.sh all

Runs the SAM2-Distillation-Pipeline SA-V image/VOS suites on an existing SA-V
val/test-style root. The wrapper creates a lightweight prepared directory with
symlinks and a sav_train_benchmark.txt file because the upstream scripts use
that filename for both raw-shard and already-prepared layouts.
EOF
}

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "${path}" ]]; then
    echo "missing ${label}: ${path}" >&2
    exit 1
  fi
}

prepare_links() {
  require_path "SAM2-Distillation-Pipeline" "${SAM2D_PIPELINE}"
  require_path "SA-V root" "${SAV_ROOT}"
  require_path "SA-V frames" "${SAV_ROOT}/JPEGImages_24fps"
  require_path "SA-V annotations" "${SAV_ROOT}/Annotations_6fps"

  mkdir -p "${PREP_ROOT}"
  if [[ "${MAX_VIDEOS}" == "0" && -z "${SAV_SPLIT_FILE}" ]]; then
    ln -sfn "$(cd "${SAV_ROOT}" && pwd)/JPEGImages_24fps" "${PREP_ROOT}/JPEGImages_24fps"
    ln -sfn "$(cd "${SAV_ROOT}" && pwd)/Annotations_6fps" "${PREP_ROOT}/Annotations_6fps"
    find "${SAV_ROOT}/JPEGImages_24fps" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
      | sort > "${PREP_ROOT}/sav_train_benchmark.txt"
  else
    mkdir -p "${PREP_ROOT}/JPEGImages_24fps" "${PREP_ROOT}/Annotations_6fps"
    if [[ -n "${SAV_SPLIT_FILE}" ]]; then
      require_path "SA-V split file" "${SAV_SPLIT_FILE}"
      mapfile -t selected_videos < <(sed '/^[[:space:]]*$/d' "${SAV_SPLIT_FILE}")
    else
      mapfile -t selected_videos < <(find "${SAV_ROOT}/JPEGImages_24fps" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
    fi
    if [[ "${MAX_VIDEOS}" != "0" ]]; then
      selected_videos=("${selected_videos[@]:0:${MAX_VIDEOS}}")
    fi
    : > "${PREP_ROOT}/sav_train_benchmark.txt"
    for video_id in "${selected_videos[@]}"; do
      [[ -n "${video_id}" ]] || continue
      require_path "SA-V video frames ${video_id}" "${SAV_ROOT}/JPEGImages_24fps/${video_id}"
      require_path "SA-V video annotations ${video_id}" "${SAV_ROOT}/Annotations_6fps/${video_id}"
      ln -sfn "$(cd "${SAV_ROOT}/JPEGImages_24fps/${video_id}" && pwd)" "${PREP_ROOT}/JPEGImages_24fps/${video_id}"
      ln -sfn "$(cd "${SAV_ROOT}/Annotations_6fps/${video_id}" && pwd)" "${PREP_ROOT}/Annotations_6fps/${video_id}"
      printf '%s\n' "${video_id}" >> "${PREP_ROOT}/sav_train_benchmark.txt"
    done
  fi
  echo "${PREP_ROOT}"
}

run_sam2_suite() {
  prepare_links
  (
    cd "${SAM2D_PIPELINE}"
    PREP_ROOT="${PREP_ROOT}" \
    OUT_ROOT="${OUT_ROOT}/sam2_stage1" \
    SAM2_ROOT="${SAM2_ROOT}" \
    SAM2L_CKPT="${SAM2L_CKPT}" \
    SAM2B_CKPT="${SAM2B_CKPT}" \
    TINYVIT21_CKPT="${TINYVIT21_CKPT}" \
    TINYVIT11_CKPT="${TINYVIT11_CKPT}" \
    TINYVIT5_CKPT="${TINYVIT5_CKPT}" \
    TV21_MSE="${TV21_MSE}" \
    TV21_MSE_COS="${TV21_MSE_COS}" \
    TV21_HIGHRES="${TV21_HIGHRES}" \
    TV11_MSE="${TV11_MSE}" \
    TV11_MSE_COS="${TV11_MSE_COS}" \
    TV5_MSE="${TV5_MSE}" \
    TV5_MSE_COS="${TV5_MSE_COS}" \
    MAX_VIDEOS="${MAX_VIDEOS}" \
    MAX_IMAGE_OBJECTS="${MAX_IMAGE_OBJECTS}" \
    IMAGE_ARTIFACT_VIDEOS="${IMAGE_ARTIFACT_VIDEOS}" \
    VOS_OVERLAY_VIDEOS="${VOS_OVERLAY_VIDEOS}" \
    VOS_OVERLAY_FRAMES="${VOS_OVERLAY_FRAMES}" \
    NUM_EVAL_PROCESSES="${NUM_EVAL_PROCESSES}" \
    DEVICE="${DEVICE}" \
    SKIP_MISSING="${SKIP_MISSING}" \
    SKIP_DONE="${SKIP_DONE}" \
    scripts/company/15_benchmark_raw_sav_shard_suite.sh image

    PREP_ROOT="${PREP_ROOT}" \
    OUT_ROOT="${OUT_ROOT}/sam2_stage1" \
    SAM2_ROOT="${SAM2_ROOT}" \
    SAM2L_CKPT="${SAM2L_CKPT}" \
    SAM2B_CKPT="${SAM2B_CKPT}" \
    TINYVIT21_CKPT="${TINYVIT21_CKPT}" \
    TINYVIT11_CKPT="${TINYVIT11_CKPT}" \
    TINYVIT5_CKPT="${TINYVIT5_CKPT}" \
    TV21_MSE="${TV21_MSE}" \
    TV21_MSE_COS="${TV21_MSE_COS}" \
    TV21_HIGHRES="${TV21_HIGHRES}" \
    TV11_MSE="${TV11_MSE}" \
    TV11_MSE_COS="${TV11_MSE_COS}" \
    TV5_MSE="${TV5_MSE}" \
    TV5_MSE_COS="${TV5_MSE_COS}" \
    MAX_VIDEOS="${MAX_VIDEOS}" \
    MAX_IMAGE_OBJECTS="${MAX_IMAGE_OBJECTS}" \
    IMAGE_ARTIFACT_VIDEOS="${IMAGE_ARTIFACT_VIDEOS}" \
    VOS_OVERLAY_VIDEOS="${VOS_OVERLAY_VIDEOS}" \
    VOS_OVERLAY_FRAMES="${VOS_OVERLAY_FRAMES}" \
    NUM_EVAL_PROCESSES="${NUM_EVAL_PROCESSES}" \
    DEVICE="${DEVICE}" \
    SKIP_MISSING="${SKIP_MISSING}" \
    SKIP_DONE="${SKIP_DONE}" \
    scripts/company/15_benchmark_raw_sav_shard_suite.sh vos

    PREP_ROOT="${PREP_ROOT}" OUT_ROOT="${OUT_ROOT}/sam2_stage1" \
      VOS_OVERLAY_VIDEOS="${VOS_OVERLAY_VIDEOS}" VOS_OVERLAY_FRAMES="${VOS_OVERLAY_FRAMES}" \
      scripts/company/15_benchmark_raw_sav_shard_suite.sh artifacts
    OUT_ROOT="${OUT_ROOT}/sam2_stage1" scripts/company/15_benchmark_raw_sav_shard_suite.sh summarize
  )
}

run_edgetam_suite() {
  prepare_links
  (
    cd "${SAM2D_PIPELINE}"
    PREP_ROOT="${PREP_ROOT}" \
    OUT_ROOT="${OUT_ROOT}/edgetam" \
    SAM2_ROOT="${SAM2_ROOT}" \
    EDGETAM_ROOT="${EDGETAM_ROOT}" \
    EDGETAM_CHECKPOINT="${EDGETAM_CHECKPOINT}" \
    TV21_MSE_COS="${TV21_MSE_COS}" \
    MAX_VIDEOS="${MAX_VIDEOS}" \
    MAX_IMAGE_OBJECTS="${MAX_IMAGE_OBJECTS}" \
    IMAGE_ARTIFACT_VIDEOS="${IMAGE_ARTIFACT_VIDEOS}" \
    VOS_OVERLAY_VIDEOS="${VOS_OVERLAY_VIDEOS}" \
    VOS_OVERLAY_FRAMES="${VOS_OVERLAY_FRAMES}" \
    NUM_EVAL_PROCESSES="${NUM_EVAL_PROCESSES}" \
    DEVICE="${DEVICE}" \
    SKIP_DONE="${SKIP_DONE}" \
    scripts/company/16_benchmark_edgetam_bridge_raw_sav.sh image

    PREP_ROOT="${PREP_ROOT}" \
    OUT_ROOT="${OUT_ROOT}/edgetam" \
    SAM2_ROOT="${SAM2_ROOT}" \
    EDGETAM_ROOT="${EDGETAM_ROOT}" \
    EDGETAM_CHECKPOINT="${EDGETAM_CHECKPOINT}" \
    TV21_MSE_COS="${TV21_MSE_COS}" \
    MAX_VIDEOS="${MAX_VIDEOS}" \
    MAX_IMAGE_OBJECTS="${MAX_IMAGE_OBJECTS}" \
    IMAGE_ARTIFACT_VIDEOS="${IMAGE_ARTIFACT_VIDEOS}" \
    VOS_OVERLAY_VIDEOS="${VOS_OVERLAY_VIDEOS}" \
    VOS_OVERLAY_FRAMES="${VOS_OVERLAY_FRAMES}" \
    NUM_EVAL_PROCESSES="${NUM_EVAL_PROCESSES}" \
    DEVICE="${DEVICE}" \
    SKIP_DONE="${SKIP_DONE}" \
    scripts/company/16_benchmark_edgetam_bridge_raw_sav.sh vos

    PREP_ROOT="${PREP_ROOT}" OUT_ROOT="${OUT_ROOT}/edgetam" \
      VOS_OVERLAY_VIDEOS="${VOS_OVERLAY_VIDEOS}" VOS_OVERLAY_FRAMES="${VOS_OVERLAY_FRAMES}" \
      scripts/company/16_benchmark_edgetam_bridge_raw_sav.sh artifacts
    OUT_ROOT="${OUT_ROOT}/edgetam" scripts/company/16_benchmark_edgetam_bridge_raw_sav.sh summarize
  )
}

case "${STAGE}" in
  prepare-links) prepare_links ;;
  sam2) run_sam2_suite ;;
  edgetam) run_edgetam_suite ;;
  all)
    run_sam2_suite
    run_edgetam_suite
    ;;
  -h|--help|"") usage ;;
  *) usage; exit 2 ;;
esac
