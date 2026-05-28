#!/usr/bin/env bash
set -euo pipefail

COCO_COUNT="${COCO_COUNT:-10}"
COCO_DIR="${COCO_DIR:-data/coco}"
MANIFEST="${MANIFEST:-data/manifests/coco_val2017_fixed${COCO_COUNT}.jsonl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-1}"
EVAL_MODE="${EVAL_MODE:-both}"
YOLO_PRESET="${YOLO_PRESET:-all}"
PREPARE_COCO="${PREPARE_COCO:-0}"
DOWNLOAD_SAM="${DOWNLOAD_SAM:-0}"
DOWNLOAD_YOLO="${DOWNLOAD_YOLO:-0}"
SKIP_MISSING="${SKIP_MISSING:-1}"
DRY_RUN="${DRY_RUN:-0}"
SAM_MODELS="${SAM_MODELS:-}"
YOLO_MODELS="${YOLO_MODELS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/thor/offline/coco_all/${RUN_ID}}"
OVERLAY_ROOT="${OVERLAY_ROOT:-overlays/thor/offline/coco_all/${RUN_ID}}"

SAM_OUTPUT_DIR="${OUTPUT_ROOT}/sam"
YOLO_OUTPUT_DIR="${OUTPUT_ROOT}/yolo"
SAM_OVERLAY_DIR="${OVERLAY_ROOT}/sam"
YOLO_OVERLAY_DIR="${OVERLAY_ROOT}/yolo"

if [[ "${PREPARE_COCO}" == "1" ]]; then
  bash scripts/prepare_coco_fixed_subset.sh "${COCO_COUNT}" "${COCO_DIR}"
fi

if [[ "${DOWNLOAD_SAM}" == "1" ]]; then
  bash scripts/download_sam3_checkpoint.sh
  bash scripts/download_efficientsam3_checkpoints.sh
  bash scripts/download_sam2_family_checkpoints.sh
  bash scripts/download_yoloe_edgetam_mobilesam_assets.sh
fi

YOLO_ARGS=()
YOLO_MODEL_ARGS=()
if [[ "${DOWNLOAD_YOLO}" == "1" ]]; then
  YOLO_ARGS+=(DOWNLOAD_WEIGHTS=1)
fi
if [[ -n "${YOLO_MODELS}" ]]; then
  read -r -a YOLO_MODEL_IDS <<< "${YOLO_MODELS}"
  YOLO_MODEL_ARGS+=(--models "${YOLO_MODEL_IDS[@]}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  YOLO_MODEL_ARGS+=(--dry-run)
fi

env "${YOLO_ARGS[@]}" \
  MANIFEST="${MANIFEST}" \
  RUN_ID="${RUN_ID}" \
  YOLO_PRESET="${YOLO_PRESET}" \
  LIMIT="${LIMIT}" \
  DEVICE="${DEVICE}" \
  EVAL_MODE="${EVAL_MODE}" \
  OUTPUT_DIR="${YOLO_OUTPUT_DIR}" \
  OVERLAY_DIR="${YOLO_OVERLAY_DIR}" \
  bash scripts/run_thor_yolo_coco_suite.sh "${YOLO_MODEL_ARGS[@]}"

SAM_SKIP_ARGS=()
if [[ "${SKIP_MISSING}" == "1" ]]; then
  SAM_SKIP_ARGS+=(--skip-missing)
fi
SAM_MODEL_ARGS=()
if [[ -n "${SAM_MODELS}" ]]; then
  read -r -a SAM_MODEL_IDS <<< "${SAM_MODELS}"
  SAM_MODEL_ARGS+=(--models "${SAM_MODEL_IDS[@]}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  SAM_MODEL_ARGS+=(--dry-run)
fi

python -m sam_backend.coco_suite \
  --manifest "${MANIFEST}" \
  --device "${DEVICE}" \
  --limit "${LIMIT}" \
  --eval-mode "${EVAL_MODE}" \
  --output-dir "${SAM_OUTPUT_DIR}" \
  --overlay-dir "${SAM_OVERLAY_DIR}" \
  "${SAM_MODEL_ARGS[@]}" \
  "${SAM_SKIP_ARGS[@]}"

python -m sam_backend.coco_all_summary \
  --sam-dir "${SAM_OUTPUT_DIR}" \
  --yolo-dir "${YOLO_OUTPUT_DIR}" \
  --output "${OUTPUT_ROOT}/coco_all_model_summary.csv"

echo "${OUTPUT_ROOT}/coco_all_model_summary.csv"
