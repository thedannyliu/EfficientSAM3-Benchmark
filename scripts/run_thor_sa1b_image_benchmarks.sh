#!/usr/bin/env bash
set -euo pipefail

SA1B_COUNT="${SA1B_COUNT:-100}"
MANIFEST="${MANIFEST:-data/manifests/sa1b_fixed${SA1B_COUNT}.jsonl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-0}"
EVAL_MODE="${EVAL_MODE:-both}"
SKIP_MISSING="${SKIP_MISSING:-1}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/thor/offline/sa1b/${RUN_ID}}"
OVERLAY_DIR="${OVERLAY_DIR:-overlays/thor/offline/sa1b/${RUN_ID}}"
MODELS="${MODELS:-sam3 instinctsam_vitb sam2p1_hiera_large sam2p1_hiera_base_plus official_edgetam efficienttam_ti efficienttam_s mobilesam_vit_t sam1_vit_h}"

if [[ ! -f "${MANIFEST}" ]]; then
  SA1B_COUNT="${SA1B_COUNT}" MANIFEST="${MANIFEST}" bash scripts/prepare_sa1b_fixed_subset.sh
fi

MODEL_ARGS=()
if [[ -n "${MODELS}" ]]; then
  read -r -a MODEL_IDS <<< "${MODELS}"
  MODEL_ARGS+=(--models "${MODEL_IDS[@]}")
fi

SKIP_ARGS=()
if [[ "${SKIP_MISSING}" == "1" ]]; then
  SKIP_ARGS+=(--skip-missing)
fi

DRY_ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
  DRY_ARGS+=(--dry-run)
fi

python -m sam_backend.coco_suite \
  --manifest "${MANIFEST}" \
  --device "${DEVICE}" \
  --limit "${LIMIT}" \
  --eval-mode "${EVAL_MODE}" \
  --output-dir "${OUTPUT_DIR}" \
  --overlay-dir "${OVERLAY_DIR}" \
  "${MODEL_ARGS[@]}" \
  "${SKIP_ARGS[@]}" \
  "${DRY_ARGS[@]}"
