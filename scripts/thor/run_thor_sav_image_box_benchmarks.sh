#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/thor/formal_full/${RUN_ID}}"
OVERLAY_ROOT="${OVERLAY_ROOT:-overlays/thor/formal_full/${RUN_ID}}"
SAV_ROOT="${SAV_ROOT:-data/sa-v/sav_test}"
SAV_IMAGE_COUNT="${SAV_IMAGE_COUNT:-1000}"
SAV_IMAGE_SEED="${SAV_IMAGE_SEED:-20260708}"
SAV_IMAGE_MIN_AREA="${SAV_IMAGE_MIN_AREA:-1}"
DEVICE="${DEVICE:-cuda}"
MODELS="${MODELS:-mobilesam_vit_t sam1_vit_l sam3}"
PROMPT_MODE="${PROMPT_MODE:-box}"
EVAL_MODE="${EVAL_MODE:-both}"
SKIP_DONE="${SKIP_DONE:-1}"

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

OUT_DIR="${OUTPUT_ROOT}/sav_image_sam_family"
OVERLAY_DIR="${OVERLAY_ROOT}/sav_image_sam_family"
MANIFEST="${OUTPUT_ROOT}/manifests/sav_test_image_box_${SAV_IMAGE_COUNT}.jsonl"

mkdir -p "${OUTPUT_ROOT}/manifests" "${OUT_DIR}" "${OVERLAY_DIR}"

echo "RUN_ID=${RUN_ID}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "OVERLAY_ROOT=${OVERLAY_ROOT}"
echo "SAV_ROOT=${SAV_ROOT}"
echo "SAV_IMAGE_COUNT=${SAV_IMAGE_COUNT}"
echo "MANIFEST=${MANIFEST}"
echo "OUT_DIR=${OUT_DIR}"
echo "OVERLAY_DIR=${OVERLAY_DIR}"

python -m sam_backend.sav_image_manifest \
  --sav-root "${SAV_ROOT}" \
  --count "${SAV_IMAGE_COUNT}" \
  --seed "${SAV_IMAGE_SEED}" \
  --min-area "${SAV_IMAGE_MIN_AREA}" \
  --output "${MANIFEST}"

run_profile() {
  local model_id="$1"
  local backend="$2"
  local checkpoint="$3"
  local external_repo="$4"
  shift 4

  local model_dir="${OUT_DIR}/${model_id}"
  local overlay_dir="${OVERLAY_DIR}/${model_id}"
  local csv_path="${model_dir}/profile.csv"
  local summary_path="${model_dir}/summary.json"
  mkdir -p "${model_dir}" "${overlay_dir}"

  if [[ "${SKIP_DONE}" == "1" && -s "${summary_path}" && -s "${csv_path}" ]]; then
    echo "skip done: ${model_id}"
    return 0
  fi

  python -m sam_backend.profile_coco \
    --manifest "${MANIFEST}" \
    --model-id "${model_id}" \
    --backend "${backend}" \
    --checkpoint-path "${checkpoint}" \
    --external-repo "${external_repo}" \
    --prompt-mode "${PROMPT_MODE}" \
    --eval-mode "${EVAL_MODE}" \
    --device "${DEVICE}" \
    --csv-output "${csv_path}" \
    --summary-output "${summary_path}" \
    --overlay-dir "${overlay_dir}" \
    "$@"
}

for model in ${MODELS}; do
  case "${model}" in
    mobilesam_vit_t)
      run_profile mobilesam_vit_t mobilesam checkpoints/mobilesam/mobile_sam.pt external/MobileSAM \
        --mobile-sam-model-type vit_t
      ;;
    sam1_vit_l)
      run_profile sam1_vit_l sam1 checkpoints/sam1/sam_vit_l_0b3195.pth external/MobileSAM \
        --mobile-sam-model-type vit_l
      ;;
    sam3)
      run_profile sam3 sam3 checkpoints/sam3/sam3.pt external/sam3
      ;;
    *)
      echo "unknown model: ${model}" >&2
      exit 2
      ;;
  esac
done

python - "${OUT_DIR}" <<'PY'
from pathlib import Path
import sys

from sam_backend.coco_suite import write_component_summary, write_model_summary

out_dir = Path(sys.argv[1])
component = write_component_summary(out_dir)
model = write_model_summary(out_dir)
if component:
    print(component)
if model:
    print(model)
PY

python -m sam_backend.thor_smoke_summary \
  --root "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/thor_formal_full_summary.csv"

echo "summary=${OUTPUT_ROOT}/thor_formal_full_summary.csv"
echo "sav_image_summary=${OUT_DIR}/coco_suite_model_summary.csv"
