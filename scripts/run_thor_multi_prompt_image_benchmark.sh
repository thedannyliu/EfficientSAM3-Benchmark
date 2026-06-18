#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

requested_asset_root="${SAM_BENCH_SCRATCH:-${repo_root}}"
if ! mkdir -p "${requested_asset_root}" 2>/dev/null || [[ ! -w "${requested_asset_root}" ]]; then
  echo "WARNING: SAM_BENCH_SCRATCH is not writable: ${requested_asset_root}" >&2
  echo "WARNING: falling back to repo-local asset root: ${repo_root}" >&2
  requested_asset_root="${repo_root}"
fi
export SAM_BENCH_SCRATCH="${requested_asset_root}"

SETUP="${SETUP:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
SACO_COUNT="${SACO_COUNT:-20}"
SACO_MANIFEST="${SACO_MANIFEST:-data/manifests/saco_veval_sav_fixed${SACO_COUNT}.jsonl}"
IMAGE_COUNT="${IMAGE_COUNT:-10}"
POINT_COUNTS="${POINT_COUNTS:-1,2,3,5,10,15}"
SUITE="${SUITE:-all}"
TEXT_PROMPT="${TEXT_PROMPT:-}"
WARMUP="${WARMUP:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-results/thor/multi_prompt_image/${RUN_ID}}"
OVERLAY_DIR="${OVERLAY_DIR:-overlays/thor/multi_prompt_image/${RUN_ID}}"
WORK_DIR="${WORK_DIR:-${OUTPUT_DIR}/work}"

if [[ "${SETUP}" == "1" ]]; then
  RUN_SUITE=0 \
  RUN_NULL_SMOKE="${RUN_NULL_SMOKE:-0}" \
  bash scripts/setup_thor_saco_stream_benchmark.sh
fi

source scripts/source_thor_ros_env.sh

args=(
  --suite "${SUITE}"
  --manifest "${SACO_MANIFEST}"
  --image-count "${IMAGE_COUNT}"
  --point-counts "${POINT_COUNTS}"
  --warmup "${WARMUP}"
  --work-dir "${WORK_DIR}"
  --csv-output "${OUTPUT_DIR}/frames.csv"
  --summary-output "${OUTPUT_DIR}/summary.csv"
  --overlay-root "${OVERLAY_DIR}"
  --skip-missing
)

if [[ -n "${TEXT_PROMPT}" ]]; then
  args+=(--text-prompt "${TEXT_PROMPT}")
fi

if [[ -n "${IMAGE_DIR:-}" ]]; then
  args+=(--image-dir "${IMAGE_DIR}")
fi

sam-profile-multi-prompt-image "${args[@]}"

cat <<EOF
Thor multi-prompt image benchmark complete.

Frame-level CSV:
  ${OUTPUT_DIR}/frames.csv

Summary CSV:
  ${OUTPUT_DIR}/summary.csv

Overlays:
  ${OVERLAY_DIR}/<model_id>/*.jpg

Latency definition:
  model_ms = model-only timed section
  MobileSAM model_ms = image encoder once + N independent point decodes
  SAM3/SAM3.1 model_ms = single-frame native session + text prompt + one-frame propagation
EOF
