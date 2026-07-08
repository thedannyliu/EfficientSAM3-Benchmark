#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/thor/formal_full/${RUN_ID}}"
OVERLAY_ROOT="${OVERLAY_ROOT:-overlays/thor/formal_full/${RUN_ID}}"
SAV_VIDEO_COUNT="${SAV_VIDEO_COUNT:-0}"
SAV_MANIFEST_COUNT="${SAV_MANIFEST_COUNT:-0}"
SAV_MAX_FRAMES="${SAV_MAX_FRAMES:-0}"
MAX_IMAGE_OBJECTS="${MAX_IMAGE_OBJECTS:-0}"
SA1B_COUNT="${SA1B_COUNT:-1000}"
SA1B_LIMIT="${SA1B_LIMIT:-${SA1B_COUNT}}"
SA1B_MAX_IMAGE_OBJECTS="${SA1B_MAX_IMAGE_OBJECTS:-0}"
SA1B_IMAGE_ARTIFACT_VIDEOS="${SA1B_IMAGE_ARTIFACT_VIDEOS:-3}"

OUTPUT_ROOT="${OUTPUT_ROOT}" \
OVERLAY_ROOT="${OVERLAY_ROOT}" \
SAV_VIDEO_COUNT="${SAV_VIDEO_COUNT}" \
SAV_MANIFEST_COUNT="${SAV_MANIFEST_COUNT}" \
SAV_MAX_FRAMES="${SAV_MAX_FRAMES}" \
MAX_IMAGE_OBJECTS="${MAX_IMAGE_OBJECTS}" \
SA1B_COUNT="${SA1B_COUNT}" \
SA1B_LIMIT="${SA1B_LIMIT}" \
SA1B_MAX_IMAGE_OBJECTS="${SA1B_MAX_IMAGE_OBJECTS}" \
SA1B_IMAGE_ARTIFACT_VIDEOS="${SA1B_IMAGE_ARTIFACT_VIDEOS}" \
bash scripts/run_thor_formal_smoke_matrix.sh

summary_dir="${OUTPUT_ROOT}"
case "${summary_dir}" in
  /*) ;;
  *) summary_dir="$(pwd)/${summary_dir}" ;;
esac

if [[ -f "${summary_dir}/thor_formal_smoke_summary.csv" ]]; then
  cp "${summary_dir}/thor_formal_smoke_summary.csv" "${summary_dir}/thor_formal_full_summary.csv"
  echo "full_summary=${summary_dir}/thor_formal_full_summary.csv"
fi
