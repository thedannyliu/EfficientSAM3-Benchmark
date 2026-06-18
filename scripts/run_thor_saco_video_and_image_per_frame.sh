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
DRY_RUN="${DRY_RUN:-0}"
RUN_OFFLINE="${RUN_OFFLINE:-1}"
RUN_ROS="${RUN_ROS:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
SACO_SPLIT="${SACO_SPLIT:-val}"
SACO_COUNT="${SACO_COUNT:-20}"
SACO_ANNOTATION="${SACO_ANNOTATION:-${SAM_BENCH_SCRATCH}/data/annotation/saco_veval_sav_${SACO_SPLIT}.json}"
SACO_MANIFEST="${SACO_MANIFEST:-data/manifests/saco_veval_sav_fixed${SACO_COUNT}.jsonl}"
MAX_FRAMES="${MAX_FRAMES:-120}"
INPUT_FPS="${INPUT_FPS:-30.0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/thor/saco_video_image_per_frame/${RUN_ID}}"
OVERLAY_DIR="${OVERLAY_DIR:-overlays/thor/saco_video_image_per_frame/${RUN_ID}}"
ROS_OUTPUT_DIR="${ROS_OUTPUT_DIR:-results/thor/ros_saco_stream/${RUN_ID}}"
ROS_OVERLAY_DIR="${ROS_OVERLAY_DIR:-overlays/thor/ros_saco_stream/${RUN_ID}}"

if [[ "${SETUP}" == "1" ]]; then
  RUN_SUITE=0 \
  RUN_NULL_SMOKE="${RUN_NULL_SMOKE:-1}" \
  bash scripts/setup_thor_saco_stream_benchmark.sh
fi

source scripts/source_thor_ros_env.sh

if [[ "${RUN_OFFLINE}" == "1" ]]; then
  suite_args=(
    --manifest "${SACO_MANIFEST}"
    --gt-annotation-file "${SACO_ANNOTATION}"
    --scratch-root "${SAM_BENCH_SCRATCH}"
    --mode-set all
    --max-frames "${MAX_FRAMES}"
    --input-fps "${INPUT_FPS}"
    --output-dir "${OUTPUT_DIR}"
    --overlay-dir "${OVERLAY_DIR}"
    --skip-missing
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    suite_args+=(--dry-run)
  fi

  if [[ -n "${SACO_MODELS:-}" ]]; then
    # shellcheck disable=SC2206
    model_args=(${SACO_MODELS})
    suite_args+=(--models "${model_args[@]}")
  fi

  sam-run-saco-stream-suite "${suite_args[@]}"
fi

if [[ "${RUN_ROS}" == "1" ]]; then
  DRY_RUN="${DRY_RUN}" \
  RUN_ID="${RUN_ID}" \
  SAM_BENCH_SCRATCH="${SAM_BENCH_SCRATCH}" \
  ROS_OUTPUT_DIR="${ROS_OUTPUT_DIR}" \
  ROS_OVERLAY_DIR="${ROS_OVERLAY_DIR}" \
  ROS_INPUT_FPS="${INPUT_FPS}" \
  ROS_MAX_MESSAGES="${MAX_FRAMES}" \
  bash scripts/run_thor_ros_saco_stream_suite.sh "${SACO_MANIFEST}"
fi

cat <<EOF
SA-Co benchmark complete.

Offline summary:
  ${OUTPUT_DIR}/saco_stream_suite_summary.csv

Offline per-model outputs:
  ${OUTPUT_DIR}/<model_id>/frames.csv
  ${OUTPUT_DIR}/<model_id>/frames_summary.csv
  ${OUTPUT_DIR}/<model_id>/summary.json

Offline overlays:
  ${OVERLAY_DIR}/<model_id>/<source_id>/overlay.mp4

ROS video stream summary:
  ${ROS_OUTPUT_DIR}/ros_saco_stream_summary.csv

ROS video stream outputs:
  ${ROS_OUTPUT_DIR}/<model_id>/results.csv
  ${ROS_OUTPUT_DIR}/<model_id>/summary.csv
  ${ROS_OVERLAY_DIR}/<model_id>/overlay.mp4

Latency columns:
  latency_ms = model itself
  offline end_to_end_ms = full per-frame benchmark step
  ROS end_to_end_ms = ROS image stamp to result publication
  FPS columns are computed from end-to-end latency
EOF
