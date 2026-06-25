#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

source scripts/source_thor_ros_env.sh

offline_base="${OFFLINE_BASE:-results/thor/saco_video_image_per_frame}"
ros_base="${ROS_BASE:-results/thor/ros_saco_stream}"
output="${OUTPUT:-results/thor/saco_model_wise_summary.csv}"

args=(--offline-base "${offline_base}" --ros-base "${ros_base}" --output "${output}")

if [[ -n "${OFFLINE_RUN_ID:-}" ]]; then
  args+=(--offline-root "${offline_base}/${OFFLINE_RUN_ID}")
fi

if [[ -n "${ROS_RUN_ID:-}" ]]; then
  args+=(--ros-root "${ros_base}/${ROS_RUN_ID}")
fi

if [[ "${OFFLINE_ONLY:-0}" == "1" ]]; then
  args+=(--offline-only)
fi

if [[ -n "${MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  model_args=(${MODELS})
  args+=(--models "${model_args[@]}")
fi

python -m sam_backend.saco_model_summary "${args[@]}"

cat <<EOF
Model-wise SA-Co summary written to:
  ${output}

Set OFFLINE_RUN_ID and ROS_RUN_ID to summarize a specific run instead of the
latest directories.
EOF
