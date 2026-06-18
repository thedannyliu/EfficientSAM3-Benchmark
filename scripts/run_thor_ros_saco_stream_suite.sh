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

MANIFEST="${1:-data/manifests/saco_veval_sav_fixed20.jsonl}"
if [[ $# -gt 0 ]]; then
  shift
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
BUILD_ROS="${BUILD_ROS:-1}"
ROS_OUTPUT_DIR="${ROS_OUTPUT_DIR:-results/thor/ros_saco_stream/${RUN_ID}}"
ROS_OVERLAY_DIR="${ROS_OVERLAY_DIR:-overlays/thor/ros_saco_stream/${RUN_ID}}"
ROS_VIDEO_PATH="${ROS_VIDEO_PATH:-}"
ROS_INPUT_VIDEO="${ROS_INPUT_VIDEO:-${ROS_OUTPUT_DIR}/input/saco_stream_source.mp4}"
ROS_PROMPT="${ROS_PROMPT:-}"
ROS_PROMPT_FILE="${ROS_OUTPUT_DIR}/input/prompt.txt"
ROS_IMAGE_TOPIC="${ROS_IMAGE_TOPIC:-/image}"
ROS_INPUT_FPS="${ROS_INPUT_FPS:-${SAM_INPUT_FPS:-${INPUT_FPS:-30.0}}}"
ROS_MAX_MESSAGES="${ROS_MAX_MESSAGES:-${SAM_MAX_MESSAGES:-${MAX_FRAMES:-120}}}"
ROS_CLIP_FRAMES="${ROS_CLIP_FRAMES:-${ROS_MAX_MESSAGES}}"
ROS_TIMEOUT_SEC="${ROS_TIMEOUT_SEC:-900}"
ROS_STARTUP_SEC="${ROS_STARTUP_SEC:-5}"
ROS_INITIAL_POINT_X="${ROS_INITIAL_POINT_X:-0.5}"
ROS_INITIAL_POINT_Y="${ROS_INITIAL_POINT_Y:-0.5}"
ROS_INITIAL_POINT_NORMALIZED="${ROS_INITIAL_POINT_NORMALIZED:-true}"
ROS_DEVICE="${ROS_DEVICE:-cuda}"
ROS_SUPPORTED_MODELS=(
  mobilesam_vit_t_bbox_chain
  sam1_vit_b_bbox_chain
  sam1_vit_l_bbox_chain
  sam1_vit_h_bbox_chain
  sam3_ref_text_bbox_chain
  sam3_ref_native
  sam3p1_ref_native
  efficientsam3_ev_m_text_bbox_chain
  efficientsam3_rv_m_text_bbox_chain
  efficientsam3_tv_m_text_bbox_chain
)

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
elif [[ "${ROS_MODELS:-}" == "all" ]]; then
  MODELS=("${ROS_SUPPORTED_MODELS[@]}")
elif [[ -n "${ROS_MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  MODELS=(${ROS_MODELS})
else
  MODELS=("${ROS_SUPPORTED_MODELS[@]}")
fi

source scripts/source_thor_ros_env.sh

if [[ "${BUILD_ROS}" == "1" ]]; then
  if command -v colcon >/dev/null 2>&1; then
    (cd ros_ws && colcon build --symlink-install --packages-select sam_benchmark_ros)
    source scripts/source_thor_ros_env.sh
  else
    echo "WARNING: colcon not found; using existing ros_ws/install if available" >&2
  fi
fi

mkdir -p "${ROS_OUTPUT_DIR}" "${ROS_OVERLAY_DIR}" "$(dirname "${ROS_INPUT_VIDEO}")"

if [[ -z "${ROS_VIDEO_PATH}" ]]; then
  MANIFEST="${MANIFEST}" \
  ROS_INPUT_VIDEO="${ROS_INPUT_VIDEO}" \
  ROS_PROMPT_FILE="${ROS_PROMPT_FILE}" \
  ROS_MAX_MESSAGES="${ROS_MAX_MESSAGES}" \
  ROS_INPUT_FPS="${ROS_INPUT_FPS}" \
  python - <<'PY'
import json
import os
from pathlib import Path

import cv2

manifest = Path(os.environ["MANIFEST"])
output = Path(os.environ["ROS_INPUT_VIDEO"])
prompt_file = Path(os.environ["ROS_PROMPT_FILE"])
max_frames = int(os.environ["ROS_MAX_MESSAGES"])
fps = float(os.environ["ROS_INPUT_FPS"])

with manifest.open("r", encoding="utf-8") as f:
    row = json.loads(next(line for line in f if line.strip()))

media_root = Path(row["media_root"])
file_names = list(row.get("file_names", []))[:max_frames]
if not file_names:
    raise SystemExit(f"manifest row has no frames: {manifest}")

first = cv2.imread(str(media_root / file_names[0]), cv2.IMREAD_COLOR)
if first is None:
    raise SystemExit(f"failed to read first SA-Co frame: {media_root / file_names[0]}")

height, width = first.shape[:2]
output.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
if not writer.isOpened():
    raise SystemExit(f"failed to create ROS input video: {output}")

written = 0
for name in file_names:
    frame = cv2.imread(str(media_root / name), cv2.IMREAD_COLOR)
    if frame is None:
        continue
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    writer.write(frame)
    written += 1
writer.release()
if written == 0:
    raise SystemExit(f"no frames were written to {output}")

prompt = row.get("text_prompt") or row.get("noun_phrase") or "monitor"
prompt_file.parent.mkdir(parents=True, exist_ok=True)
prompt_file.write_text(str(prompt), encoding="utf-8")
print(json.dumps({"video": str(output), "frames": written, "prompt": prompt}))
PY
  ROS_VIDEO_PATH="${ROS_INPUT_VIDEO}"
fi

if [[ -z "${ROS_PROMPT}" ]]; then
  if [[ -f "${ROS_PROMPT_FILE}" ]]; then
    ROS_PROMPT="$(<"${ROS_PROMPT_FILE}")"
  else
    ROS_PROMPT="monitor"
  fi
fi

SUMMARY_CSV="${ROS_OUTPUT_DIR}/ros_saco_stream_summary.csv"
printf "model_id,status,result_csv,summary_csv,overlay_video,input_video,prompt,message\n" > "${SUMMARY_CSV}"

quote_cmd() {
  printf "%q " "$@"
  printf "\n"
}

cleanup_pids() {
  local pid
  for pid in "$@"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -INT "${pid}" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "$@"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "$@"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
}

wait_for_pid() {
  local pid="$1"
  local timeout_sec="$2"
  local deadline=$((SECONDS + timeout_sec))
  while kill -0 "${pid}" 2>/dev/null; do
    if (( SECONDS >= deadline )); then
      return 124
    fi
    sleep 1
  done
  wait "${pid}" || return $?
}

append_summary() {
  local model="$1"
  local status="$2"
  local result_csv="$3"
  local summary_csv="$4"
  local overlay_video="$5"
  local message="$6"
  python - "$SUMMARY_CSV" "$model" "$status" "$result_csv" "$summary_csv" "$overlay_video" "$ROS_VIDEO_PATH" "$ROS_PROMPT" "$message" <<'PY'
import csv
import sys

path, *values = sys.argv[1:]
with open(path, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(values)
PY
}

backend_cmd_for_model() {
  local model="$1"
  local result_topic="$2"
  local overlay_topic="$3"
  local mask_topic="$4"
  local segmented_topic="$5"
  local frame_dir="$6"

  case "${model}" in
    mobilesam_vit_t_bbox_chain)
      BACKEND_CMD=(ros2 run sam_benchmark_ros mobile_sam_interactive_node --ros-args
        -p "backend:=mobilesam"
        -p "checkpoint_path:=checkpoints/mobilesam/mobile_sam.pt"
        -p "external_repo:=external/MobileSAM"
        -p "device:=${ROS_DEVICE}"
        -p "mobile_sam_model_type:=vit_t"
        -p "enable_display:=false"
        -p "auto_start:=true"
        -p "initial_point_x:=${ROS_INITIAL_POINT_X}"
        -p "initial_point_y:=${ROS_INITIAL_POINT_Y}"
        -p "initial_point_normalized:=${ROS_INITIAL_POINT_NORMALIZED}"
        -p "image_topic:=${ROS_IMAGE_TOPIC}"
        -p "result_topic:=${result_topic}"
        -p "overlay_topic:=${overlay_topic}"
        -p "mask_topic:=${mask_topic}"
        -p "segmented_image_topic:=${segmented_topic}")
      ;;
    sam1_vit_b_bbox_chain|sam1_vit_l_bbox_chain|sam1_vit_h_bbox_chain)
      local model_type checkpoint
      case "${model}" in
        sam1_vit_b_bbox_chain) model_type="vit_b"; checkpoint="checkpoints/sam1/sam_vit_b_01ec64.pth" ;;
        sam1_vit_l_bbox_chain) model_type="vit_l"; checkpoint="checkpoints/sam1/sam_vit_l_0b3195.pth" ;;
        *) model_type="vit_h"; checkpoint="checkpoints/sam1/sam_vit_h_4b8939.pth" ;;
      esac
      BACKEND_CMD=(ros2 run sam_benchmark_ros mobile_sam_interactive_node --ros-args
        -p "backend:=sam1"
        -p "checkpoint_path:=${checkpoint}"
        -p "external_repo:=external/MobileSAM"
        -p "device:=${ROS_DEVICE}"
        -p "mobile_sam_model_type:=${model_type}"
        -p "enable_display:=false"
        -p "auto_start:=true"
        -p "initial_point_x:=${ROS_INITIAL_POINT_X}"
        -p "initial_point_y:=${ROS_INITIAL_POINT_Y}"
        -p "initial_point_normalized:=${ROS_INITIAL_POINT_NORMALIZED}"
        -p "image_topic:=${ROS_IMAGE_TOPIC}"
        -p "result_topic:=${result_topic}"
        -p "overlay_topic:=${overlay_topic}"
        -p "mask_topic:=${mask_topic}"
        -p "segmented_image_topic:=${segmented_topic}")
      ;;
    sam3_ref_text_bbox_chain)
      BACKEND_CMD=(ros2 run sam_benchmark_ros sam_backend_node --ros-args
        -p "backend:=sam3"
        -p "checkpoint_path:=checkpoints/sam3/sam3.pt"
        -p "external_repo:=external/sam3"
        -p "device:=${ROS_DEVICE}"
        -p "prompt_mode:=text"
        -p "prompt:=${ROS_PROMPT}"
        -p "image_topic:=${ROS_IMAGE_TOPIC}"
        -p "result_topic:=${result_topic}"
        -p "overlay_topic:=${overlay_topic}"
        -p "mask_topic:=${mask_topic}"
        -p "segmented_image_topic:=${segmented_topic}")
      ;;
    sam3_ref_native|sam3p1_ref_native)
      local checkpoint version
      if [[ "${model}" == "sam3p1_ref_native" ]]; then
        checkpoint="checkpoints/sam3p1/sam3.1_multiplex.pt"
        version="sam3.1"
      else
        checkpoint="checkpoints/sam3/sam3.pt"
        version="sam3"
      fi
      BACKEND_CMD=(ros2 run sam_benchmark_ros sam3_native_clip_node --ros-args
        -p "image_topic:=${ROS_IMAGE_TOPIC}"
        -p "checkpoint_path:=${checkpoint}"
        -p "external_repo:=external/sam3"
        -p "prompt:=${ROS_PROMPT}"
        -p "clip_frames:=${ROS_CLIP_FRAMES}"
        -p "frame_dir:=${frame_dir}"
        -p "version:=${version}"
        -p "result_topic:=${result_topic}"
        -p "overlay_topic:=${overlay_topic}"
        -p "mask_topic:=${mask_topic}"
        -p "segmented_image_topic:=${segmented_topic}")
      ;;
    efficientsam3_ev_m_text_bbox_chain|efficientsam3_rv_m_text_bbox_chain|efficientsam3_tv_m_text_bbox_chain)
      local checkpoint backbone model_name
      case "${model}" in
        efficientsam3_ev_m_text_bbox_chain)
          checkpoint="checkpoints/efficientsam3_ft/efficientsam3_efficientvit.pt"
          backbone="efficientvit"
          model_name="b1"
          ;;
        efficientsam3_rv_m_text_bbox_chain)
          checkpoint="checkpoints/efficientsam3_ft/efficientsam3_repvit.pt"
          backbone="repvit"
          model_name="m1.1"
          ;;
        *)
          checkpoint="checkpoints/efficientsam3_ft/efficientsam3_tinyvit.pt"
          backbone="tinyvit"
          model_name="11m"
          ;;
      esac
      BACKEND_CMD=(ros2 run sam_benchmark_ros sam_backend_node --ros-args
        -p "backend:=efficientsam3"
        -p "checkpoint_path:=${checkpoint}"
        -p "external_repo:=external/efficientsam3"
        -p "device:=${ROS_DEVICE}"
        -p "backbone_type:=${backbone}"
        -p "model_name:=${model_name}"
        -p "text_encoder_type:=MobileCLIP-S0"
        -p "text_encoder_context_length:=16"
        -p "text_encoder_pos_embed_table_size:=16"
        -p "prompt_mode:=text"
        -p "prompt:=${ROS_PROMPT}"
        -p "image_topic:=${ROS_IMAGE_TOPIC}"
        -p "result_topic:=${result_topic}"
        -p "overlay_topic:=${overlay_topic}"
        -p "mask_topic:=${mask_topic}"
        -p "segmented_image_topic:=${segmented_topic}")
      ;;
    *)
      return 1
      ;;
  esac
}

run_model() {
  local model="$1"
  local model_dir="${ROS_OUTPUT_DIR}/${model}"
  local overlay_dir="${ROS_OVERLAY_DIR}/${model}"
  local result_csv="${model_dir}/results.csv"
  local summary_csv="${model_dir}/summary.csv"
  local overlay_video="${overlay_dir}/overlay.mp4"
  local result_topic="/sam/${model}/result_json"
  local overlay_topic="/sam/${model}/overlay"
  local mask_topic="/sam/${model}/mask"
  local segmented_topic="/sam/${model}/segmented_image"
  local frame_dir="${model_dir}/frames"
  local publisher_pid="" backend_pid="" result_pid="" overlay_pid=""

  mkdir -p "${model_dir}" "${overlay_dir}" "${frame_dir}"
  if ! backend_cmd_for_model "${model}" "${result_topic}" "${overlay_topic}" "${mask_topic}" "${segmented_topic}" "${frame_dir}"; then
    echo "WARNING: unsupported ROS model id: ${model}" >&2
    append_summary "${model}" "skipped" "${result_csv}" "${summary_csv}" "${overlay_video}" "unsupported ROS model id"
    return 0
  fi

  RESULT_CMD=(ros2 run sam_benchmark_ros result_recorder_node --ros-args
    -p "result_topic:=${result_topic}"
    -p "max_messages:=${ROS_MAX_MESSAGES}"
    -p "csv_output:=${result_csv}"
    -p "summary_output:=${summary_csv}")
  OVERLAY_CMD=(ros2 run sam_benchmark_ros overlay_video_recorder_node --ros-args
    -p "overlay_topic:=${overlay_topic}"
    -p "fps:=${ROS_INPUT_FPS}"
    -p "max_frames:=${ROS_MAX_MESSAGES}"
    -p "video_output:=${overlay_video}")
  PUBLISHER_CMD=(ros2 run sam_benchmark_ros video_stream_node --ros-args
    -p "video_path:=${ROS_VIDEO_PATH}"
    -p "image_topic:=${ROS_IMAGE_TOPIC}"
    -p "fps:=${ROS_INPUT_FPS}"
    -p "frame_id:=saco_video")

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "model=${model}"
    echo "  result:  $(quote_cmd "${RESULT_CMD[@]}")"
    echo "  overlay: $(quote_cmd "${OVERLAY_CMD[@]}")"
    echo "  backend: $(quote_cmd "${BACKEND_CMD[@]}")"
    echo "  source:  $(quote_cmd "${PUBLISHER_CMD[@]}")"
    append_summary "${model}" "dry-run" "${result_csv}" "${summary_csv}" "${overlay_video}" "dry-run"
    return 0
  fi

  "${RESULT_CMD[@]}" &
  result_pid=$!
  "${OVERLAY_CMD[@]}" &
  overlay_pid=$!
  sleep 1
  "${BACKEND_CMD[@]}" &
  backend_pid=$!
  sleep "${ROS_STARTUP_SEC}"
  "${PUBLISHER_CMD[@]}" &
  publisher_pid=$!

  if wait_for_pid "${result_pid}" "${ROS_TIMEOUT_SEC}"; then
    append_summary "${model}" "ok" "${result_csv}" "${summary_csv}" "${overlay_video}" ""
  else
    append_summary "${model}" "failed" "${result_csv}" "${summary_csv}" "${overlay_video}" "timed out or recorder failed"
  fi
  cleanup_pids "${publisher_pid}" "${backend_pid}" "${overlay_pid}" "${result_pid}"
}

echo "ROS SA-Co video stream benchmark"
echo "  input video: ${ROS_VIDEO_PATH}"
echo "  prompt: ${ROS_PROMPT}"
echo "  fps: ${ROS_INPUT_FPS}"
echo "  max messages: ${ROS_MAX_MESSAGES}"
echo "  output: ${ROS_OUTPUT_DIR}"
echo "  overlays: ${ROS_OVERLAY_DIR}"

for model in "${MODELS[@]}"; do
  run_model "${model}"
done

cat <<EOF
ROS SA-Co video stream benchmark complete.

Summary:
  ${SUMMARY_CSV}

Per-model outputs:
  ${ROS_OUTPUT_DIR}/<model_id>/results.csv
  ${ROS_OUTPUT_DIR}/<model_id>/summary.csv

Overlays:
  ${ROS_OVERLAY_DIR}/<model_id>/overlay.mp4

Latency columns:
  latency_ms = model itself
  end_to_end_ms = ROS image stamp to result publication
  mean_end_to_end_fps = 1000 / mean_end_to_end_ms
EOF
