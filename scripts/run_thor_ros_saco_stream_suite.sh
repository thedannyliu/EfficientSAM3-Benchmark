#!/usr/bin/env bash
set -euo pipefail

SCRATCH_ROOT="${SAM_BENCH_SCRATCH:-/storage/scratch1/9/eliu354/efficientsam3-benchmark}"
MANIFEST="${1:?usage: scripts/run_thor_ros_saco_stream_suite.sh MANIFEST_JSONL [MODEL_ID ...]}"
shift || true
MODELS=("$@")
FPS="${SAM_INPUT_FPS:-30.0}"
MAX_MESSAGES="${SAM_MAX_MESSAGES:-300}"

mkdir -p results/thor/ros_saco overlays/thor/ros_saco

if [[ ${#MODELS[@]} -eq 0 ]]; then
  mapfile -t MODELS < <(python - <<'PY'
from sam_backend.saco_stream_suite import DEFAULT_RUNS
for run in DEFAULT_RUNS:
    print(run.model_id)
PY
)
fi

echo "This helper records ROS result/overlay topics for one model at a time."
echo "Start the matching backend command from sam-run-saco-stream-suite dry-run output, then run:"
for model in "${MODELS[@]}"; do
  echo "  model=${model}"
  mkdir -p "results/thor/ros_saco/${model}" "overlays/thor/ros_saco/${model}"
  echo "  ros2 run sam_benchmark_ros result_recorder_node --ros-args -p max_messages:=${MAX_MESSAGES} -p csv_output:=results/thor/ros_saco/${model}/results.csv -p summary_output:=results/thor/ros_saco/${model}/summary.csv"
  echo "  ros2 run sam_benchmark_ros overlay_video_recorder_node --ros-args -p fps:=${FPS} -p max_frames:=${MAX_MESSAGES} -p video_output:=overlays/thor/ros_saco/${model}/overlay.mp4"
done
