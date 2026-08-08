#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 LABEL STATEFUL_TRUE_OR_FALSE" >&2
    exit 2
fi

label=$1
stateful=$2
if [[ "$stateful" != "true" && "$stateful" != "false" ]]; then
    echo "stateful must be true or false" >&2
    exit 2
fi

runtime_container=instinctsam-t02-refresh30-headless
ether_container=ether-onboard-gi-eval
repo_dir=/home/magni/efficientsam3-benchmark
bag_path=/workspace/mercury_20260617_141052/mercury_20260617_141052.mcap
run_root=/home/magni/ether-onboard/.scene_graph_runs/gi-scene-graph-t03-20260808
output_dir="$run_root/$label"
nas_root=/mnt/nas/danny/thor-scene-graph/run-artifacts/gi-scene-graph-t03-20260808

if [[ -e "$output_dir" || -e "$nas_root/$label" ]]; then
    echo "refusing to overwrite existing T03 output for $label" >&2
    exit 1
fi

mkdir -p "$output_dir"
cp "$repo_dir/sam_backend/scene_graph_ab_recorder.py" "$output_dir/"
cp "$repo_dir/sam_backend/scene_graph_pose_fixture.py" "$output_dir/"
cp "$repo_dir/configs/scene_graph_tracking_t01_prompts.json" "$output_dir/prompts.json"
pose_fixture_sha256=$(sha256sum "$output_dir/scene_graph_pose_fixture.py" | awk '{print $1}')

cleanup() {
    docker stop "$ether_container" >/dev/null 2>&1 || true
    docker stop "$runtime_container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

runtime_started_ns=$(date +%s%N)
runtime_log_since=$(date --iso-8601=seconds)
docker start "$runtime_container" >/dev/null
ready=false
for _ in $(seq 1 240); do
    if curl --silent --show-error --fail --max-time 2 \
        http://127.0.0.1:8767/status.json >"$output_dir/runtime_ready_status.json" 2>/dev/null; then
        ready=true
        break
    fi
    sleep 1
done
if [[ "$ready" != "true" ]]; then
    echo "GI runtime did not become ready" >&2
    exit 1
fi
runtime_ready_ns=$(date +%s%N)

docker start "$ether_container" >/dev/null

container_output="/workspace/.scene_graph_runs/gi-scene-graph-t03-20260808/$label"
ros_setup='source /opt/ros/humble/setup.bash; source /workspace/install/setup.bash; export RCUTILS_COLORIZED_OUTPUT=0'

docker exec -d "$ether_container" bash -lc \
    "$ros_setup; export PYTHONPATH=/workspace/src/scene_graph/src:\${PYTHONPATH}; exec python3 /workspace/src/scene_graph/src/scene_graph_ros_node.py --ros-args -p mode:=online -p category_config_file:=/root/.ros/ether/scene/maps/scene_objects.json -p online_scene_graph_file:='$container_output/final_graph.json' -p detection.confidence_threshold:=0.5 >'$container_output/scene_graph.log' 2>&1"

docker exec -d "$ether_container" bash -lc \
    "$ros_setup; export PYTHONPATH=/workspace/src/scene_graph/src:\${PYTHONPATH}; exec python3 /workspace/src/scene_graph/src/detection_ros_node.py --ros-args -p config_file_path:='$container_output/prompts.json' -p detection_node.backend:=instinctsam_http -p detection_node.instinctsam_url:=http://127.0.0.1:8767 -p detection_node.instinctsam_timeout:=30.0 -p detection_node.instinctsam_stateful:=$stateful -p detection_node.headless:=true -p detection_node.depth:=camera -p detection.confidence_threshold:=0.5 -p use_sim_time:=true >'$container_output/detection.log' 2>&1"

docker exec -d "$ether_container" bash -lc \
    "$ros_setup; export PYTHONPATH=/workspace/src/scene_graph/src:\${PYTHONPATH}; exec python3 '$container_output/scene_graph_ab_recorder.py' --output-dir '$container_output' --label '$label' --sample-period 5.0 --max-wall-duration 42 >'$container_output/recorder.log' 2>&1"

docker exec -d "$ether_container" bash -lc \
    "$ros_setup; exec python3 '$container_output/scene_graph_pose_fixture.py' >'$container_output/pose_fixture.log' 2>&1"

sleep 5

python3 -m sam_backend.thor_resources \
    --output "$output_dir/resources.jsonl" \
    --duration 35 \
    --interval 1 \
    --container "$runtime_container" \
    --container "$ether_container" \
    --label "$label" \
    >"$output_dir/resource_sampler.log" 2>&1 &
resource_pid=$!

play_started_ns=$(date +%s%N)
set +e
docker exec "$ether_container" bash -lc \
    "$ros_setup; timeout --signal=INT 30s ros2 bag play '$bag_path' --rate 1.0 --clock 100 --start-offset 145.3 --topics /d435/color/image_raw_jpeg /d435/aligned_depth_to_color/image_raw /d435/color/camera_info --disable-keyboard-controls" \
    >"$output_dir/bag_play.log" 2>&1
bag_rc=$?
set -e
play_ended_ns=$(date +%s%N)
if [[ $bag_rc -ne 0 && $bag_rc -ne 124 && $bag_rc -ne 130 ]]; then
    echo "bag playback failed with exit $bag_rc" >&2
    exit "$bag_rc"
fi

sleep 3
wait "$resource_pid"

curl --silent --show-error --fail --max-time 3 \
    http://127.0.0.1:8767/status.json >"$output_dir/runtime_final_status.json"
for _ in $(seq 1 10); do
    [[ -f "$output_dir/recorder_summary.json" ]] && break
    sleep 1
done
if [[ ! -f "$output_dir/recorder_summary.json" ]]; then
    echo "recorder did not write its summary" >&2
    exit 1
fi
docker logs --since "$runtime_log_since" "$runtime_container" >"$output_dir/runtime.log" 2>&1

printf '{\n  "label": "%s",\n  "stateful": %s,\n  "runtime_started_ns": %s,\n  "runtime_ready_ns": %s,\n  "runtime_startup_seconds": %.6f,\n  "play_started_ns": %s,\n  "play_ended_ns": %s,\n  "play_wall_seconds": %.6f,\n  "bag_exit_code": %s,\n  "bag_start_offset_seconds": 145.3,\n  "requested_play_seconds": 30.0,\n  "scene_graph_commit": "46673c6",\n  "pose_fixture_sha256": "%s",\n  "runtime_overlay_sha256": "c6685227317c6698e4cd56f2ba1ba28905cb756ba182d61fb3af96192d703efd"\n}\n' \
    "$label" "$stateful" "$runtime_started_ns" "$runtime_ready_ns" \
    "$(awk -v a="$runtime_started_ns" -v b="$runtime_ready_ns" 'BEGIN {print (b-a)/1000000000}')" \
    "$play_started_ns" "$play_ended_ns" \
    "$(awk -v a="$play_started_ns" -v b="$play_ended_ns" 'BEGIN {print (b-a)/1000000000}')" \
    "$bag_rc" "$pose_fixture_sha256" >"$output_dir/run_metadata.json"

cleanup
trap - EXIT

mkdir -p "$nas_root"
cp -a "$output_dir" "$nas_root/"
checksum_tmp=$(mktemp)
(
    cd "$nas_root/$label"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"$checksum_tmp"
    mv "$checksum_tmp" SHA256SUMS
    sha256sum --check SHA256SUMS >/dev/null
)

echo "$nas_root/$label"
