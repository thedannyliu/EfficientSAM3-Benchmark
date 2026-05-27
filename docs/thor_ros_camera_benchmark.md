# Jetson Thor ROS Camera Benchmark and Profiling

This guide runs the live ROS camera pipeline on Jetson Thor:

```text
camera_stream_node -> /image
/image -> sam_backend_node -> /sam/result_json
                         \
                          -> /sam/overlay
result_recorder_node -> CSV + summary CSV
overlay_video_recorder_node -> overlay MP4
```

Use this path after the offline benchmark works. The ROS numbers include model
latency plus callback and transport overhead.

## 1. Prepare The Same Environment As Offline

Start from the feature branch:

```bash
git clone git@github.com:thedannyliu/EfficientSAM3-Benchmark.git
cd EfficientSAM3-Benchmark
git fetch origin
git checkout benchmark-coco-sav-profiling
```

Create the Thor venv and install Jetson-compatible PyTorch first. Follow
NVIDIA's current PyTorch for Jetson instructions:

```text
https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
```

Then install this repo. Use `--system-site-packages` so the venv can use the
ROS and Jetson OpenCV packages installed by APT.

```bash
python3 -m venv --system-site-packages ~/venvs/effisam3_venv_ros
source ~/venvs/effisam3_venv_ros/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-thor.txt
python -m pip install -e .
```

Do not use `requirements.txt` on Thor unless you intentionally want to manage
PyTorch yourself; it pins the PACE CUDA PyTorch packages.

Install model source repos and checkpoints:

```bash
source ~/venvs/effisam3_venv_ros/bin/activate
cd EfficientSAM3-Benchmark
bash scripts/setup_model_repos.sh
bash scripts/download_sam3_checkpoint.sh
bash scripts/download_efficientsam3_checkpoints.sh
bash scripts/download_sam2_family_checkpoints.sh
bash scripts/download_yoloe_edgetam_mobilesam_assets.sh
```

## 2. Install And Source ROS 2

This repo assumes ROS 2 Jazzy on Thor.

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-ros-base \
  ros-jazzy-cv-bridge \
  ros-jazzy-sensor-msgs \
  ros-jazzy-std-msgs \
  python3-opencv \
  python3-colcon-common-extensions
```

Use the repo helper in every ROS terminal:

```bash
cd EfficientSAM3-Benchmark
export THOR_ROS_SETUP=/opt/ros/jazzy/setup.bash
export THOR_VENV=~/venvs/effisam3_venv_ros
source scripts/source_thor_ros_env.sh
```

Check imports:

```bash
python - <<'PY'
import cv2, rclpy, cv_bridge, torch, sam_backend
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("ros imports ok")
PY
```

## 3. Build The ROS Workspace

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh
cd ros_ws
colcon build --symlink-install
cd ..
source scripts/source_thor_ros_env.sh
```

Confirm entrypoints:

```bash
ros2 pkg executables sam_benchmark_ros
```

Expected entries include:

```text
camera_stream_node
video_stream_node
sam_backend_node
result_recorder_node
overlay_video_recorder_node
```

## 4. Start The Camera Publisher

Terminal A, simple OpenCV camera index:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros camera_stream_node --ros-args \
  -p camera_index:=0 \
  -p image_topic:=/image \
  -p width:=1280 \
  -p height:=720 \
  -p fps:=30.0 \
  -p frame_id:=camera
```

If Thor needs a GStreamer source, pass it as one string:

```bash
ros2 run sam_benchmark_ros camera_stream_node --ros-args \
  -p image_topic:=/image \
  -p fps:=30.0 \
  -p gstreamer_pipeline:='YOUR_GSTREAMER_PIPELINE_STRING'
```

Verify publishing:

```bash
ros2 topic hz /image
ros2 topic echo /image/header --once
```

## 5. Run A Null Backend Smoke Test

Terminal B:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=null \
  -p device:=cpu \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay
```

Terminal C, record 100 result messages:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh
mkdir -p results/thor/ros_camera/null overlays/thor/ros_camera/null

ros2 run sam_benchmark_ros result_recorder_node --ros-args \
  -p result_topic:=/sam/result_json \
  -p csv_output:=results/thor/ros_camera/null/results.csv \
  -p summary_output:=results/thor/ros_camera/null/summary.csv \
  -p max_messages:=100
```

Terminal D, record matching overlays:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros overlay_video_recorder_node --ros-args \
  -p overlay_topic:=/sam/overlay \
  -p video_output:=overlays/thor/ros_camera/null/overlay.mp4 \
  -p fps:=30.0 \
  -p max_frames:=100
```

Proceed to real models only after the null CSV and overlay MP4 are created.

## 6. Run SAM3 Text-Prompt Camera Benchmark

Stop the null backend. Keep the camera publisher running.

Terminal B:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=sam3 \
  -p external_repo:=external/sam3 \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p device:=cuda \
  -p prompt_mode:=text \
  -p prompt:=person \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay
```

Terminals C and D:

```bash
mkdir -p results/thor/ros_camera/sam3 overlays/thor/ros_camera/sam3

ros2 run sam_benchmark_ros result_recorder_node --ros-args \
  -p csv_output:=results/thor/ros_camera/sam3/results.csv \
  -p summary_output:=results/thor/ros_camera/sam3/summary.csv \
  -p max_messages:=300

ros2 run sam_benchmark_ros overlay_video_recorder_node --ros-args \
  -p video_output:=overlays/thor/ros_camera/sam3/overlay.mp4 \
  -p fps:=30.0 \
  -p max_frames:=300
```

## 7. Run EfficientSAM3 Text-Prompt Camera Benchmark

EfficientSAM3 weak image / weak text:

```bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=efficientsam3 \
  -p external_repo:=external/efficientsam3 \
  -p checkpoint_path:=checkpoints/stage1_sam3p1/efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt \
  -p device:=cuda \
  -p backbone_type:=efficientvit \
  -p model_name:=b0 \
  -p text_encoder_type:=MobileCLIP-S0 \
  -p text_encoder_context_length:=16 \
  -p text_encoder_pos_embed_table_size:=16 \
  -p prompt_mode:=text \
  -p prompt:=person \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay
```

For other variants, change checkpoint and model parameters:

```text
es3p1_strong_image_weak_text:
  checkpoint_path=checkpoints/stage1_sam3p1/efficient_sam3p1_efficientvit_l_mobileclip_s0_ctx16.pt
  model_name=b2
  text_encoder_type=MobileCLIP-S0
  text_encoder_pos_embed_table_size=16

es3_weak_image_strong_available_text:
  checkpoint_path=checkpoints/stage1_all_converted/efficient_sam3_efficientvit-b0_mobileclip_s1.pth
  model_name=b0
  text_encoder_type=MobileCLIP-S1
  text_encoder_pos_embed_table_size=77

es3_strong_image_strong_available_text:
  checkpoint_path=checkpoints/stage1_all_converted/efficient_sam3_efficientvit-b2_mobileclip_s1.pth
  model_name=b2
  text_encoder_type=MobileCLIP-S1
  text_encoder_pos_embed_table_size=77
```

Use separate output folders per variant, for example:

```text
results/thor/ros_camera/es3p1_weak_image_weak_text/
overlays/thor/ros_camera/es3p1_weak_image_weak_text/
```

## 8. Run Point-Prompt Camera Benchmarks

Point prompt is fixed relative to the incoming image when `point_normalized` is
true. `point_x:=0.5 -p point_y:=0.5` means the center of the frame.

SAM2.1 tiny:

```bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=sam2 \
  -p external_repo:=external/sam2 \
  -p checkpoint_path:=checkpoints/sam2/sam2.1_hiera_tiny.pt \
  -p model_config:=configs/sam2.1/sam2.1_hiera_t.yaml \
  -p device:=cuda \
  -p prompt_mode:=point \
  -p point_x:=0.5 \
  -p point_y:=0.5 \
  -p point_normalized:=true \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay
```

Efficient-SAM2.1 tiny:

```bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=efficient-sam2 \
  -p external_repo:=external/Efficient-SAM2 \
  -p checkpoint_path:=checkpoints/efficient-sam2/sam2.1_hiera_tiny.pt \
  -p model_config:=configs/sam2.1/sam2.1_hiera_t.yaml \
  -p device:=cuda \
  -p prompt_mode:=point \
  -p point_x:=0.5 \
  -p point_y:=0.5 \
  -p point_normalized:=true \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay
```

EfficientTAM-Ti:

```bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=efficienttam \
  -p external_repo:=external/EfficientTAM \
  -p checkpoint_path:=checkpoints/efficienttam/efficienttam_ti.pt \
  -p model_config:=configs/efficienttam/efficienttam_ti.yaml \
  -p device:=cuda \
  -p prompt_mode:=point \
  -p point_x:=0.5 \
  -p point_y:=0.5 \
  -p point_normalized:=true \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay
```

MobileSAM:

```bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=mobilesam \
  -p external_repo:=external/MobileSAM \
  -p checkpoint_path:=checkpoints/mobilesam/mobile_sam.pt \
  -p mobile_sam_model_type:=vit_t \
  -p device:=cuda \
  -p prompt_mode:=point \
  -p point_x:=0.5 \
  -p point_y:=0.5 \
  -p point_normalized:=true \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay
```

## 9. Read The ROS Profiling Output

Per-frame CSV:

```text
results/thor/ros_camera/<model>/results.csv
```

Summary CSV:

```text
results/thor/ros_camera/<model>/summary.csv
```

Important fields:

```text
latency_ms                 backend.predict() latency
callback_total_ms          full ROS callback including conversion and overlay publish
end_to_end_ms              image timestamp to result publish timestamp
image_encoder_ms
text_encoder_ms
prompt_encoder_ms
mask_decoder_ms
grounding_ms
detector_ms
memory_attention_ms
memory_encoder_ms
cuda_peak_allocated_mb
params_*
weight_*_bytes
```

Overlay MP4:

```text
overlays/thor/ros_camera/<model>/overlay.mp4
```

## 10. Benchmark Checklist

For each ROS camera run, record:

```text
git branch and commit
JetPack/L4T version
camera source: index or GStreamer pipeline
camera resolution and FPS
model ID and checkpoint path
prompt mode and prompt value
result CSV path
summary CSV path
overlay MP4 path
mean/p95 callback_total_ms
mean/p95 end_to_end_ms
CUDA peak memory
params_total and weight_total_bytes
```

If overlays are blank or point prompts are on the wrong object, adjust
`prompt`, `point_x`, or `point_y` and rerun. Do not treat a fast run with wrong
masks as a successful camera benchmark.
