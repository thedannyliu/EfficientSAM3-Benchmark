# Jetson Thor ROS Video Streaming Benchmark and Profiling

This guide runs the live ROS camera pipeline on Jetson Thor:

```text
camera_stream_node -> /image
/image -> sam_backend_node -> /sam/result_json
                         \
                          -> /sam/overlay
                          -> /segmentation_mask
                          -> /segmented_image
result_recorder_node -> CSV + summary CSV
overlay_video_recorder_node -> overlay MP4
live_viewer_node -> image with segmentation overlay, metrics panel on the right
```

Use this path after the offline benchmark works. The ROS numbers include model
latency plus callback and transport overhead.

For the first video-streaming demo, use a recorded video as the ROS frame
publisher. This is still a live ROS topic pipeline: the video file only replaces
the physical camera as the image source.

Supported Terminal B backends in this guide:

```text
SAM3 reference per-frame text segmentation:
  backend=sam3
  checkpoint_path=checkpoints/sam3/sam3.pt

SAM3 native clip tracking:
  node=sam3_native_clip_node
  checkpoint_path=checkpoints/sam3/sam3.pt
  prompt=monitor

MobileSAM live bbox-chain tracking:
  node=mobile_sam_interactive_node
  backend=mobilesam
  checkpoint_path=checkpoints/mobilesam/mobile_sam.pt

SAM1-H live bbox-chain tracking:
  node=mobile_sam_interactive_node
  backend=sam1
  checkpoint_path=checkpoints/mobilesam/sam_vit_h_4b8939.pth

Distilled RepViT-S EfficientSAM3:
  backend=efficientsam3
  checkpoint_path=checkpoints/efficient_sam3_repvit_s.pt
  inferred backbone_type=repvit
  inferred model_name=m0.9

InstinctSAM ViT-B text segmentation:
  backend=efficientsam3
  checkpoint_path=checkpoints/instinctsam/instinctsam_vitb_concept.pt
  backbone_type=vit_base
  model_name=base
  text_encoder_type=MobileCLIP-S1

YOLOE open-vocabulary segmentation:
  node=yoloe_text_backend_node
  weights=checkpoints/yoloe/yoloe-26m-seg.pt
  prompt=monitor
```

Camera-stream support matrix:

```text
MobileSAM:
  live interactive point prompt -> mask -> bbox -> next-frame box prompt
  overlay window shows FPS, backend latency, callback latency, and end-to-end latency

SAM1-H:
  same live interactive bbox-chain path as MobileSAM, with backend=sam1 and vit_h weights
  overlay window shows FPS, backend latency, callback latency, and end-to-end latency

SAM3:
  live ROS camera path uses per-frame text-prompt segmentation through sam_backend_node
  native video tracking uses sam3_native_clip_node, which first captures a fixed
  camera clip, materializes it as a frame folder, then starts the native SAM3
  tracking session
  use prompt for one noun phrase, or prompts for multiple nouns/phrases
```

For SAM3 multi-object text prompts, prefer comma-separated values:

```text
-p prompts:="cup,notebook,monitor"
```

Whitespace-separated values such as `cup notebook monitor` are also accepted
for one-word nouns, but comma separation is safer for multi-word phrases.

If you are already inside `~/EfficientSAM3-Benchmark`, skip repeated
`cd EfficientSAM3-Benchmark` lines in the command blocks.

## 0. Quick Demo Usage

Use this section when you want to run a live demo rather than the full benchmark
recording workflow. The flow is always:

```text
Terminal A: choose one image source
Terminal B: choose one model/backend
Terminal C: open viewer only for non-interactive models
```

All terminals should start from the same environment:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh
```

### Terminal A: Choose The Stream Source

Use a recorded video as a ROS image stream:

```bash
ros2 run sam_benchmark_ros video_stream_node --ros-args \
  -p video_path:=videos/test1.mov \
  -p image_topic:=/image \
  -p fps:=0.0 \
  -p playback_rate:=1.0 \
  -p frame_id:=video \
  -p resize_width:=640
```

Useful video stream controls:

```text
video_path       local video file to publish
fps              publish rate; use 0.0 to auto-use the video's source FPS
playback_rate    speed multiplier; use 0.5 for half-speed playback
resize_width     shrink or enlarge the frames before publishing
resize_height    alternative to resize_width
```

Use the RealSense RGB camera as the ROS image stream:

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=false \
  rgb_camera.color_profile:=1280x720x30
```

Then use this image topic for the camera commands:

```text
/camera/camera/color/image_raw
```

If your wrapper uses a different namespace, find it with:

```bash
ros2 topic list | grep color
```

### Terminal B: Choose The Demo Model

The commands below use `image_topic:=/image` for recorded video. For the
RealSense camera source, replace it with
`image_topic:=/camera/camera/color/image_raw`.

For **MobileSAM interactive point prompt tracking**, use the source topic from
Terminal A. Use `/image` for video stream or
`/camera/camera/color/image_raw` for RealSense:

```bash
ros2 run sam_benchmark_ros mobile_sam_interactive_node --ros-args \
  -p image_topic:=/image \
  -p backend:=mobilesam \
  -p checkpoint_path:=checkpoints/mobilesam/mobile_sam.pt \
  -p external_repo:=external/MobileSAM \
  -p device:=cuda \
  -p mobile_sam_model_type:=vit_t \
  -p display_max_width:=1600 \
  -p bbox_scale:=1.2 \
  -p record_overlay:=false \
  -p overlay_video_output:=overlays/ros/mobile_sam_demo.mp4 \
  -p overlay_video_preserve_timing:=true \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

For **SAM1-H interactive point prompt tracking**, use the same node with SAM1-H
weights:

```bash
ros2 run sam_benchmark_ros mobile_sam_interactive_node --ros-args \
  -p image_topic:=/image \
  -p backend:=sam1 \
  -p checkpoint_path:=checkpoints/mobilesam/sam_vit_h_4b8939.pth \
  -p external_repo:=external/MobileSAM \
  -p device:=cuda \
  -p mobile_sam_model_type:=vit_h \
  -p window_name:="SAM1-H ROS Demo" \
  -p display_max_width:=1600 \
  -p bbox_scale:=1.2 \
  -p record_overlay:=false \
  -p overlay_video_output:=overlays/ros/sam1_h_demo.mp4 \
  -p overlay_video_preserve_timing:=true \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

MobileSAM and SAM1-H controls:

```text
left click on the image    initialize or reset the point prompt
r                          clear tracking state
q or Esc                   exit
```

The clicked point is shown on the overlay. After the first point prompt, the
node uses the previous mask's bounding box as the next frame's box prompt.
`bbox_scale:=1.2` expands that next-frame box by about 20% around its center.
Set `record_overlay:=true` to save the overlay MP4 at `overlay_video_output`.
The recorder preserves ROS timestamp timing by default, so a slow backend will
not make the saved MP4 play 4x faster.

For **SAM3 text prompt per-frame segmentation**:

```bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=sam3 \
  -p external_repo:=external/sam3 \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p device:=cuda \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

For **SAM3 native clip tracking**, the node first captures a fixed clip from
the ROS image topic, writes frames to `frame_dir`, then runs SAM3's native video
tracking path:

```bash
ros2 run sam_benchmark_ros sam3_native_clip_node --ros-args \
  -p image_topic:=/image \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p external_repo:=external/sam3 \
  -p prompt:=monitor \
  -p clip_frames:=120 \
  -p frame_dir:=results/thor/ros_camera/sam3_native_clip/frames \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

For **EfficientSAM3 text prompt per-frame segmentation**:

```bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=efficientsam3 \
  -p external_repo:=external/efficientsam3 \
  -p checkpoint_path:=checkpoints/efficient_sam3_repvit_s.pt \
  -p device:=cuda \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

For **InstinctSAM ViT-B text prompt per-frame segmentation**:

```bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=efficientsam3 \
  -p external_repo:=external/efficientsam3 \
  -p checkpoint_path:=checkpoints/instinctsam/instinctsam_vitb_concept.pt \
  -p device:=cuda \
  -p backbone_type:=vit_base \
  -p model_name:=base \
  -p text_encoder_type:=MobileCLIP-S1 \
  -p text_encoder_context_length:=16 \
  -p text_encoder_pos_embed_table_size:=77 \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

For **YOLOE open-vocabulary segmentation**:

```bash
ros2 run sam_benchmark_ros yoloe_text_backend_node --ros-args \
  -p image_topic:=/image \
  -p weights:=checkpoints/yoloe/yoloe-26m-seg.pt \
  -p device:=cuda \
  -p prompt:=monitor \
  -p imgsz:=640 \
  -p conf:=0.25 \
  -p iou:=0.7 \
  -p max_det:=20 \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

### Terminal C: Viewer For Non-Interactive Models

Skip Terminal C for MobileSAM and SAM1-H because their node already opens the
interactive overlay window. For SAM3, EfficientSAM3, SAM3 native clip tracking,
and YOLOE, open the viewer:

```bash
ros2 run sam_benchmark_ros live_viewer_node --ros-args \
  -p image_topic:=/image \
  -p segmented_image_topic:=/segmented_image \
  -p result_topic:=/sam/result_json \
  -p display_max_width:=1600 \
  -p record_overlay:=false \
  -p overlay_video_output:=overlays/ros/live_viewer_demo.mp4 \
  -p overlay_video_preserve_timing:=true
```

The viewer shows the image with mask overlay on the left and profiling metrics
on the right, so metrics do not cover the object. Set `record_overlay:=true` to
save the overlay MP4. The viewer preserves ROS timestamp timing by default, so
if the model only produces 7.5 FPS of overlays from a 30 FPS stream, the saved
MP4 keeps the original duration by repeating frames as needed.

### Common Topic And Display Checks

```bash
ros2 topic hz /image
ros2 topic hz /segmentation_mask
ros2 topic hz /segmented_image
ros2 topic echo /sam/result_json --once
```

If a new video path, FPS, display width, or recording setting does not appear to
take effect, stop old nodes and restart from the rebuilt workspace:

```bash
pkill -f video_stream_node || true
pkill -f mobile_sam_interactive_node || true
pkill -f live_viewer_node || true

cd ros_ws
colcon build --symlink-install --packages-select sam_benchmark_ros
cd ..
source scripts/source_thor_ros_env.sh
```

## 1. Prepare The Same Environment As Offline

Start from `main`:

```bash
git clone git@github.com:thedannyliu/EfficientSAM3-Benchmark.git
cd EfficientSAM3-Benchmark
git fetch origin
git checkout main
```

Create the Thor venv and install Jetson-compatible PyTorch first. Follow
NVIDIA's current PyTorch for Jetson instructions:

```text
https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
```

Then install this repo. Use `--system-site-packages` so the venv can use the
ROS and Jetson OpenCV packages installed by APT.
The command block below assumes the ROS Jazzy packages in the next section are
already installed because `scripts/source_thor_ros_env.sh` sources
`/opt/ros/jazzy/setup.bash`.

```bash
python3 -m venv --system-site-packages ~/venvs/effisam3_venv_ros
export THOR_VENV=~/venvs/effisam3_venv_ros
export SAM3_SOURCE=~/efficientsam3/sam3
export THOR_ROS_SETUP=/opt/ros/jazzy/setup.bash
source scripts/source_thor_ros_env.sh

python -m pip install -U pip
python -m pip install "numpy>=1.26,<2" opencv-python-headless pillow pyyaml huggingface_hub
python -m pip install timm tqdm ftfy==6.1.1 regex iopath typing_extensions psutil
python -m pip install -e . --no-deps
```

Do not use `requirements.txt` on Thor unless you intentionally want to manage
PyTorch yourself; it pins the PACE CUDA PyTorch packages.

Use the same helper in every Thor terminal. If your paths differ, set them
before sourcing:

```bash
export THOR_VENV=/path/to/venv
export SAM3_SOURCE=/path/to/efficientsam3/sam3
export THOR_ROS_SETUP=/opt/ros/jazzy/setup.bash
source scripts/source_thor_ros_env.sh
```

Install model source repos and checkpoints:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh
bash scripts/setup_model_repos.sh
bash scripts/download_sam3_checkpoint.sh
bash scripts/download_efficientsam3_checkpoints.sh
bash scripts/download_instinctsam_vitb_checkpoint.sh
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
  ros-jazzy-realsense2-camera \
  ros-jazzy-realsense2-description \
  ros-jazzy-sensor-msgs \
  ros-jazzy-std-msgs \
  python3-opencv \
  python3-colcon-common-extensions
```

If the RealSense packages are not available from APT on Thor, build the
official `realsense-ros` wrapper from source in a separate ROS workspace and
source that workspace before this repo's ROS workspace.

Use the repo helper in every ROS terminal:

```bash
cd EfficientSAM3-Benchmark
export THOR_ROS_SETUP=/opt/ros/jazzy/setup.bash
export THOR_VENV=~/venvs/effisam3_venv_ros
export SAM3_SOURCE=~/efficientsam3/sam3
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
live_viewer_node
mobile_sam_interactive_node
sam3_native_clip_node
video_stream_node
sam_backend_node
result_recorder_node
overlay_video_recorder_node
yoloe_text_backend_node
```

## 4. Run The Video Streaming Demo

This demo shows one live OpenCV window:

```text
left: image with segmentation overlay     right: profiling metrics panel
```

It also publishes machine-readable and visual segmentation topics:

```text
/segmentation_mask   sensor_msgs/Image mono8
/segmented_image     sensor_msgs/Image rgb8
```

Terminal A, publish a recorded video into ROS:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros video_stream_node --ros-args \
  -p video_path:=videos/test1.mov \
  -p image_topic:=/image \
  -p fps:=0.0 \
  -p playback_rate:=1.0 \
  -p frame_id:=video \
  -p resize_width:=640
```

Use `resize_width` or `resize_height` to shrink large videos before they enter
the ROS stream. For example, a 1280x720 video with `resize_width:=640` shows as
about 640x360 plus the metrics panel in the MobileSAM window.
Use `fps:=0.0` for original video speed. Use `playback_rate:=0.5` for half
speed, or set an explicit `fps` only when you intentionally want to override the
source video's FPS.

Choose one Terminal B backend option.

Terminal B option 1, run SAM3 on each incoming ROS frame:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=sam3 \
  -p external_repo:=external/sam3 \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p device:=cuda \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

Terminal B option 1b, run SAM3 native tracking on a fixed camera/video clip.
Start Terminal C and the recorder terminals before this command if you want to
see and save every published tracking frame:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros sam3_native_clip_node --ros-args \
  -p image_topic:=/image \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p external_repo:=external/sam3 \
  -p prompt:=monitor \
  -p clip_frames:=120 \
  -p frame_dir:=results/thor/ros_camera/sam3_native_clip/frames \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

This node first captures `clip_frames` frames, then starts SAM3 native video
tracking on the materialized frame folder. The reported end-to-end latency
therefore includes capture time plus native tracking time; use it separately
from the per-frame live SAM3 numbers above.

Terminal B option 2, run the distilled RepViT-S EfficientSAM3 checkpoint on the
same incoming ROS frames:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=efficientsam3 \
  -p external_repo:=external/efficientsam3 \
  -p checkpoint_path:=checkpoints/efficient_sam3_repvit_s.pt \
  -p device:=cuda \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

The backend infers `backbone_type:=repvit` and `model_name:=m0.9` from the
`efficient_sam3_repvit_s.pt` filename. Use either the SAM3 command or this
RepViT-S command for Terminal B, not both at the same time.

Terminal B option 3, run InstinctSAM ViT-B text-prompt segmentation on the same
incoming ROS frames. The checkpoint is assembled from
`GM717/InstinctSAM-ViT-B` trunk weights plus local SAM3 heads by
`scripts/download_instinctsam_vitb_checkpoint.sh`:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=efficientsam3 \
  -p external_repo:=external/efficientsam3 \
  -p checkpoint_path:=checkpoints/instinctsam/instinctsam_vitb_concept.pt \
  -p device:=cuda \
  -p backbone_type:=vit_base \
  -p model_name:=base \
  -p text_encoder_type:=MobileCLIP-S1 \
  -p text_encoder_context_length:=16 \
  -p text_encoder_pos_embed_table_size:=77 \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

Use either the SAM3, RepViT-S, or InstinctSAM command for Terminal B, not more
than one at the same time.

Terminal B option 4, run interactive MobileSAM bbox-chain tracking on the same
incoming ROS video frames:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros mobile_sam_interactive_node --ros-args \
  -p image_topic:=/image \
  -p backend:=mobilesam \
  -p checkpoint_path:=checkpoints/mobilesam/mobile_sam.pt \
  -p external_repo:=external/MobileSAM \
  -p device:=cuda \
  -p mobile_sam_model_type:=vit_t \
  -p display_max_width:=1600 \
  -p bbox_scale:=1.2 \
  -p record_overlay:=false \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

For MobileSAM, click the image to initialize or reset the point prompt; clicks
on the profiling panel are ignored. Later frames use the previous mask bounding
box as the next box prompt. Press `r` to reset tracking, or `q`/`Esc` to exit.
The clicked point is shown as a persistent marker on the overlay until reset or
until another point is clicked.
The next-frame box prompt defaults to `bbox_scale:=1.2`, which expands the
mask-derived box by about 20% around its center before passing it to the next
frame.
Use `display_max_width` to cap the full window width, or `display_scale` to set
a fixed display ratio such as `0.5`. Set `record_overlay:=true` and optionally
`overlay_video_output:=overlays/ros/mobile_sam_demo.mp4` to save the overlay
video directly from the interactive node.

Terminal B option 5, run SAM1-H bbox-chain tracking with the same interactive
node:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros mobile_sam_interactive_node --ros-args \
  -p image_topic:=/image \
  -p backend:=sam1 \
  -p checkpoint_path:=checkpoints/mobilesam/sam_vit_h_4b8939.pth \
  -p external_repo:=external/MobileSAM \
  -p device:=cuda \
  -p mobile_sam_model_type:=vit_h \
  -p window_name:="SAM1-H ROS Video" \
  -p bbox_scale:=1.2 \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

For SAM1-H, click the image to initialize the point prompt. Later frames use the
previous mask bounding box expanded by `bbox_scale:=1.2` as the next box prompt.
Press `r` to reset tracking, or `q`/`Esc` to exit.

Terminal B option 6, run YOLOE open-vocabulary segmentation with a text prompt
on the same incoming ROS frames:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros yoloe_text_backend_node --ros-args \
  -p image_topic:=/image \
  -p weights:=checkpoints/yoloe/yoloe-26m-seg.pt \
  -p device:=cuda \
  -p prompt:=monitor \
  -p imgsz:=640 \
  -p conf:=0.25 \
  -p iou:=0.7 \
  -p max_det:=20 \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

YOLOE is the text-prompt YOLO path in this repo. It runs per-frame
open-vocabulary segmentation, not video tracking.

For SAM3 per-frame, SAM3 native clip tracking, RepViT-S, or YOLOE, Terminal C
opens the live overlay viewer:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros live_viewer_node --ros-args \
  -p image_topic:=/image \
  -p segmented_image_topic:=/segmented_image \
  -p result_topic:=/sam/result_json \
  -p display_max_width:=1600 \
  -p record_overlay:=false
```

Skip Terminal C when using MobileSAM or SAM1-H because
`mobile_sam_interactive_node` already opens the interactive overlay window.
The viewer shows FPS, per-frame backend latency, callback/end-to-end latency,
CUDA memory, and Jetson GPU utilization in the right-side panel when
`tegrastats` is available. Set `record_overlay:=true` and optionally
`overlay_video_output:=overlays/ros/live_viewer_demo.mp4` to save the overlay
video directly from the viewer. Press `q` or `Esc` in the viewer window to
close it.

Verify the output topics:

```bash
ros2 topic hz /image
ros2 topic hz /segmentation_mask
ros2 topic hz /segmented_image
ros2 topic echo /sam/result_json --once
```

Use `videos/test2.mov` or another local video path by changing
`video_path:=...`.

## 5. Run SAM3, EfficientSAM3, YOLOE, MobileSAM, Or SAM1-H RealSense Stream

Use this path for the Intel RealSense D455f hardware demo. The D455f is used as
an RGB ROS camera source in v1; depth is intentionally disabled.

Connect the camera through a USB3 port and check that Thor sees it:

```bash
lsusb | grep -i realsense || true
dmesg | tail -n 50
```

Terminal A, start the official RealSense ROS wrapper with RGB enabled:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=false \
  rgb_camera.color_profile:=1280x720x30
```

Verify the RGB topic. If your wrapper uses a different namespace, use the topic
reported by `ros2 topic list | grep color`.

```bash
ros2 topic list | grep color
ros2 topic hz /camera/camera/color/image_raw
```

Choose one Terminal B backend option.

Terminal B option 1, run SAM3 text-prompt segmentation on the RealSense RGB
stream:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=sam3 \
  -p external_repo:=external/sam3 \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p device:=cuda \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p image_topic:=/camera/camera/color/image_raw \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

Terminal B option 1b, run SAM3 native tracking on a fixed RealSense RGB clip.
Start Terminal C and the recorder terminals before this command if you want to
see and save every published tracking frame:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros sam3_native_clip_node --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p external_repo:=external/sam3 \
  -p prompt:=monitor \
  -p clip_frames:=120 \
  -p frame_dir:=results/thor/ros_camera/sam3_native_clip/frames \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

This is SAM3's native tracking mode on a materialized camera clip. It is not an
unbounded online tracker; the upstream SAM3 predictor starts from a video or
frame folder, so this node captures the clip first and publishes tracking
results after propagation begins.

Terminal B option 2, run EfficientSAM3 text-prompt segmentation on the same
RealSense RGB stream:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

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
  -p prompt:=monitor \
  -p image_topic:=/camera/camera/color/image_raw \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

Terminal B option 3, run YOLOE open-vocabulary segmentation with a text prompt
on the RealSense RGB stream:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros yoloe_text_backend_node --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p weights:=checkpoints/yoloe/yoloe-26m-seg.pt \
  -p device:=cuda \
  -p prompt:=monitor \
  -p imgsz:=640 \
  -p conf:=0.25 \
  -p iou:=0.7 \
  -p max_det:=20 \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

For SAM3 per-frame, SAM3 native clip tracking, EfficientSAM3, or YOLOE,
Terminal C opens the live overlay viewer:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros live_viewer_node --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p segmented_image_topic:=/segmented_image \
  -p result_topic:=/sam/result_json \
  -p display_max_width:=1600 \
  -p record_overlay:=false
```

Terminal B option 4, run interactive MobileSAM bbox-chain tracking:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros mobile_sam_interactive_node --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p backend:=mobilesam \
  -p checkpoint_path:=checkpoints/mobilesam/mobile_sam.pt \
  -p external_repo:=external/MobileSAM \
  -p device:=cuda \
  -p mobile_sam_model_type:=vit_t \
  -p display_max_width:=1600 \
  -p bbox_scale:=1.2 \
  -p record_overlay:=false \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

Terminal B option 5, run SAM1-H bbox-chain tracking:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros mobile_sam_interactive_node --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p backend:=sam1 \
  -p checkpoint_path:=checkpoints/mobilesam/sam_vit_h_4b8939.pth \
  -p external_repo:=external/MobileSAM \
  -p device:=cuda \
  -p mobile_sam_model_type:=vit_h \
  -p window_name:="SAM1-H RealSense" \
  -p bbox_scale:=1.2 \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

The MobileSAM/SAM1-H window shows its own interactive overlay view, so do not start
Terminal C when using these options:

```text
left: live RGB frame with mask overlay     right: profiling metrics panel
```

Controls:

```text
left click on the image: initialize or reset the point prompt
r: clear current tracking state
q or Esc: exit
```

Tracking behavior:

```text
first click -> point prompt
next frames -> previous mask bbox becomes the next box prompt
new click -> reset and track the clicked object
empty mask -> tracking lost until the next click
```

Verify the outputs:

```bash
ros2 topic hz /segmentation_mask
ros2 topic hz /segmented_image
ros2 topic echo /sam/result_json --once
```

## 6. Start The Camera Publisher

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

## 7. Run A Null Backend Smoke Test

Terminal B:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=null \
  -p device:=cpu \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
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
  -p preserve_timing:=true \
  -p max_frames:=100
```

Proceed to real models only after the null CSV and overlay MP4 are created.

## 8. Run SAM3 Text-Prompt Camera Benchmark

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
  -p prompt:=monitor \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
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
  -p preserve_timing:=true \
  -p max_frames:=300
```

## 9. Run SAM3 Native Clip Tracking Camera Benchmark

Stop the per-frame SAM3 backend. Keep the camera publisher running. Start the
recorder terminals before Terminal B because `sam3_native_clip_node` publishes
after it finishes capturing the clip.

Terminal C, record native tracking results:

```bash
mkdir -p results/thor/ros_camera/sam3_native_clip overlays/thor/ros_camera/sam3_native_clip

ros2 run sam_benchmark_ros result_recorder_node --ros-args \
  -p csv_output:=results/thor/ros_camera/sam3_native_clip/results.csv \
  -p summary_output:=results/thor/ros_camera/sam3_native_clip/summary.csv \
  -p max_messages:=120
```

Terminal D, record native tracking overlays:

```bash
ros2 run sam_benchmark_ros overlay_video_recorder_node --ros-args \
  -p overlay_topic:=/sam/overlay \
  -p video_output:=overlays/thor/ros_camera/sam3_native_clip/overlay.mp4 \
  -p fps:=30.0 \
  -p preserve_timing:=true \
  -p max_frames:=120
```

Terminal B, capture a 120-frame camera clip and run SAM3 native tracking:

```bash
ros2 run sam_benchmark_ros sam3_native_clip_node --ros-args \
  -p image_topic:=/image \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p external_repo:=external/sam3 \
  -p prompt:=monitor \
  -p clip_frames:=120 \
  -p frame_dir:=results/thor/ros_camera/sam3_native_clip/frames \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

This is the SAM3 native tracking path. Do not compare its end-to-end latency
directly against per-frame live backends unless you explicitly want the
capture-then-track delay included.

## 10. Run EfficientSAM3 Text-Prompt Camera Benchmark

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
  -p prompt:=monitor \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

Distilled RepViT-S image encoder checkpoint:

```bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=efficientsam3 \
  -p external_repo:=external/efficientsam3 \
  -p checkpoint_path:=checkpoints/efficient_sam3_repvit_s.pt \
  -p device:=cuda \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

The backend infers `backbone_type:=repvit` and `model_name:=m0.9` from the
`efficient_sam3_repvit_s.pt` filename. You can still pass those parameters
explicitly if you want the run command to show the architecture.

InstinctSAM ViT-B:

```bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=efficientsam3 \
  -p external_repo:=external/efficientsam3 \
  -p checkpoint_path:=checkpoints/instinctsam/instinctsam_vitb_concept.pt \
  -p device:=cuda \
  -p backbone_type:=vit_base \
  -p model_name:=base \
  -p text_encoder_type:=MobileCLIP-S1 \
  -p text_encoder_context_length:=16 \
  -p text_encoder_pos_embed_table_size:=77 \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

For InstinctSAM camera benchmark recording, use separate output folders:

```bash
mkdir -p results/thor/ros_camera/instinctsam_vitb overlays/thor/ros_camera/instinctsam_vitb

ros2 run sam_benchmark_ros result_recorder_node --ros-args \
  -p csv_output:=results/thor/ros_camera/instinctsam_vitb/results.csv \
  -p summary_output:=results/thor/ros_camera/instinctsam_vitb/summary.csv \
  -p max_messages:=300
```

For RepViT-S camera benchmark recording, use separate output folders:

```bash
mkdir -p results/thor/ros_camera/repvit_s overlays/thor/ros_camera/repvit_s

ros2 run sam_benchmark_ros result_recorder_node --ros-args \
  -p csv_output:=results/thor/ros_camera/repvit_s/results.csv \
  -p summary_output:=results/thor/ros_camera/repvit_s/summary.csv \
  -p max_messages:=300

ros2 run sam_benchmark_ros overlay_video_recorder_node --ros-args \
  -p overlay_topic:=/sam/overlay \
  -p video_output:=overlays/thor/ros_camera/repvit_s/overlay.mp4 \
  -p fps:=30.0 \
  -p preserve_timing:=true \
  -p max_frames:=300
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
results/thor/ros_camera/repvit_s/
overlays/thor/ros_camera/repvit_s/
results/thor/ros_camera/es3p1_weak_image_weak_text/
overlays/thor/ros_camera/es3p1_weak_image_weak_text/
```

## 11. Run Point-Prompt And Bbox-Chain Camera Benchmarks

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
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
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
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
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
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

MobileSAM bbox-chain tracking:

Terminal C, record results:

```bash
mkdir -p results/thor/ros_camera/mobilesam_bbox_chain overlays/thor/ros_camera/mobilesam_bbox_chain

ros2 run sam_benchmark_ros result_recorder_node --ros-args \
  -p csv_output:=results/thor/ros_camera/mobilesam_bbox_chain/results.csv \
  -p summary_output:=results/thor/ros_camera/mobilesam_bbox_chain/summary.csv \
  -p max_messages:=300
```

Terminal D, record overlays:

```bash
ros2 run sam_benchmark_ros overlay_video_recorder_node --ros-args \
  -p overlay_topic:=/sam/overlay \
  -p video_output:=overlays/thor/ros_camera/mobilesam_bbox_chain/overlay.mp4 \
  -p fps:=30.0 \
  -p preserve_timing:=true \
  -p max_frames:=300
```

Terminal B, run the bbox-chain node:

```bash
ros2 run sam_benchmark_ros mobile_sam_interactive_node --ros-args \
  -p image_topic:=/image \
  -p backend:=mobilesam \
  -p checkpoint_path:=checkpoints/mobilesam/mobile_sam.pt \
  -p external_repo:=external/MobileSAM \
  -p mobile_sam_model_type:=vit_t \
  -p device:=cuda \
  -p window_name:="MobileSAM Camera" \
  -p bbox_scale:=1.2 \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

SAM1-H bbox-chain tracking:

Terminal C, record results:

```bash
mkdir -p results/thor/ros_camera/sam1_h_bbox_chain overlays/thor/ros_camera/sam1_h_bbox_chain

ros2 run sam_benchmark_ros result_recorder_node --ros-args \
  -p csv_output:=results/thor/ros_camera/sam1_h_bbox_chain/results.csv \
  -p summary_output:=results/thor/ros_camera/sam1_h_bbox_chain/summary.csv \
  -p max_messages:=300
```

Terminal D, record overlays:

```bash
ros2 run sam_benchmark_ros overlay_video_recorder_node --ros-args \
  -p overlay_topic:=/sam/overlay \
  -p video_output:=overlays/thor/ros_camera/sam1_h_bbox_chain/overlay.mp4 \
  -p fps:=30.0 \
  -p preserve_timing:=true \
  -p max_frames:=300
```

Terminal B, run the bbox-chain node:

```bash
ros2 run sam_benchmark_ros mobile_sam_interactive_node --ros-args \
  -p image_topic:=/image \
  -p backend:=sam1 \
  -p checkpoint_path:=checkpoints/mobilesam/sam_vit_h_4b8939.pth \
  -p external_repo:=external/MobileSAM \
  -p mobile_sam_model_type:=vit_h \
  -p device:=cuda \
  -p window_name:="SAM1-H Camera" \
  -p bbox_scale:=1.2 \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

For MobileSAM and SAM1-H, click the left side of the model window once to
initialize tracking. The node records the first prompt as a point prompt and
uses the previous predicted mask bbox as the next frame's box prompt.

## 12. Read The ROS Profiling Output

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
tracking_fps               rolling publish/tracking FPS when emitted by the backend node
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

Summary CSV also reports `mean_latency_fps`, `mean_callback_fps`,
`mean_end_to_end_fps`, and `mean_tracking_fps` when the source rows contain the
needed timing fields.

Overlay MP4:

```text
overlays/thor/ros_camera/<model>/overlay.mp4
```

## 13. Benchmark Checklist

For each ROS camera run, record:

```text
git branch and commit
JetPack/L4T version
source type: recorded video, camera index, or GStreamer pipeline
source path/index/pipeline, resolution, and FPS
model ID and checkpoint path
backend, backbone_type, and model_name
prompt mode and prompt value
result CSV path
summary CSV path
overlay MP4 path
mean/p95 callback_total_ms
mean/p95 end_to_end_ms
mean_callback_fps, mean_end_to_end_fps, and mean_tracking_fps when present
CUDA peak memory
params_total and weight_total_bytes
```

If overlays are blank or point prompts are on the wrong object, adjust
`prompt`, `point_x`, or `point_y` and rerun. Do not treat a fast run with wrong
masks as a successful camera benchmark.
