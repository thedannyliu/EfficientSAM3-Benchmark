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

Every image-segmentation and video-tracking node in this guide can be recorded
through `/sam/overlay`. Start this recorder in another terminal; it uses the
source ROS timestamps to preserve the live camera speed even when inference
drops frames:

```bash
source scripts/source_thor_ros_env.sh
mkdir -p overlays/thor/ros_camera/live

ros2 run sam_benchmark_ros overlay_video_recorder_node --ros-args \
  -p overlay_topic:=/sam/overlay \
  -p video_output:=overlays/thor/ros_camera/live/overlay.mp4 \
  -p fps:=30.0 \
  -p preserve_timing:=true
```

Stop the recorder with `Ctrl-C` so the MP4 is finalized. Set `fps` to the
camera's configured output rate (for example `30.0` for `640x480x30`). Point
and box prompt markers default to 0.5 seconds on interactive model overlays;
override this with `-p prompt_display_seconds:=0.5` on the model node.

Use this path after the offline benchmark works. The ROS numbers include model
latency plus callback and transport overhead.

For the first video-streaming demo, use a recorded video as the ROS frame
publisher. This is still a live ROS topic pipeline: the video file only replaces
the physical camera as the image source.

## Code-Audited Camera Tracking Status

As of 2026-07-14, the checked-in ROS camera code has these behaviors:

| family | ROS camera node | first prompt UI | subsequent tracking | status |
| --- | --- | --- | --- | --- |
| MobileSAM | `mobile_sam_interactive_node` | click point or left-button drag box in the OpenCV window | previous mask bbox becomes the next frame box prompt | implemented |
| SAM1 ViT-B/L/H | `mobile_sam_interactive_node` with `backend:=sam1` | click point or left-button drag box in the OpenCV window | previous mask bbox becomes the next frame box prompt | implemented |
| SAM2.1 / Efficient-SAM2.1 image backends | `sam_backend_node` | fixed parameter point only | independent per-frame image segmentation | implemented, not memory tracking |
| SAM2.1 online memory tracking | `sam2_online_tracking_node` | each point click or left-button drag adds an object | tracks multiple objects on each incoming ROS frame with per-object IDs and memory encoder updates | implemented for online camera streams |
| SAM2.1 native video memory tracking | `sam2_native_clip_node` | click point or left-button drag box in the OpenCV window | captures a bounded clip, then runs `SAM2VideoPredictor.init_state` + `add_new_points_or_box` + `propagate_in_video` | implemented for bounded clips |
| EdgeTAM native video memory tracking | `sam2_online_tracking_node` or `sam2_native_clip_node` with `external_repo:=external/EdgeTAM` | each point click or left-button drag adds an object | online node supports multiple objects by re-anchoring existing masks when the EdgeTAM object batch grows; clip node runs bounded tracking | implemented for online streams and bounded clips |
| SAM2.1 + distilled TinyViT/RepViT encoders | `sam2_online_tracking_node` or `sam2_native_clip_node` with `model_kind:=stage1-student` | each point click or left-button drag adds an object online | same multi-object SAM2 memory path, with `forward_image` patched to the selected Stage1 student encoder | implemented for online streams and bounded clips |
| SAM3 per-frame image backend | `sam_backend_node` | fixed text or fixed parameter point | independent per-frame image segmentation | implemented, not memory tracking |
| SAM3 online memory tracking | `sam3_online_tracking_node` | text prompt, click point, or left-button drag box in the OpenCV window | appends each incoming ROS frame to one persistent native SAM3 inference state and updates detector/tracker memory immediately | implemented for unbounded online camera streams |
| SAM3 native video memory tracking | `sam3_native_clip_node` | text prompt, click point, or left-button drag box in the OpenCV window | captures a bounded clip, then runs native `start_session` + `add_prompt` + `propagate_in_video` | implemented for bounded text/geometry clips |
| SAM3 geometry prompt tracking | `sam3_online_tracking_node` or `sam3_native_clip_node` with `prompt_mode:=interactive`, `point`, or `box` | click point or left-button drag box in the OpenCV window | online state or bounded SAM3 native video session, with geometry prompt data passed to `add_prompt` | implemented for online streams and bounded clips |

Do not use `sam_backend_node` runs as evidence of SAM2/SAM3 memory tracking.
Those rows are useful for per-frame image latency, but they do not carry SAM2 or
SAM3 video memory across frames. Use `sam2_online_tracking_node` for SAM2
online memory tracking on live ROS frames, `sam2_native_clip_node` for
SAM2/EdgeTAM bounded clip memory tracking, `sam3_online_tracking_node` for SAM3
online memory tracking, and `sam3_native_clip_node` for bounded SAM3 camera
clips. The SAM2 and SAM3 online nodes keep live in-memory predictor state
instead of materializing a JPEG frame folder. SAM2 adds each new point or box
as another tracked object. SAM3 starts a new memory session from the current
frame when its text, point, or box prompt changes. For SAM1-family models, the
intended camera tracking baseline is different: the first point or box prompt
creates a mask, then the previous mask bounding box is passed as the next frame
prompt.

### Thor File Layout For Camera Runs

Keep the camera files aligned with the offline benchmark layout:

```text
~/EfficientSAM3-Benchmark/
  checkpoints/
    sam3/sam3.pt
    sam2/sam2.1_hiera_tiny.pt
    sam2/sam2.1_hiera_small.pt
    sam2/sam2.1_hiera_base_plus.pt
    sam2/sam2.1_hiera_large.pt
    efficient-sam2/sam2.1_hiera_tiny.pt
    efficient-sam2/sam2.1_hiera_small.pt
    efficient-sam2/sam2.1_hiera_base_plus.pt
    efficient-sam2/sam2.1_hiera_large.pt
    edgetam/edgetam.pt
    efficienttam/efficienttam_ti.pt
    efficienttam/efficienttam_s.pt
    mobilesam/mobile_sam.pt
    mobilesam/sam_vit_b_01ec64.pth
    mobilesam/sam_vit_l_0b3195.pth
    mobilesam/sam_vit_h_4b8939.pth
    instinctsam/instinctsam_vitb_concept.pt
    sam2_distill/
      stage1/
        tv21m_mse.pt
        tv21m_mse_cos.pt
        tv21m_highres.pt
        tv11m_mse.pt
        tv11m_mse_cos.pt
        tv5m_mse.pt
        tv5m_mse_cos.pt
        tv5_proj_sam21l_msehr_cos025_best.pt
        repvit_m09_proj_sam21l_msehr_cos025_l1010_best.pt
      tinyvit/
        tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
        tiny_vit_11m_224.dist_in22k_ft_in1k.safetensors
        tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors
      repvit/
        repvit_m0_9.dist_450e_in1k.safetensors
  external/
    sam3/
    sam2/
    Efficient-SAM2/
    EfficientTAM/
    EdgeTAM/
    MobileSAM/
    SAM2-Distillation-Pipeline/
  results/thor/ros_camera/
  overlays/thor/ros_camera/
```

The `sam2_distill/` weights are required when `sam2_native_clip_node` runs with
`model_kind:=stage1-student`; in that mode the node loads a full SAM2/EdgeTAM
checkpoint for the prompt, mask, and memory modules, then patches only the
image encoder with the selected Stage1 TinyViT or RepViT checkpoint.

### Acceptance Criteria For Native Camera Memory Tracking

Before marking a camera stream row as SAM2/SAM3 memory tracking, verify all of
the following in code and in the CSV:

```text
initial prompt:
  OpenCV left click produces prompt_mode=point, or left-button drag/release produces prompt_mode=box

SAM2 native:
  for online camera tracking, first prompt builds an in-memory inference_state from the current ROS frame
  later online frames call _run_single_frame_inference with run_mem_encoder=True
  online result rows include stream_mode=native_online
  memory_history_size bounds old non-conditioning frame memory

SAM2 native bounded clip:
  node captures a bounded clip after the first prompt
  node initializes one video state from the captured JPEG frame folder
  first prompt calls add_new_points_or_box with either points/labels or box
  later frames come from propagate_in_video, not repeated sam_backend_node image calls
  result rows include stream_mode=native_video_clip

SAM2 distilled TinyViT:
  same SAM2 native path, but image encoder is replaced by the selected Stage1 TinyViT variant
  prompt encoder, mask decoder, and memory modules stay from SAM2.1-L or EdgeTAM as documented

SAM3 native geometry:
  node starts a SAM3 video session after the first point or box prompt
  add_prompt receives geometry prompt data, not only text
  later frames come from propagate_in_video
  result rows include stream_mode=native_clip and prompt_mode=point or box

SAM1/MobileSAM:
  first prompt may be point or box
  later frames intentionally use bbox_chain, not native memory
```

Supported Terminal B backends in this guide:

```text
SAM3 reference per-frame text segmentation:
  backend=sam3
  checkpoint_path=checkpoints/sam3/sam3.pt

SAM3 native clip tracking:
  node=sam3_native_clip_node
  checkpoint_path=checkpoints/sam3/sam3.pt
  prompt_mode=text, interactive, point, or box
  note=captures a bounded clip before native propagation

SAM2 native online tracking:
  node=sam2_online_tracking_node
  checkpoint_path=checkpoints/sam2/sam2.1_hiera_large.pt
  prompt_mode=interactive point or box from OpenCV mouse input
  note=tracks incoming ROS frames immediately after the prompt and updates SAM2 memory online

SAM2 native clip tracking:
  node=sam2_native_clip_node
  checkpoint_path=checkpoints/sam2/sam2.1_hiera_large.pt
  prompt_mode=interactive point or box from OpenCV mouse input
  note=captures a bounded clip before native propagation

MobileSAM live point/box bbox-chain tracking:
  node=mobile_sam_interactive_node
  backend=mobilesam
  checkpoint_path=checkpoints/mobilesam/mobile_sam.pt

SAM1-H live point/box bbox-chain tracking:
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
  live interactive click point or drag box -> mask -> bbox -> next-frame box prompt
  overlay window shows FPS, backend latency, callback latency, and end-to-end latency

SAM1-H:
  same live interactive bbox-chain path as MobileSAM, with backend=sam1 and vit_h weights
  overlay window shows FPS, backend latency, callback latency, and end-to-end latency

SAM2.1 / Efficient-SAM2.1:
  current ROS camera path is per-frame point-prompt image segmentation through sam_backend_node
  online native memory tracking uses sam2_online_tracking_node:
    click point or drag box -> initialize memory on current frame -> track each incoming frame
    memory_history_size bounds old non-conditioning frame outputs
  bounded native memory tracking uses sam2_native_clip_node:
    click point or drag box -> capture a bounded clip -> init_state/add_new_points_or_box/propagate_in_video
  stage1-student mode patches forward_image to the selected TinyViT or RepViT encoder before online tracking or propagation

SAM3:
  live ROS camera path uses per-frame text/point image segmentation through sam_backend_node
  current native video tracking uses sam3_native_clip_node, which first captures a fixed
  camera clip, materializes it as a frame folder, then starts the native SAM3
  tracking session with text, point, or box prompts
  prompt_mode=interactive enables click point or left-button drag box geometry prompts
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
  enable_depth:=false
```

Then use this image topic for the camera commands:

```text
/camera/camera/color/image_raw
```

If your wrapper uses a different namespace, find it with:

```bash
ros2 topic list | grep color
```

#### Adjust RealSense Resolution And Frame Rate

The `rgb_camera.color_profile` value has the form `WIDTHxHEIGHTxFPS`, but the
available combinations depend on the exact RealSense sensor and USB link. Do
not assume that common values such as `1280x720x30` or `640x480x60` are
supported. First start the driver without overriding the profile:

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=false
```

While that fallback/default camera node is running, inspect the exact values
accepted by this driver and the profile it selected:

```bash
ros2 param describe /camera/camera rgb_camera.color_profile
ros2 param get /camera/camera rgb_camera.color_profile
rs-enumerate-devices -s
```

Stop the camera node with `Ctrl-C`, choose one exact profile from the
`ros2 param describe` output, and relaunch. For example, use the following only
when `640x480x30` appears in that camera's supported list:

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=false \
  rgb_camera.color_profile:=640x480x30
```

An error such as the following means the requested combination is unsupported;
the final value is the profile selected by the driver, not the requested FPS:

```text
Given value, 1280x720x30 is invalid ... Setting ROS param back to: 640x480x15
```

In that example, use `640x480x15` immediately or select another value shown by
`ros2 param describe`. Do not continue reporting the run as 30 FPS after the
driver falls back to 15 FPS.

Also verify that the camera negotiated a USB 3 link. A `480M` entry is USB 2
and can remove higher-bandwidth profiles; `5000M` or higher is USB 3:

```bash
lsusb -t
```

If the RealSense row shows `480M`, reconnect it directly to a Thor USB 3 port
with a USB 3 data cable, avoid a USB 2 hub, restart the camera node, and query
the supported profiles again.

After launching the camera, verify the negotiated dimensions and actual ROS
publish rate instead of relying only on the requested profile:

```bash
ros2 topic echo --once /camera/camera/color/camera_info | grep -E 'width:|height:'
ros2 topic hz /camera/camera/color/image_raw
```

For fast cup motion, prioritize the highest supported FPS profile and strong
lighting over maximum resolution. Shorter exposure reduces motion blur, but
requires more light. Inspect the exact exposure, gain, and white-balance
parameter names and accepted ranges exposed by the installed RealSense wrapper:

```bash
ros2 param list /camera/camera | grep -E 'exposure|gain|white_balance'
ros2 param describe /camera/camera rgb_camera.exposure
ros2 param get /camera/camera rgb_camera.enable_auto_exposure
```

To use manual exposure, first disable auto exposure and then choose a value
inside the range reported by `ros2 param describe`:

```bash
ros2 param set /camera/camera rgb_camera.enable_auto_exposure false
EXPOSURE_VALUE=100  # Example only; replace with a supported value for this camera.
ros2 param set /camera/camera rgb_camera.exposure "${EXPOSURE_VALUE}"
```

The raw ROS image topic has no JPEG quality setting. Spatial image quality is
controlled by the camera profile, focus/lighting/exposure, and sensor itself.
For SAM2-family nodes, `display_scale` and `display_max_width` change only the
OpenCV window size; they do not change the camera image or model input. SAM2
internally resizes its image input to the predictor image size, normally 1024,
so very large camera profiles primarily increase ROS transport, overlay, and
display cost after that point.

Camera FPS and tracking FPS are different. Even if the camera supports 60 FPS,
a model with 450 ms latency can process only about 2.2 tracking updates per
second. Keep `input_queue_size:=1` for live use so the online tracker receives
the newest available camera frame instead of building a stale queue. Read the
actual tracking FPS and latency from the model window's metrics panel.

### Terminal B: Choose The Demo Model

The commands below use `image_topic:=/image` for recorded video. For the
RealSense camera source, replace it with
`image_topic:=/camera/camera/color/image_raw`.

For **MobileSAM interactive point/box prompt tracking**, use the source topic from
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

For **SAM1-H interactive point/box prompt tracking**, use the same node with SAM1-H
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
left click on the image             initialize or reset with a point prompt
left-button drag, then release      initialize or reset with a box prompt
r                                  clear tracking state
q or Esc                           exit
```

The clicked point or dragged box is shown on the overlay. After the first point
or box prompt, the node uses the previous mask's bounding box as the next
frame's box prompt.
`bbox_scale:=1.2` expands that next-frame box by about 20% around its center.
`box_drag_min_pixels:=5.0` controls the minimum drag width and height before a
mouse gesture is treated as a box instead of a point click.
Set `record_overlay:=true` to save the overlay MP4 at `overlay_video_output`.
The recorder preserves ROS timestamp timing by default, so a slow backend will
not make the saved MP4 play 4x faster.
Point and box markers disappear after `prompt_display_seconds` (default 0.5)
while the mask and tracking state remain visible.

### SAM2 Family Online Camera Memory Tracking

The following commands are the Thor RealSense quick start for official
SAM2.1-L, SAM2.1-L with the distilled TinyViT-21M or TinyViT-5M image encoder,
and official EdgeTAM. They all use `sam2_online_tracking_node`, process incoming
camera frames immediately, and run the native SAM2/EdgeTAM memory encoder on
every tracked frame. They are not independent per-frame image segmentation.

After pulling a version that changes the ROS node, rebuild and re-source the
workspace before testing. Otherwise `ros2 run` can continue executing the old
installed node:

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh
cd ros_ws
colcon build --symlink-install --packages-select sam_benchmark_ros
cd ..
source scripts/source_thor_ros_env.sh
```

Start the camera in Terminal A:

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=false
```

Before starting a model, verify the stream and required files in another
terminal:

```bash
set -e
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 topic hz /camera/camera/color/image_raw

test -f checkpoints/sam2/sam2.1_hiera_large.pt
test -f checkpoints/sam2_distill/stage1/tv21m_mse_cos.pt
test -f checkpoints/sam2_distill/stage1/tv5_proj_sam21l_msehr_cos025_best.pt
test -f checkpoints/sam2_distill/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
test -f checkpoints/sam2_distill/tinyvit/tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors
test -f checkpoints/edgetam/edgetam.pt
test -d external/sam2
test -d external/EdgeTAM
test -d external/SAM2-Distillation-Pipeline
echo "camera model files found"
```

Run exactly one of the following model commands in Terminal B. Each command
uses the RealSense `best_effort` image QoS and opens its own interactive OpenCV
window, so do not start `live_viewer_node` for these runs.

#### Official SAM2.1-L

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros sam2_online_tracking_node --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p external_repo:=external/sam2 \
  -p model_kind:=sam2 \
  -p checkpoint_path:=checkpoints/sam2/sam2.1_hiera_large.pt \
  -p model_config:=configs/sam2.1/sam2.1_hiera_l.yaml \
  -p device:=cuda \
  -p input_queue_size:=1 \
  -p image_qos_reliability:=best_effort \
  -p memory_history_size:=32 \
  -p enable_display:=true \
  -p display_scale:=1.0 \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay \
  -p window_name:="SAM2.1-L Online Memory"
```

#### SAM2.1-L With TinyViT-21M

This command uses the `tv21m_mse_cos` Stage1 student. The TinyViT image encoder
is replaced, while the prompt encoder, mask decoder, object pointers, memory
attention, and memory encoder remain from SAM2.1-L.

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros sam2_online_tracking_node --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p external_repo:=external/sam2 \
  -p sam2_distill_root:=external/SAM2-Distillation-Pipeline \
  -p model_kind:=stage1-student \
  -p checkpoint_path:=checkpoints/sam2_distill/stage1/tv21m_mse_cos.pt \
  -p sam2_checkpoint_path:=checkpoints/sam2/sam2.1_hiera_large.pt \
  -p student_family:=tinyvit \
  -p student_model_name:=tiny_vit_21m_512.dist_in22k_ft_in1k \
  -p student_backbone_checkpoint:=checkpoints/sam2_distill/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors \
  -p student_adapter_mode:=auto \
  -p model_config:=configs/sam2.1/sam2.1_hiera_l.yaml \
  -p device:=cuda \
  -p input_queue_size:=1 \
  -p image_qos_reliability:=best_effort \
  -p memory_history_size:=32 \
  -p enable_display:=true \
  -p display_scale:=1.0 \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay \
  -p window_name:="SAM2.1-L TinyViT-21M Online Memory"
```

#### SAM2.1-L With TinyViT-5M Projection

This command uses the TinyViT-5M projection-only Stage1 checkpoint currently
used by the Thor camera and multi-object video demos.

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros sam2_online_tracking_node --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p external_repo:=external/sam2 \
  -p sam2_distill_root:=external/SAM2-Distillation-Pipeline \
  -p model_kind:=stage1-student \
  -p checkpoint_path:=checkpoints/sam2_distill/stage1/tv5_proj_sam21l_msehr_cos025_best.pt \
  -p sam2_checkpoint_path:=checkpoints/sam2/sam2.1_hiera_large.pt \
  -p student_family:=tinyvit \
  -p student_model_name:=tiny_vit_5m_224.dist_in22k_ft_in1k \
  -p student_backbone_checkpoint:=checkpoints/sam2_distill/tinyvit/tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors \
  -p student_adapter_mode:=projection \
  -p model_config:=configs/sam2.1/sam2.1_hiera_l.yaml \
  -p device:=cuda \
  -p input_queue_size:=1 \
  -p image_qos_reliability:=best_effort \
  -p memory_history_size:=32 \
  -p enable_display:=true \
  -p display_scale:=1.0 \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay \
  -p window_name:="SAM2.1-L TinyViT-5M Projection Online Memory"
```

#### Official EdgeTAM

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros sam2_online_tracking_node --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p external_repo:=external/EdgeTAM \
  -p model_kind:=sam2 \
  -p checkpoint_path:=checkpoints/edgetam/edgetam.pt \
  -p model_config:=configs/edgetam.yaml \
  -p device:=cuda \
  -p input_queue_size:=1 \
  -p image_qos_reliability:=best_effort \
  -p memory_history_size:=32 \
  -p enable_display:=true \
  -p display_scale:=1.0 \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay \
  -p window_name:="EdgeTAM Online Memory"
```

Interactive controls are identical for all four models:

```text
left click                          add one point-prompt object
left-button drag, then release     add one box-prompt object
r                                   clear every object and memory state
q or Esc                            exit
```

Every click or drag creates a new object ID. The point or box marker disappears
after 0.5 seconds, while its mask, ID label, and memory tracking remain active.
When a slow model is already processing a frame, a new point, box, or reset is
queued and applied as soon as that inference finishes instead of being dropped.
The right-side metrics panel reports tracking FPS and per-frame latency. The
node drops queued camera frames when inference is slower than the camera; using
`input_queue_size:=1` therefore measures the latest available frame instead of
building an increasingly stale frame backlog.

Official SAM2.1-L and both TinyViT variants retain existing object memory when
another object is added. EdgeTAM cannot increase its object batch after
propagation begins, so adding an EdgeTAM object re-anchors all existing objects
from their latest masks and restarts its memory history on that camera frame.

If the window is blank or the desktop reports that it is not responding, first
confirm `ros2 topic hz /camera/camera/color/image_raw` produces a rate, then
check that Terminal A and Terminal B have the same `ROS_DOMAIN_ID`. Also verify
that `DISPLAY` is set and keep `image_qos_reliability:=best_effort` for the
RealSense publisher. Watch the terminal while the selected checkpoint loads.
After loading, the current node opens a window displaying
`Waiting for /camera/camera/color/image_raw` when no camera frame has arrived. A
live image with `Click point or drag box` confirms that the model and RealSense
stream are connected. The OpenCV event loop remains on the main thread while
ROS image inference runs in a background executor, so the window remains
responsive while a slow SAM2.1-L frame is running. If an updated checkout still
freezes after a prompt, rebuild and re-source the ROS workspace with the command
above before testing again.

## Interactive Multi-Object Video Demo

Use the bounded video demo when every source frame must be tracked and the
same selected first-frame prompts must be compared among official SAM2.1-L, the
TinyViT-21M MSE+cos student, the TinyViT-5M projection student, and official
EdgeTAM. This runner reads the video file directly;
do not start `video_stream_node`, `sam2_online_tracking_node`, or the ROS
overlay recorder for this workflow.

The first frame opens before any model runs. A click adds a point-prompt
object and a left-button drag adds a box-prompt object. Select between one and
255 objects, then press Enter or Space. `u` removes the last object, `r` clears
all objects, and `q` or Escape cancels. All four models reuse the same saved
prompts.
All four tracking passes finish and display their results before any MP4 encoding
starts. The final comparison window accepts Enter, Space, or `s` to save the
command defaults. Press `f` to save only the latency-removed source-FPS video,
`l` to save only the measured-latency video, `b` to save both, or `n`, `q`, or
Escape to discard videos. Before choosing a mode, `1`, `2`, `4`, and `8` set
the latency video's playback-speed multiplier. The source-FPS output always
stays at 1x and ignores this multiplier.

During any model's tracking pass, press `s` once to arm automatic saving;
press it again to disarm. An armed run skips the final menu and saves with the
command's `--timing-mode` and `--save-speed` values. `SAVE ARMED` appears in
the live window but is not burned into the output video. The live status also
shows a rolling display FPS calculated from the latest completed frames; this
can be lower than `1000 / latency_ms` because it includes mask storage,
overlay rendering, and display work.

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

python -m sam_backend.sam2_video_demo \
  --video-path videos/iphone16pro_demo.mov \
  --timing-mode both \
  --external-repo external/sam2 \
  --sam2-checkpoint checkpoints/sam2/sam2.1_hiera_large.pt \
  --sam2-distill-root external/SAM2-Distillation-Pipeline \
  --tv21-stage1-checkpoint checkpoints/sam2_distill/stage1/tv21m_mse_cos.pt \
  --tv21-backbone-checkpoint checkpoints/sam2_distill/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors \
  --tv5-stage1-checkpoint checkpoints/sam2_distill/stage1/tv5_proj_sam21l_msehr_cos025_best.pt \
  --tv5-backbone-checkpoint checkpoints/sam2_distill/tinyvit/tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors \
  --edgetam-external-repo external/EdgeTAM \
  --edgetam-checkpoint checkpoints/edgetam/edgetam.pt \
  --edgetam-model-config configs/edgetam.yaml \
  --model-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda \
  --sam2-save-speed 20 \
  --tv21-save-speed 10 \
  --tv5-save-speed 5 \
  --edgetam-save-speed 2 \
  --save-masks \
  --output-dir overlays/thor/video_demo/iphone16pro_multi_object
```

This example saves the latency-expanded SAM2.1-L, TinyViT-21M, TinyViT-5M,
and EdgeTAM outputs at 20x, 10x, 5x, and 2x respectively when `realtime` or
`both` is selected. Every `source_fps` output remains at the original 1x speed.

To preview only the TinyViT-5M projection student before running the full
four-model comparison:

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

VIDEO="$HOME/EfficientSAM3-Benchmark/videos/cup-1.MOV"
OUT="overlays/thor/video_demo/tv5m_$(date +%Y%m%d-%H%M%S)"

python -m sam_backend.sam2_video_demo \
  --video-path "${VIDEO}" \
  --models tv5m_projection \
  --timing-mode both \
  --save-speed 2 \
  --output-dir "${OUT}"
```

`--models` accepts `sam2p1_l`, `tv21m_mse_cos`, `tv5m_projection`, and
`official_edgetam`. Omit it to run all four models in that order. EdgeTAM uses
the same one-or-more first-frame point/box prompts and native memory propagation.
Use `--prompts-json PATH` to reuse a previous run's `prompts.json` and skip the
selection window. The coordinates must come from the same source video.

Verify the system encoder before choosing to save:

```bash
ffmpeg -version
```

If that command is missing on Thor, install the Ubuntu package once with
`sudo apt-get update && sudo apt-get install -y ffmpeg`.
The saver prefers `libx264`. If that encoder is absent from the Thor FFmpeg
build, it automatically checks `h264_nvmpi`, `h264_v4l2m2m`, `h264_nvenc`, and
finally high-quality `mpeg4`; hardware encoders use a resolution/FPS-scaled
bitrate instead of unsupported x264 CRF flags.
For iPhone MOV inputs, the saver probes every audio stream but maps only the
first stream with a real decoder such as AAC. Auxiliary streams whose reported
codec is `none` or `unknown` are excluded instead of causing FFmpeg to exit.
If the selected audio stream ends slightly before the video, FFmpeg pads it
with silence so `-shortest` cannot close the raw-video pipe before the final
overlay frame is written.

`--timing-mode` accepts:

| Value | Output behavior |
| --- | --- |
| `source_fps` | writes every tracked source frame exactly once at the source video's detected FPS; inference waiting is removed and source audio is retained when present |
| `realtime` | writes at the original nominal FPS, expands duration from synchronized per-frame inference latency, and applies the selected acceleration; smooth transition frames replace duplicate-frame holds, and audio is omitted because the duration changes |
| `both` | produces both outputs for all selected models |

`--save-speed` accepts any positive multiplier and applies only to `realtime`.
`2` halves the latency-expanded duration and `4` produces one quarter of that
duration while the encoded FPS stays equal to the source FPS. The encoder
interpolates adjacent tracked overlays across each compressed latency interval
so accelerated output does not consist of long duplicate-frame holds. These
are visual transition frames, not additional model predictions. The
latency-removed `source_fps` output always remains at 1x with original audio.
A non-1x latency filename includes the multiplier, for example
`sam2p1_l_realtime_2x.mp4`.
Use `--sam2-save-speed`, `--tv21-save-speed`, `--tv5-save-speed`, and
`--edgetam-save-speed` to override the fallback separately. When any per-model
override is present, the final save window locks the configured speed values
instead of accepting `1/2/4/8`.

The runner preserves the decoded source width, height, frame count, and average
source FPS. SAM2 inference frames are downscaled to at most 1024 pixels on the
long side because SAM2 internally evaluates at 1024, while masks are resized
back onto the original-resolution decoded frames. Output is high-quality H.264
at CRF 18 by default. Overlay rendering requires re-encoding and OpenCV uses an
8-bit BGR path, so iPhone HDR/Dolby Vision bit depth and codec metadata are not
bit-for-bit preserved.

The output directory always contains `prompts.json` and `summary.json`. For the
four-model command above, accepting the final `both` save choice also creates:

```text
sam2p1_l_source_fps.mp4
sam2p1_l_realtime_20x.mp4
tv21m_mse_cos_source_fps.mp4
tv21m_mse_cos_realtime_10x.mp4
tv5m_projection_source_fps.mp4
tv5m_projection_realtime_5x.mp4
official_edgetam_source_fps.mp4
official_edgetam_realtime_2x.mp4
```

Each model's `summary.json` record includes `frames`, `prompt_ms`,
`mean_latency_ms`, `p50_latency_ms`, `p95_latency_ms`, and
`mean_display_fps`. Tracking latency is measured around each native
`propagate_in_video` step with CUDA synchronization; display FPS covers the
complete result-to-result wall-clock pipeline.
The top-level `model_save_speeds` object records the effective realtime
acceleration used for every selected model.

### Saved Masks And Clean Postprocessing

Add `--save-masks` to the tracking command when later postprocessing is
required. Without this flag, per-frame masks remain temporary and are deleted
when the command exits. With it, each selected model produces:

```text
OUTPUT_DIR/
  masks/
    sam2p1_l/
      000000.png
      000001.png
      ...
      manifest.json
    tv21m_mse_cos/
      ...
    tv5m_projection/
      ...
    official_edgetam/
      ...
```

Each PNG is an original-video-resolution `uint8` label map. Pixel value `0` is
background and values `1..N` are the selected object IDs. These PNGs can take
substantial disk space for a long high-resolution video, so persistence is
opt-in. The postprocessor also accepts masks saved by older demo revisions at
the inference resolution; it resizes those label maps to the decoded source
frame with nearest-neighbor interpolation before compositing.

The standalone postprocessor needs only the source video and one mask
directory. It does not draw latency, FPS, object IDs, or model names. This
example displays the mask from 0.1 seconds in the future on the current video
frame at 20% opacity:

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

VIDEO="$HOME/EfficientSAM3-Benchmark/videos/LBJ-dunk.mp4"
RUN="overlays/thor/video_demo/four_models_with_masks"

python -m sam_backend.sam2_mask_postprocess \
  --video-path="${VIDEO}" \
  --mask-dir="${RUN}/masks/tv5m_projection" \
  --output-path="${RUN}/tv5m_prediction_lead_0p1s.mp4" \
  --lead-seconds=0.1 \
  --alpha=0.20
```

Use `--lead-frames=2` or `--lead-frames=3` instead of `--lead-seconds` for an
exact frame offset. The output keeps the source width, height, nominal FPS, and
audio. Near the end of the video, frames with no corresponding future mask are
written without an overlay.

To stack multiple future masks on each source frame as a motion-prediction
trail:

```bash
python -m sam_backend.sam2_mask_postprocess \
  --video-path="${VIDEO}" \
  --mask-dir="${OUT}/masks/sam2p1_l" \
  --output-path="${OUT}/soccer_motion_prediction.mp4" \
  --lead-seconds-list 0.05 0.10 0.15 0.20 0.25 \
  --alpha=0.20
```

The nearest prediction is drawn first and faintest. With five masks and
`--alpha=0.20`, opacity increases from `0.04` at 0.05 seconds to `0.20` at 0.25
seconds. The farthest prediction is drawn last so it remains dominant where
masks overlap. The resulting video still contains only the source frames,
source audio, and transparent masks; no latency or model metadata is drawn.
The separate `prompt_ms` covers initialization of all selected objects and is
not included in `mean_latency_ms`. Both the top-level summary and each model
record include `object_count`. `videos_saved` records the final save/discard
choice; `save_timing_mode`, `save_speed`, and `save_output_fps` record the
selected output mode, multiplier, and encoded nominal FPS. During inference,
compact one-channel object-ID masks are held in a temporary directory; they are
removed after deferred encoding or discard.

For **RepViT-M0.9 Stage1 encoder + SAM2.1-L online memory tracking**, place
the full distilled Stage1 `best.pt` and optional ImageNet initialization at the
paths shown in the file layout above. RepViT-M0.9 is the architecture name; the
backbone has about 5.5M parameters, not 0.9M parameters. Make sure the Thor
copy of `external/SAM2-Distillation-Pipeline` contains
`sam2_distill/models/repvit_adapter.py`. The pipeline's pinned downloader can
prepare the public ImageNet initialization directly on Thor:

```bash
python external/SAM2-Distillation-Pipeline/tools/data/download_repvit_pretrained.py \
  --out-root checkpoints/sam2_distill/repvit
```

Transfer the distilled M0.9 `best.pt` from the training run's
`repvit_m09_proj_sam21l_msehr_cos025_l1010/checkpoints/best.pt` to the Stage1
destination shown above, then run:

```bash
ros2 run sam_benchmark_ros sam2_online_tracking_node --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p external_repo:=external/sam2 \
  -p sam2_distill_root:=external/SAM2-Distillation-Pipeline \
  -p model_kind:=stage1-student \
  -p checkpoint_path:=checkpoints/sam2_distill/stage1/repvit_m09_proj_sam21l_msehr_cos025_l1010_best.pt \
  -p sam2_checkpoint_path:=checkpoints/sam2/sam2.1_hiera_large.pt \
  -p student_family:=repvit \
  -p student_model_name:=repvit_m0_9.dist_450e_in1k \
  -p student_backbone_checkpoint:=checkpoints/sam2_distill/repvit/repvit_m0_9.dist_450e_in1k.safetensors \
  -p student_adapter_mode:=projection \
  -p model_config:=configs/sam2.1/sam2.1_hiera_l.yaml \
  -p device:=cuda \
  -p input_queue_size:=3 \
  -p image_qos_reliability:=best_effort \
  -p memory_history_size:=32 \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay \
  -p window_name:="SAM2.1-L RepViT-M0.9 Online Memory"
```

The Stage1 training job stores the complete RepViT backbone and projection
heads in `best.pt`, so `student_backbone_checkpoint` can be omitted for that
checkpoint. Keeping the pinned safetensors path makes the initialization
source explicit and also supports projection-only checkpoints. The node still
uses SAM2.1-L prompt, mask, object-pointer, memory-attention, and memory-encoder
weights. Each click or drag adds an object to the same online SAM2 memory
state; this is not per-frame image segmentation.

Use `auto_start:=true` when you want a non-interactive smoke run from the first
incoming frame. With `auto_start`, `initial_point_x/y` can be normalized
coordinates such as `0.5,0.5`.

The same multi-object behavior documented in the SAM2 family online camera
quick start applies to the RepViT Stage1 command above. Overlays use a different
color and an `ID` label for each object. `/sam/result_json` and recorder CSV
rows include `object_count`, `object_ids`, and `prompt_object_id`.

For **SAM2.1 native point/box bounded clip memory tracking**, use the source
topic from Terminal A. Click the OpenCV window for a point prompt, or
left-button drag and release for a box prompt. The node captures `clip_frames`
frames after the prompt and then runs the official SAM2 memory path:

```bash
ros2 run sam_benchmark_ros sam2_native_clip_node --ros-args \
  -p image_topic:=/image \
  -p external_repo:=external/sam2 \
  -p checkpoint_path:=checkpoints/sam2/sam2.1_hiera_large.pt \
  -p model_config:=configs/sam2.1/sam2.1_hiera_l.yaml \
  -p device:=cuda \
  -p clip_frames:=120 \
  -p frame_dir:=results/thor/ros_camera/sam2p1_l_native_clip/frames \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

For **EdgeTAM native point/box memory tracking**, use the same node with the
EdgeTAM repo and config:

```bash
ros2 run sam_benchmark_ros sam2_native_clip_node --ros-args \
  -p image_topic:=/image \
  -p external_repo:=external/EdgeTAM \
  -p checkpoint_path:=checkpoints/edgetam/edgetam.pt \
  -p model_config:=configs/edgetam.yaml \
  -p device:=cuda \
  -p clip_frames:=120 \
  -p frame_dir:=results/thor/ros_camera/edgetam_native_clip/frames \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

For **Stage1 TinyViT encoder + SAM2 bounded clip memory tracking**, load a full
SAM2 checkpoint for the prompt/mask/memory modules and patch only the image
encoder:

```bash
ros2 run sam_benchmark_ros sam2_native_clip_node --ros-args \
  -p image_topic:=/image \
  -p external_repo:=external/sam2 \
  -p sam2_distill_root:=external/SAM2-Distillation-Pipeline \
  -p model_kind:=stage1-student \
  -p checkpoint_path:=checkpoints/sam2_distill/stage1/tv21m_mse_cos.pt \
  -p sam2_checkpoint_path:=checkpoints/sam2/sam2.1_hiera_large.pt \
  -p tinyvit_checkpoint:=checkpoints/sam2_distill/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors \
  -p tinyvit_model_name:=tiny_vit_21m_512.dist_in22k_ft_in1k \
  -p model_config:=configs/sam2.1/sam2.1_hiera_l.yaml \
  -p device:=cuda \
  -p clip_frames:=120 \
  -p frame_dir:=results/thor/ros_camera/tv21m_mse_cos_native_clip/frames \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

For **SAM3 text prompt per-frame segmentation**:

```bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=sam3 \
  -p external_repo:=external/sam3 \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p device:=cuda \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p text_prompt_topic:=/sam/text_prompt \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

Run `live_viewer_node` for the SAM3 image-segmentation UI. Focus its OpenCV
window, press `t`, type the new prompt, and press `Enter`; the prompt takes
effect on the next image frame. `Backspace` edits and `Esc` cancels text input.
The viewer must use the same text prompt topic as the backend:

```bash
source scripts/source_thor_ros_env.sh
ros2 run sam_benchmark_ros live_viewer_node --ros-args \
  -p image_topic:=/image \
  -p segmented_image_topic:=/segmented_image \
  -p result_topic:=/sam/result_json \
  -p text_prompt_topic:=/sam/text_prompt
```

For **SAM3 native text clip tracking**, the node first captures a fixed clip
from the ROS image topic, writes frames to `frame_dir`, then runs SAM3's native
video tracking path:

```bash
ros2 run sam_benchmark_ros sam3_native_clip_node --ros-args \
  -p image_topic:=/image \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p external_repo:=external/sam3 \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p text_prompt_topic:=/sam/text_prompt \
  -p clip_frames:=120 \
  -p frame_dir:=results/thor/ros_camera/sam3_native_clip/frames \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

The SAM3 tracking window accepts all prompt types in one interface: click for a
point, drag for a box, or press `t` and type a text prompt. A new text prompt
immediately discards the old partial clip and starts capture from the current
camera frame. If native propagation is already processing a completed clip,
the single-threaded node applies the queued prompt after it finishes.

For **SAM3 native point/box clip tracking**, use the same node with
`prompt_mode:=interactive`. Click the OpenCV window for a point prompt, or
left-button drag and release for a box prompt. After the prompt, the node
captures `clip_frames` frames and sends the geometry prompt through SAM3
`add_prompt` before `propagate_in_video`:

```bash
ros2 run sam_benchmark_ros sam3_native_clip_node --ros-args \
  -p image_topic:=/image \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p external_repo:=external/sam3 \
  -p prompt_mode:=interactive \
  -p clip_frames:=120 \
  -p frame_dir:=results/thor/ros_camera/sam3_geometry_native_clip/frames \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

Use `prompt_mode:=point` to force clicks only, or `prompt_mode:=box` to require
a drag box. Use `auto_start:=true` for a non-interactive point-prompt smoke run
from the first incoming frame.

For **EfficientSAM3 text prompt per-frame segmentation**:

```bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=efficientsam3 \
  -p external_repo:=external/efficientsam3 \
  -p checkpoint_path:=checkpoints/efficient_sam3_repvit_s.pt \
  -p device:=cuda \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p text_prompt_topic:=/sam/text_prompt \
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

Skip Terminal C for MobileSAM, SAM1-H, and SAM3 native clip tracking because
their nodes already open an interactive overlay window. For SAM3,
EfficientSAM3, and YOLOE, open the viewer:

```bash
ros2 run sam_benchmark_ros live_viewer_node --ros-args \
  -p image_topic:=/image \
  -p image_qos_reliability:=best_effort \
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

Before the first messages arrive, the window remains responsive and reports
whether it is waiting for the camera image or the segmented image. RealSense
normally publishes sensor images with best-effort QoS, so keep
`image_qos_reliability:=best_effort` on both `sam_backend_node` and
`live_viewer_node`. Use `reliable` only when the image publisher explicitly
offers reliable QoS.

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
sam2_online_tracking_node
sam3_native_clip_node
sam3_online_tracking_node
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

For MobileSAM, click the image to initialize or reset with a point prompt, or
left-button drag and release to initialize or reset with a box prompt. Clicks
and drags on the profiling panel are ignored. Later frames use the previous mask
bounding box as the next box prompt. Press `r` to reset tracking, or `q`/`Esc`
to exit. The clicked point or dragged box is shown as a persistent prompt marker
on the overlay until reset or until another prompt is selected.
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

For SAM1-H, click the image to initialize with a point prompt, or left-button
drag and release to initialize with a box prompt. Later frames use the previous
mask bounding box expanded by `bbox_scale:=1.2` as the next box prompt.
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
  enable_depth:=false
```

Verify the RGB topic. If your wrapper uses a different namespace, use the topic
reported by `ros2 topic list | grep color`. Use the supported-profile procedure
in `Adjust RealSense Resolution And Frame Rate` before adding a
`rgb_camera.color_profile` override; do not assume `1280x720x30` is available.

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

For SAM3 per-frame, EfficientSAM3, or YOLOE, Terminal C opens the live overlay
viewer. SAM3 native clip tracking has its own window:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros live_viewer_node --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p image_qos_reliability:=best_effort \
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
left click on the image: initialize or reset with a point prompt
left-button drag, then release: initialize or reset with a box prompt
r: clear current tracking state
q or Esc: exit
```

Tracking behavior:

```text
first click or drag -> point or box prompt
next frames -> previous mask bbox becomes the next box prompt
new click or drag -> reset and track the selected object
empty mask -> tracking lost until the next prompt
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
  -p text_prompt_topic:=/sam/text_prompt \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

Terminal C, open the interactive image-segmentation viewer:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

ros2 run sam_benchmark_ros live_viewer_node --ros-args \
  -p image_topic:=/image \
  -p segmented_image_topic:=/segmented_image \
  -p result_topic:=/sam/result_json \
  -p text_prompt_topic:=/sam/text_prompt
```

Focus the viewer and press `t` to edit the current prompt. Type the complete
English prompt and press `Enter`; `Backspace` edits and `Esc` cancels. The
backend uses the submitted prompt on its next processed image.

Terminals D and E, record metrics and overlays:

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

Terminal B, capture a 120-frame camera clip and run SAM3 native text tracking:

```bash
ros2 run sam_benchmark_ros sam3_native_clip_node --ros-args \
  -p image_topic:=/image \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p external_repo:=external/sam3 \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p text_prompt_topic:=/sam/text_prompt \
  -p clip_frames:=120 \
  -p frame_dir:=results/thor/ros_camera/sam3_native_clip/frames \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

The native tracking window uses one prompt interface:

| input | action |
| --- | --- |
| left click | start a new point-prompt tracking clip |
| left-button drag and release | start a new box-prompt tracking clip |
| `t`, type, `Enter` | start a new text-prompt tracking clip |
| `Backspace` | delete the last text character while editing |
| `Esc` | cancel text editing; outside editing, close the node |
| `r` | clear the current tracking state |
| `q` | close the node |

Submitting any new point, box, or text prompt replaces the previous target and
starts capture from the current camera frame. The initial
`prompt_mode:=text` only selects startup behavior; the window can switch among
all three prompt types afterward.

For SAM3 native point/box tracking in the same recorder setup, replace Terminal
B with:

```bash
ros2 run sam_benchmark_ros sam3_native_clip_node --ros-args \
  -p image_topic:=/image \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p external_repo:=external/sam3 \
  -p prompt_mode:=interactive \
  -p clip_frames:=120 \
  -p frame_dir:=results/thor/ros_camera/sam3_native_clip/frames \
  -p result_topic:=/sam/result_json \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image \
  -p overlay_topic:=/sam/overlay
```

This is the SAM3 native tracking path. The same window also accepts `t` for a
text prompt after starting with `prompt_mode:=interactive`. Do not compare its end-to-end
latency directly against per-frame live backends unless you explicitly want the
capture-then-track delay included.

## 10. Run EfficientSAM3 Text-Prompt Camera Benchmark

The `live_viewer_node` text interface from Section 8 also applies to every
text-prompt image backend in this section because `sam_backend_node` subscribes
to `/sam/text_prompt` by default.

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

InstinctSAM3 ViT-B:

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

For InstinctSAM3 camera benchmark recording, use separate output folders:

```bash
mkdir -p results/thor/ros_camera/instinctsam_vitb overlays/thor/ros_camera/instinctsam_vitb

ros2 run sam_benchmark_ros result_recorder_node --ros-args \
  -p csv_output:=results/thor/ros_camera/instinctsam_vitb/results.csv \
  -p summary_output:=results/thor/ros_camera/instinctsam_vitb/summary.csv \
  -p max_messages:=300

ros2 run sam_benchmark_ros overlay_video_recorder_node --ros-args \
  -p overlay_topic:=/sam/overlay \
  -p video_output:=overlays/thor/ros_camera/instinctsam_vitb/overlay.mp4 \
  -p fps:=30.0 \
  -p preserve_timing:=true \
  -p max_frames:=300
```

Use the same command with `image_topic:=/camera/camera/color/image_raw` for the
RealSense RGB stream. The output folder stays
`results/thor/ros_camera/instinctsam_vitb/` unless you intentionally want a
separate recorded-video versus camera split.

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

The SAM2.1, Efficient-SAM2.1, and EfficientTAM commands in this section use
`sam_backend_node`, so they are independent per-frame image segmentation checks.
They do not initialize SAM2 memory and should not be reported as SAM2
video-memory tracking. Use them only for ROS transport and point-prompt image
latency checks. For SAM2 online camera memory tracking, use
`sam2_online_tracking_node`; for bounded clip memory tracking, use
`sam2_native_clip_node`.

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
initialize tracking with a point prompt, or left-button drag and release to
initialize with a box prompt. The node uses the previous predicted mask bbox as
the next frame's box prompt.

## 12. InstinctSAM GIText-Large And Hiera-L Camera Comparison

These two InstinctSAM files are component checkpoints. Keep the official SAM3
checkpoint because it supplies the remaining SAM3 decoder and segmentation
heads:

```text
checkpoints/
|-- sam3/
|   `-- sam3.pt
`-- instinctsam/
    |-- gitext_large_v4.pt
    `-- hiera_large_concept_trunk.pt
```

The GitHub source is accessible over SSH and may be cloned for reference. The
camera backend below contains the required adapter and does not import this
clone at runtime:

```bash
git clone git@github.com:william-Dic/InstinctSAM.git external/InstinctSAM
```

Accept access to the gated `GM717/InstinctSAM-ViT-B` Hugging Face repository,
authenticate on Thor, and download both components:

```bash
source scripts/source_thor_ros_env.sh
huggingface-cli login
bash scripts/download_instinctsam_compressed_checkpoints.sh
```

After pulling this change, rebuild and source the ROS workspace:

```bash
source scripts/source_thor_ros_env.sh
cd ros_ws
colcon build --symlink-install --packages-select sam_benchmark_ros
cd ..
source scripts/source_thor_ros_env.sh
```

Keep the RealSense publisher running on
`/camera/camera/color/image_raw`. Run exactly one of the following backend
commands at a time.

Official SAM3 baseline:

```bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=sam3 \
  -p external_repo:=external/efficientsam3 \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p device:=cuda \
  -p autocast_dtype:=float16 \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p image_topic:=/camera/camera/color/image_raw \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

InstinctSAM LiteText with GIText-large and the official SAM3 vision encoder:

```bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=instinctsam \
  -p external_repo:=external/efficientsam3 \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p instinctsam_text_checkpoint:=checkpoints/instinctsam/gitext_large_v4.pt \
  -p device:=cuda \
  -p autocast_dtype:=float16 \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p image_topic:=/camera/camera/color/image_raw \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

InstinctSAM Hiera-L compressed vision with the official SAM3 text encoder:

```bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=instinctsam \
  -p external_repo:=external/efficientsam3 \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p instinctsam_vision_checkpoint:=checkpoints/instinctsam/hiera_large_concept_trunk.pt \
  -p device:=cuda \
  -p autocast_dtype:=float16 \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p image_topic:=/camera/camera/color/image_raw \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

Use the same UI for all three runs:

```bash
ros2 run sam_benchmark_ros live_viewer_node --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p segmented_image_topic:=/segmented_image \
  -p result_topic:=/sam/result_json \
  -p text_prompt_topic:=/sam/text_prompt \
  -p window_name:="InstinctSAM vs SAM3"
```

Enter a new text prompt in the viewer and press Enter. The backend receives it
through `/sam/text_prompt` without a restart. These three commands perform
independent per-frame text-prompt image segmentation and expose comparable
latency in the viewer. They are not SAM3 native memory-tracking runs.

### Online native memory tracking

`sam3_online_tracking_node` starts from the first available camera frame and
then runs one native SAM3 detector/tracker/memory update for every incoming
frame. It does not use `clip_frames`, does not write a frame directory, and
does not wait for a clip to finish before publishing masks.

Run the official SAM3 online baseline:

```bash
ros2 run sam_benchmark_ros sam3_online_tracking_node --ros-args \
  -p external_repo:=external/efficientsam3 \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p version:=sam3 \
  -p image_topic:=/camera/camera/color/image_raw \
  -p image_qos_reliability:=best_effort \
  -p input_queue_size:=1 \
  -p memory_history_size:=32 \
  -p nms_backend:=torch \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

Run InstinctSAM LiteText with GIText-large through the same online memory path:

```bash
ros2 run sam_benchmark_ros sam3_online_tracking_node --ros-args \
  -p external_repo:=external/efficientsam3 \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p version:=sam3 \
  -p instinctsam_text_checkpoint:=checkpoints/instinctsam/gitext_large_v4.pt \
  -p image_topic:=/camera/camera/color/image_raw \
  -p image_qos_reliability:=best_effort \
  -p input_queue_size:=1 \
  -p memory_history_size:=32 \
  -p nms_backend:=torch \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

Run InstinctSAM Hiera-L compressed vision through the same online memory path:

```bash
ros2 run sam_benchmark_ros sam3_online_tracking_node --ros-args \
  -p external_repo:=external/efficientsam3 \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p version:=sam3 \
  -p instinctsam_vision_checkpoint:=checkpoints/instinctsam/hiera_large_concept_trunk.pt \
  -p image_topic:=/camera/camera/color/image_raw \
  -p image_qos_reliability:=best_effort \
  -p input_queue_size:=1 \
  -p memory_history_size:=32 \
  -p nms_backend:=torch \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

The online node owns its OpenCV UI, so do not start `live_viewer_node` for
these commands. Text mode starts automatically with `prompt:=monitor`. In the
same window, press `t`, type a replacement text prompt, and press Enter; click
once for a point prompt; or hold the left button, drag, and release for a box
prompt. Each prompt change discards the previous SAM3 state and starts a new
native memory session from the current frame. Point and box markers disappear
after 0.5 seconds. Press `r` to reset and `q` to exit.

`input_queue_size:=1` prioritizes the newest RealSense frame when inference is
slower than the camera. `memory_history_size:=32` bounds retained raw frame
tensors and non-conditioning tracker outputs; SAM3's conditioning frames and
native memory selection remain intact.

### Bounded native clip tracking

For native tracking, `sam3_native_clip_node` first captures `clip_frames` live
camera frames, starts an official SAM3 video session, adds the text prompt on
frame 0, and obtains later masks from `propagate_in_video`. The SAM3 tracker,
memory, association, and decoder remain unchanged; only the selected detector
component is replaced. This is bounded live-camera clip tracking, not an
unbounded online session.

Run the GIText-large native tracker:

```bash
ros2 run sam_benchmark_ros sam3_native_clip_node --ros-args \
  -p external_repo:=external/efficientsam3 \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p version:=sam3 \
  -p instinctsam_text_checkpoint:=checkpoints/instinctsam/gitext_large_v4.pt \
  -p image_topic:=/camera/camera/color/image_raw \
  -p image_qos_reliability:=best_effort \
  -p nms_backend:=torch \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p clip_frames:=120 \
  -p frame_dir:=results/thor/ros_camera/instinctsam_gitext_native/frames \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

Run the Hiera-L compressed-vision native tracker:

```bash
ros2 run sam_benchmark_ros sam3_native_clip_node --ros-args \
  -p external_repo:=external/efficientsam3 \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p version:=sam3 \
  -p instinctsam_vision_checkpoint:=checkpoints/instinctsam/hiera_large_concept_trunk.pt \
  -p image_topic:=/camera/camera/color/image_raw \
  -p image_qos_reliability:=best_effort \
  -p nms_backend:=torch \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p clip_frames:=120 \
  -p frame_dir:=results/thor/ros_camera/instinctsam_hiera_l_native/frames \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

Run the official SAM3 native baseline through the same builder:

```bash
ros2 run sam_benchmark_ros sam3_native_clip_node --ros-args \
  -p external_repo:=external/efficientsam3 \
  -p checkpoint_path:=checkpoints/sam3/sam3.pt \
  -p version:=sam3 \
  -p image_topic:=/camera/camera/color/image_raw \
  -p image_qos_reliability:=best_effort \
  -p nms_backend:=torch \
  -p prompt_mode:=text \
  -p prompt:=monitor \
  -p clip_frames:=120 \
  -p frame_dir:=results/thor/ros_camera/sam3_native_baseline/frames \
  -p result_topic:=/sam/result_json \
  -p overlay_topic:=/sam/overlay \
  -p mask_topic:=/segmentation_mask \
  -p segmented_image_topic:=/segmented_image
```

The native node contains its own camera/tracking UI. Press `t`, type a new
prompt, and press Enter to discard the old clip state and capture a new clip
with the updated prompt. Do not run `live_viewer_node` for these commands.

Thor reports CUDA architecture `sm_110a`. Triton's bundled `ptxas` may reject
that target while compiling SAM3 mask NMS or connected components.
`nms_backend:=torch` uses the same score ordering and IoU suppression rule
through regular device-resident PyTorch operations, and uses SAM3's official
CPU connected-components fallback for hole filling. `nms_backend:=auto` also
selects these fallbacks automatically on compute capability 10 or newer; use
`native` only where the installed Triton/PTXAS supports the GPU architecture.

## 13. Read The ROS Profiling Output

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

## 14. Benchmark Checklist

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
