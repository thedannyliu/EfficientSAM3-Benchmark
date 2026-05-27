# Status and Plan

Last updated: 2026-05-27.

## Current Status

- Repo tracks benchmark code, ROS wrappers, scripts, configs, and lightweight tests only.
- Ignored local/runtime content includes `.venv/`, `external/`, `videos/`, `images/`,
  `results/`, `overlays/`, `logs/`, checkpoints, TensorRT engines, and rosbags.
- Default prompt for current monitor demos is `monitor`.
- Default recorded videos are `videos/test1.mov` and `videos/test2.mov`.
- PACE remains the backend benchmark/profiling environment.
- Jetson Thor is the ROS, JetPack CUDA/TensorRT, and deployment validation environment.
- Thor uses one unified environment via `scripts/source_thor_ros_env.sh`.

## Document Layers

Use the docs in this order to avoid mixing different pipelines:

1. `docs/thor_setup.md`
   - Main Thor tutorial.
   - Covers environment setup, offline benchmarks, ROS recorded-video segmentation,
     and troubleshooting.
   - Camera input is intentionally deferred until recorded-video ROS works.

2. `docs/status_plan.md`
   - This file.
   - Records what is done, what each pipeline means, and the next research steps.

3. Generated outputs
   - `results/` for CSV/summary metrics.
   - `overlays/` for image/video demos.
   - These are local artifacts and must not be committed.

## Pipeline Layers

### Layer 1: Offline Frame-by-Frame Segmentation

Implemented.

This is the current `sam_backend.profile_video` path:

```text
video file -> cv2.VideoCapture -> frame RGB -> backend.predict(frame, prompt)
           -> per-frame CSV metrics -> optional overlay MP4
```

This is not tracking. Every processed frame is segmented independently with the
same prompt.

Main tools:

- `sam_backend/profile_video.py`
- `sam_backend/profile_image.py`
- `sam_backend/summarize_results.py`
- `sam_backend/variant_runner.py`

Current known-good EfficientSAM3 example:

```text
backend=efficientsam3
checkpoint=checkpoints/effsam3/efficient_sam3_efficientvit_s_sa_1b_1p.pt
backbone_type=efficientvit
model_name=b0
prompt=monitor
```

Measured metrics are per processed frame:

```text
total_ms
image_encoder_ms
text_encoder_ms
grounding_ms
other_ms
FPS = 1000 / latency_ms
```

### Layer 2: ROS Recorded-Video Frame-by-Frame Segmentation

Implemented and pushed.

This is the current ROS path:

```text
video_stream_node -> /image
/image -> sam_backend_node -> /sam/result_json
                         \
                          -> /sam/overlay
```

It is also not tracking. The backend node runs `backend.predict(...)` on each
incoming ROS image.

ROS nodes:

- `video_stream_node`: reads a recorded video file and publishes `/image`.
- `sam_backend_node`: runs one backend per image and publishes JSON plus optional overlay image.
- `result_recorder_node`: records `/sam/result_json` to CSV and summary CSV.
- `overlay_video_recorder_node`: records `/sam/overlay` to MP4.

Current ROS scope:

- Use recorded videos only.
- Validate backend segmentation, latency/profile JSON, CSV summaries, and overlay MP4.
- Do not switch to real camera until this recorded-video path is stable.

### Layer 3: Native Video Segment-and-Track

Not implemented in this repo yet.

This is the next priority. It should use the original SAM3 video API:

```text
build_sam3_video_predictor()
start_session(video_path)
add_prompt(frame_index=0, text=monitor)
propagate_in_video()
```

This path is true video tracking because it maintains session state, object IDs,
and propagation across frames.

First target backends:

- Original SAM3 video predictor.
- SAM3-LiteText video predictor, because it keeps the SAM3 image/tracker path
  and replaces only the text encoder.

Expected outputs:

```text
session_init_ms
add_prompt_ms
propagate_total_ms
mean_propagate_ms_per_frame
effective_tracking_fps
frames_tracked
object_count
per-frame object IDs
overlay MP4 with stable object colors
```

### Layer 4: EfficientSAM3 Native Video Tracking

Research/development stage.

EfficientSAM3 code contains `build_efficientsam3_video_predictor`, but the
currently used public EfficientSAM3 weights are image/text encoder weights, not
full video tracking checkpoints.

A full video checkpoint would need detector, student image/text encoder,
tracker, temporal memory, and propagation-related weights. The currently used
`efficient_sam3_efficientvit_s_sa_1b_1p.pt` should be treated as an image
segmentation checkpoint, not a complete video tracking checkpoint.

Do not claim EfficientSAM3 native tracking results until a matching full video
checkpoint is available or trained.

## Completed Work

- Added model-independent backend API in `sam_backend/`.
- Added offline video profiler with CSV metrics and overlay MP4 output.
- Added offline image profiler with text and point prompts.
- Added benchmark summary generator with latency and FPS summaries.
- Added PACE setup and Slurm scripts for backend profiling.
- Added Thor unified ROS environment helper.
- Fixed Thor ROS Python path issues for `/usr/bin/python3` ROS entrypoints.
- Documented NumPy `<2` requirement for ROS `cv_bridge` compatibility.
- Added ROS recorded-video segmentation nodes and recorders.
- Updated Thor tutorial to separate offline benchmarking from ROS recorded-video deployment checks.

## Current Interpretation of Results

- Offline `profile_video` results measure per-frame segmentation latency, not tracking.
- ROS recorded-video results measure per-frame segmentation plus ROS callback/transport overhead, not tracking.
- SAM3-LiteText success suggests the reduced text encoder can preserve text-prompt behavior while keeping SAM3 visual/tracker components.
- EfficientSAM3 image encoder checkpoints can be tested for frame-by-frame segmentation, but do not prove native video tracking capability.

## Next Plan

1. Add an offline SAM3 video tracking benchmark.
   - Start with original SAM3.
   - Then run SAM3-LiteText.
   - Export per-frame tracked masks, object IDs, CSV metrics, summary CSV, and overlay MP4.

2. Compare tracking against current frame-by-frame segmentation.
   - SAM3 image-per-frame vs SAM3 native video tracking.
   - SAM3-LiteText image-per-frame vs SAM3-LiteText video tracking.
   - Compare latency, effective FPS, object stability, and overlay quality.

3. Add a ROS video tracking node after offline tracking works.
   - The node should consume a recorded `video_path`, not a live camera first.
   - It should run `start_session`, `add_prompt`, and `propagate_in_video`.
   - It should publish tracked result JSON and overlay images for the existing recorders.

4. Try smaller encoder variants only after SAM3/SAM3-LiteText tracking baselines are stable.
   - Treat this as experimental until checkpoint compatibility is proven.
   - Record exact checkpoint, builder config, missing/unexpected keys, and visual quality.

5. If smaller encoder video tracking is poor or blocked, design a distillation pipeline.
   - Teacher: SAM3 native video predictor.
   - Student: smaller image/text encoder plus compact temporal memory.
   - Target missing release components: Stage 2 temporal memory and Stage 3 end-to-end video fine-tuning.
   - Metrics: tracking FPS, mask IoU when labels exist, temporal consistency, object dropout, and overlay quality.
