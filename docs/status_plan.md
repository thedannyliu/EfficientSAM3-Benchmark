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
- Active PACE L40S benchmark jobs submitted on 2026-05-27:
  - `9206977`: COCO fixed10 image suite, completed successfully on `embers`.
    Outputs: `results/coco/suite/9206977/` and `overlays/coco/suite/9206977/`.
    Main comparison CSV: `results/coco/suite/9206977/coco_suite_component_summary.csv`.
    Coverage: SAM3, four EfficientSAM3 variants, SAM2.1 tiny, Efficient-SAM2.1
    tiny, EfficientTAM-Ti, and EfficientTAM-S. MobileSAM was skipped in this
    suite because its checkpoint had not been downloaded yet.
  - `9210795`: SA-V fixed3 SAM2-family video suite, pending on `Priority`.
    Outputs: `results/sav/video/9210795/` and `overlays/sav/video/9210795/`.
    Main comparison CSV: `results/sav/video/9210795/sav_video_suite_summary.csv`.
  - `9213409`, `9213421`, and `9214646` were cancelled before start because
    `9213409` had been submitted before the YOLOE+EdgeTAM Slurm script was
    changed to use `python -m` instead of a not-yet-installed console entrypoint.
  - `9215034`, `9215042`, and `9215043` were cancelled before start after
    adding YOLOE top-1/best-instance localization diagnostics, so the saved jobs
    would not miss the new output fields.
  - `9215144`: YOLOE-26M-seg + EdgeTAM POC on `videos/test1.mov` with
    text prompt `monitor`, pending/running on PACE.
    Outputs: `results/yoloe_edgetam/9215144/` and `overlays/yoloe_edgetam/9215144/`.
  - `9215147`: MobileSAM `vit_t` COCO fixed10 point-prompt baseline, pending on
    `Priority`. Its previous `afterok:9215144` dependency was removed after
    shared assets were prepared.
    Outputs: `results/coco/mobilesam/9215147/` and `overlays/coco/mobilesam/9215147/`.
  - `9214205`: CPU-only SA-V salient fixed3 preparation, completed data
    extraction and review generation. The Slurm job itself exited nonzero only
    because the console entrypoint `sam-review-sav-manifest` was not installed
    in `.venv`; `scripts/prepare_sav_salient_subset.sh` now calls
    `python -m sam_backend.sav_review`, and the review was generated manually.
    Outputs: `data/manifests/sav_val_salient_fixed3.jsonl`,
    `data/sa-v/sav_val_salient_fixed3/`, and
    `overlays/sav/review/sav_val_salient_fixed3/contact_sheet.png`.
    Manual text prompt template:
    `configs/datasets/sav_val_salient_fixed3_text_prompts.json`.
  - `9215148`: YOLOE-26M-seg + EdgeTAM on the manually labeled
    `data/manifests/sav_val_salient_fixed3_text.jsonl`, submitted with
    dependency `afterok:9215144`.
    Outputs: `results/yoloe_edgetam/sav_text/9215148/` and
    `overlays/yoloe_edgetam/sav_text/9215148/`.
  - Earlier COCO job `9205042` failed before this update because SAM3 returned
    CUDA `bfloat16` tensors; `sam_backend.overlay.to_numpy` now casts bfloat16
    tensors to float32 before NumPy conversion.
  - Earlier SA-V job `9205043` completed SAM2 outputs but failed on the
    Efficient-SAM2 fork's missing `enable_MeP_info` attribute; the video adapter
    now calls `init_memory_info(enable_MeP_info=False)` when available.
    The partial SAM2 artifacts are usable: `results/sav/video/9205043/sav_video_suite_summary.csv`
    summarizes 3 videos, 291 tracked frames, 75 GT frames, mean IoU about 0.680,
    and 3 readable overlay MP4s.
  - Earlier pending jobs `9204912` and `9204913` were cancelled before start so
    the saved Slurm script would not miss the latest suite-level report step.
  - Added scope on 2026-05-27:
  - YOLOE-26M-seg + EdgeTAM video POC for text-prompt open-vocabulary
    localization plus promptable video tracking.
  - MobileSAM `vit_t` point-prompt COCO fixed10 image baseline.
  - Dataset/checkpoint/external storage budget is now 300 GiB.
  - YOLOE, EdgeTAM, and MobileSAM assets are prepared locally:
    `checkpoints/yoloe/yoloe-26m-seg.pt`,
    `checkpoints/edgetam/edgetam.pt`, and
    `checkpoints/mobilesam/mobile_sam.pt`.
  - `scripts/check_pace_qos.sh` verifies Slurm scripts and recent jobs use
    `QOS=embers`; `START_DATE=2026-05-01 scripts/check_pace_qos.sh` currently
    reports no non-embers jobs for this user.

## Document Layers

Use the docs in this order to avoid mixing different pipelines:

1. `docs/thor_offline_benchmark.md`
   - Thor-side offline image/video benchmark and profiling procedure.
   - Covers environment setup, model repo/checkpoint downloads, fixed COCO,
     SA-V, YOLOE+EdgeTAM, overlays, and result interpretation.

2. `docs/thor_ros_camera_benchmark.md`
   - Thor-side live ROS camera benchmark and profiling procedure.
   - Covers camera publishing, backend node prompt modes, result recording,
     overlay recording, and per-model command examples.

3. `docs/thor_setup.md`
   - Older combined Thor setup notes and troubleshooting.

4. `docs/status_plan.md`
   - This file.
   - Records what is done, what each pipeline means, and the next research steps.

5. Generated outputs
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
- `sam_backend/coco_manifest.py`
- `sam_backend/profile_coco.py`
- `sam_backend/coco_suite.py`
- `sam_backend/sav_manifest.py`
- `sam_backend/profile_sav_video.py`
- `sam_backend/summarize_results.py`
- `sam_backend/variant_runner.py`
- `scripts/prepare_coco_fixed_subset.sh`
- `scripts/prepare_benchmark_datasets.sh`
- `scripts/download_sav_valtest_subset.sh`
- `docs/benchmark_dataset_protocol.md`

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
prompt_encoder_ms
mask_decoder_ms
transformer_ms
geometry_encoder_ms
segmentation_head_ms
grounding_ms
detector_ms
memory_attention_ms
memory_encoder_ms
other_ms
FPS = 1000 / latency_ms
params_* and weight_*_bytes for available components
```

### Layer 2: ROS Camera Or Recorded-Video Frame-by-Frame Segmentation

Implemented and pushed.

This is the current ROS path:

```text
camera_stream_node or video_stream_node -> /image
/image -> sam_backend_node -> /sam/result_json
                         \
                          -> /sam/overlay
```

It is also not tracking. The backend node runs `backend.predict(...)` on each
incoming ROS image.

ROS nodes:

- `camera_stream_node`: reads an OpenCV camera index or GStreamer pipeline and publishes `/image`.
- `video_stream_node`: reads a recorded video file and publishes `/image`.
- `sam_backend_node`: runs one backend per image and publishes JSON plus optional overlay image.
  It supports text prompts for SAM3/EfficientSAM3 and point prompts for
  SAM2/Efficient-SAM2/EfficientTAM/MobileSAM.
- `result_recorder_node`: records `/sam/result_json` to CSV and summary CSV.
- `overlay_video_recorder_node`: records `/sam/overlay` to MP4.

Current ROS scope:

- Use `video_stream_node` for deterministic recorded-video checks.
- Use `camera_stream_node` for live Thor camera benchmark/profiling.
- Validate backend segmentation, latency/profile JSON, CSV summaries, and overlay MP4.

### Layer 1b: Fixed COCO Image Prompt/Eval Profiling

Implemented for SAM3-style image backends and the null smoke backend.

This path creates a fixed COCO val2017 JSONL manifest, then runs text and/or
point prompt profiling with simple IoU metrics:

```text
COCO instances JSON -> fixed 10-image manifest
manifest row -> image RGB -> backend.predict(prompt)
             -> component timing + CUDA memory + parameter counts
             -> component weight bytes
             -> best-mask IoU and merged-mask IoU against selected COCO object
             -> CSV + optional overlay PNGs + summary JSON
```

Commands:

```bash
bash scripts/prepare_coco_fixed_subset.sh 10

sam-profile-coco \
  --backend efficientsam3 \
  --checkpoint-path checkpoints/stage1_sam3p1/efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt \
  --device cuda \
  --backbone-type efficientvit \
  --model-name b0 \
  --text-encoder-type MobileCLIP-S0 \
  --text-encoder-context-length 16 \
  --text-encoder-pos-embed-table-size 16 \
  --manifest data/manifests/coco_val2017_fixed10.jsonl \
  --prompt-mode both \
  --csv-output results/coco/single/es3p1_weak_image_weak_text_fixed10/profile.csv \
  --summary-output results/coco/single/es3p1_weak_image_weak_text_fixed10/summary.json \
  --overlay-dir overlays/coco/single/es3p1_weak_image_weak_text_fixed10
```

Current local COCO status:

- `data/coco/images/val2017` exists.
- `data/coco/annotations/instances_val2017.json` exists.
- `data/manifests/coco_val2017_fixed10.jsonl` exists with 10 rows.
- `data/coco/coco_val2017_fixed10_selection.json` records the fixed image IDs,
  annotation IDs, text prompts, and point prompts.
- Null smoke output currently exists at
  `results/coco/smoke/null_fixed10/profile.csv` with 20 rows and
  `overlays/coco/smoke/null_fixed10/` with 20 overlay PNGs.
  New runs should use nested paths under
  `results/coco/<suite-or-smoke>/<run>/<model>/`.
- Current local storage for `data + checkpoints + external` is about 13.78 GiB,
  checked with `scripts/check_storage_budget.sh 300 data checkpoints external`.
- Official SAM3 image checkpoint is stored under `checkpoints/sam3/sam3.pt`
  via `scripts/download_sam3_checkpoint.sh`, so the COCO suite does not rely on
  an unmanaged Hugging Face cache path.

PACE suite command:

```bash
sbatch scripts/pace_l40s_coco_suite.sbatch
```

Manifest selection assumptions:

- 10 random eligible COCO val2017 images with fixed seed `20260527`.
- The prompt target is the largest non-crowd annotated object in each image.
- Text prompt is the COCO category name for that object.
- Point prompt is the centroid of that object's segmentation mask.
- IoU is measured against that selected annotation, not every instance of the category.
- Full prompt/eval rules are recorded in
  `docs/benchmark_dataset_protocol.md`.
- The fixed 10 text and point prompts were visually reviewed on 2026-05-27 and recorded
  in `configs/datasets/coco_val2017_fixed10_prompts.json`.

Supported image-backend adapter status:

- `sam3`: text and point prompts through the native SAM3 image processor.
- `efficientsam3`: text and point prompts through the EfficientSAM3 SAM3-compatible image processor.
- `sam2`: point prompts through `SAM2ImagePredictor.set_image/predict`.
- `efficient-sam2`: point prompts through the Efficient-SAM2 fork's `SAM2ImagePredictor`.
- `efficienttam`: point prompts through `EfficientTAMImagePredictor.set_image/predict`.

Local ignored checkouts currently available for adapter development:

- `external/sam2`
- `external/Efficient-SAM2`
- `external/EfficientTAM`

Default fixed COCO model matrix is implemented in `sam_backend/coco_suite.py`:

- `sam3`
- `es3p1_weak_image_weak_text`
- `es3p1_strong_image_weak_text`
- `es3_weak_image_strong_available_text`
- `es3_strong_image_strong_available_text`
- `sam2p1_hiera_tiny`
- `efficient_sam2p1_hiera_tiny`
- `efficienttam_ti`
- `efficienttam_s`
- `mobilesam_vit_t`

### Layer 3: Native Video Segment-and-Track

Partially implemented for SAM2-family video predictors.

Implemented:

- `sam_backend/sav_manifest.py`
  - Selects a fixed SA-V val/test video subset with seed `20260527`.
  - Uses the largest object in the first available annotation frame as the point prompt target.
  - Also supports `SAV_SELECTION_POLICY=salient_first_mask` via the download
    script to filter out tiny or very thin annotated objects before selecting
    fixed POC targets.
- `sam_backend/sav_review.py`
  - Writes per-sample initial GT mask overlays and a contact sheet so selected
    SA-V object IDs can be visually audited before running expensive tracking.
- `sam_backend/sav_text_prompts.py`
  - Creates a manual text-prompt JSON template for selected SA-V object IDs and
    merges filled prompts back into a separate text-enabled JSONL manifest.
  - Tracks `instance_hint` separately from `text_prompt` because text-only
    prompts such as `person` or `basketball player` are ambiguous when multiple
    same-class instances appear in one frame.
- `sam_backend/profile_sav_video.py`
  - Runs native `build_sam2_video_predictor` or `build_efficienttam_video_predictor`.
  - Calls `init_state`, `add_new_points_or_box`, and `propagate_in_video`.
  - Writes per-frame CSV, per-video summary CSV, optional SA-V-style prediction PNGs, overlay MP4s, component timings, parameter counts, weight bytes, CUDA peak memory, and IoU on official val/test annotations.

Still pending:

- Original SAM3 native video tracking benchmark.
- SAM3-LiteText native video tracking benchmark.
- Full EfficientSAM3 native video tracking is still checkpoint-dependent; current public image encoder checkpoints should not be treated as complete tracking checkpoints.
- Official SA-V val/test subset extraction is handled by
  `scripts/download_sav_valtest_subset.sh`.
- SA-V val/test has masks but no semantic object category names, so current
  SA-V tracking evaluation uses point prompts. Text-prompt video tracking needs
  a separate documented text-label source.
- Current fixed3 SA-V target IDs were selected by the original largest-mask
  policy and include visually minor objects: `sav_018669` object `000` covers
  about `0.421%` of its first frame with a thin bbox; `sav_023216` object `002`
  covers about `0.683%`; `sav_018332` object `000` covers about `3.122%`.
  They are valid official GT objects but poor demo targets. Prefer
  `bash scripts/prepare_sav_salient_subset.sh` for visual POC overlays; it
  writes `data/manifests/sav_val_salient_fixed3.jsonl` and leaves the current
  fixed3 manifest untouched.
- Keep dataset/checkpoint/external storage under 300 GiB. Use
  `scripts/check_storage_budget.sh 300 data checkpoints external` before and
  after SA-V downloads/extraction.
- Do not download the full SA-V archive set into this repo. Use
  `bash scripts/download_sav_valtest_subset.sh val 3`, which downloads the
  official val tar, keeps only 3 GT videos, records the selected IDs in
  `data/sa-v/sav_val_fixed3/official_subset_manifest.json`, and removes the
  tar unless `KEEP_SAV_ARCHIVE=1`.

PACE SAM2-family SA-V command:

```bash
DOWNLOAD_SAM2_FAMILY_CHECKPOINTS=1 sbatch scripts/pace_l40s_sav_video_sam2_family.sbatch
```

### Layer 3a: Sampled Camera-Frame Smoke

Implemented as a ROS-free bridge between offline benchmark code and the future
Thor camera path.

- `sam_backend/thor_pipeline_smoke.py` now accepts a video path or OpenCV
  camera index, text prompts, point prompts, SAM3/EfficientSAM3 image backends,
  and SAM2-family point-prompt image backends.
- `scripts/run_sampled_camera_frame_smoke.sh` samples one frame by default and
  runs:
  - `efficientsam3_es3p1_weak_image_weak_text` with text prompt `monitor`.
  - `efficient_sam2p1_hiera_tiny` with normalized point prompt `0.5,0.5`.
- `scripts/pace_l40s_sampled_camera_frame_smoke.sbatch` runs the same smoke on
  PACE L40S with `embers` QOS.

Current submitted PACE job:

```text
9210880 sampled-camera-smoke PENDING (Priority)
```

Expected output layout:

```text
results/camera_sample/<run_id>/<model>/result.jsonl
results/camera_sample/<run_id>/<model>/sampled_frames/frame_000000.png
overlays/camera_sample/<run_id>/<model>/frames/frame_000000.png
overlays/camera_sample/<run_id>/<model>/overlay.mp4
```

### Layer 3b: YOLOE-26M-seg + EdgeTAM POC

Implemented as a reusable CLI/Slurm entry point, pending GPU run.

This path intentionally excludes VLM, Grounding DINO, EfficientViT-SAM,
SAM3 teacher, distillation, and fallback models. The first POC only validates:

```text
text prompt -> YOLOE-26M-seg instance mask/box
            -> EdgeTAM box prompt
            -> EdgeTAM per-frame mask tracking
            -> YOLOE low-frequency validation/re-grounding
```

Tools:

- `sam_backend/profile_yoloe_edgetam.py`
- `scripts/download_yoloe_edgetam_mobilesam_assets.sh`
- `scripts/pace_l40s_yoloe_edgetam_poc.sbatch`

Expected outputs:

```text
results/yoloe_edgetam/<run_id>/frames.csv
results/yoloe_edgetam/<run_id>/frames_summary.csv
results/yoloe_edgetam/<run_id>/summary.json
overlays/yoloe_edgetam/<run_id>/<source_id>/overlay.mp4
```

Recorded profiling fields include YOLOE `set_classes` latency, YOLOE initial
localization latency, low-frequency YOLOE validation latency, EdgeTAM
`init_state`, `add_new_points_or_box`, and `propagate_in_video` latency,
effective tracking FPS, re-ground count, CUDA memory, and YOLOE/EdgeTAM
parameter and weight sizes.

Default POC settings:

```text
YOLOE model: checkpoints/yoloe/yoloe-26m-seg.pt
YOLOE input: 640
YOLOE interval: 20 frames
EdgeTAM checkpoint: checkpoints/edgetam/edgetam.pt
EdgeTAM config: external/EdgeTAM/sam2/configs/edgetam.yaml
Prompt source: operator-supplied text prompt per recorded video
```

### Layer 1c: MobileSAM Image Baseline

Implemented as a point-prompt COCO fixed10 backend, pending GPU run.

Tools:

- `sam_backend/backends.py` backend id `mobilesam`
- `scripts/pace_l40s_mobilesam_coco.sbatch`

Expected outputs:

```text
results/coco/mobilesam/<run_id>/profile.csv
results/coco/mobilesam/<run_id>/summary.json
overlays/coco/mobilesam/<run_id>/*.png
```

The MobileSAM run uses `vit_t` and `checkpoints/mobilesam/mobile_sam.pt`,
records the same IoU/overlay/profile columns as the other COCO image backends,
and fills image encoder, prompt encoder, mask decoder, total parameter, and
weight-size fields when the native module exposes them.

### Layer 3c: SAM3 Native Video Tracking

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
