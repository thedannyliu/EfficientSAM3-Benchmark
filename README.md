# EfficientSAM3 / SAM3 Jetson Thor Benchmark

Portable benchmark and ROS 2 integration scaffold for developing SAM3 or
EfficientSAM3 backend code on PACE, then validating final deployment on Jetson
Thor.

## Current Feasibility Notes

PACE is useful for backend development and GPU benchmarking, but it is not a
drop-in Thor replica. This workspace probe found:

- PACE login node: RHEL 9.6, x86_64, Python 3.13 conda env.
- PACE GPU queues include `gpu-l40s`, `gpu-h100`, `gpu-h200`, and others.
- Active login env has CPU-only PyTorch and no ROS 2.
- PACE modules include Python 3.12, CUDA 12.6/13.0, and PyTorch modules.

Recommended split:

- PACE: backend API, image/video benchmark loops, Slurm GPU runs.
- Thor: JetPack CUDA/TensorRT validation, ROS 2 Jazzy build, camera pipeline,
  final latency.

Use Python 3.12 for the real environment because ROS 2 Jazzy on Ubuntu 24.04
and upstream SAM3/EfficientSAM3 are aligned there better than with Python 3.13.
On Jetson Thor, install the NVIDIA/JetPack PyTorch wheels first, then use
`requirements-thor.txt` so generic PyPI does not replace the Jetson PyTorch
build.

## Layout

```text
sam_backend/                  # model-independent backend API and benchmark CLI
ros_ws/src/sam_benchmark_ros/ # ROS 2 wrapper nodes
scripts/                      # PACE/Thor helper scripts
configs/                      # environment-specific run configs
tests/                        # smoke tests that do not need model weights
```

## Quick Local Smoke Test

This validates the benchmark loop without requiring CUDA or checkpoints:

```bash
python3 -m unittest
python3 -m sam_backend.benchmark --backend null --synthetic-frames 8 --prompt person
scripts/check_storage_budget.sh 300 data checkpoints external
```

Prepare all fixed benchmark datasets in one step:

```bash
bash scripts/prepare_benchmark_datasets.sh
```

The default COCO fixed10 text and point prompts are recorded in
`configs/datasets/coco_val2017_fixed10_prompts.json`.

Check that Slurm scripts and recent jobs are using the free backfill QOS:

```bash
START_DATE=2026-05-01 scripts/check_pace_qos.sh
```

## COCO Fixed-Image Profiling Shape

Create the fixed 10-image COCO val2017 manifest, then run text and point
prompt profiling with IoU metrics:

```bash
bash scripts/prepare_coco_fixed_subset.sh 10

sam-profile-coco \
  --backend null \
  --device cpu \
  --manifest data/manifests/coco_val2017_fixed10.jsonl \
  --prompt-mode both \
  --eval-mode both \
  --csv-output results/coco/smoke/null_fixed10/profile.csv \
  --summary-output results/coco/smoke/null_fixed10/summary.json
```

For real SAM3/EfficientSAM3 runs, replace `--backend null` with the selected
backend and checkpoint/model flags. The manifest chooses 10 random images with
a fixed seed, uses the largest non-crowd COCO object as the text prompt, uses
that object mask centroid as the point prompt, and reports best-mask and merged
IoU against that selected annotation.

Profiling CSVs include per-run CUDA memory, component latency hooks, component
parameter counts, and component weight bytes. Component columns are shared for
readability, but each backend only fills the components present in its native
model tree: SAM3/EfficientSAM3 use image/text/transformer/geometry/segmentation
and optional interactive prompt/mask components; SAM2-family models use image
encoder, prompt encoder, mask decoder, and video memory components.

Use `--eval-mode both` for metrics plus overlays, `--eval-mode gt` for metrics
only, `--eval-mode overlay` for visual inspection only, or `--eval-mode profile`
for profiling without GT/overlay work.

The selected image IDs, annotation IDs, category prompts, and points are recorded
in `data/coco/coco_val2017_fixed10_selection.json`. The visually reviewed
fixed text and point prompts are tracked in
`configs/datasets/coco_val2017_fixed10_prompts.json` for reproduction on
other devices. The prompt/eval protocol is documented in
`docs/benchmark_dataset_protocol.md`.

Current fixed text prompts are:

```text
cow, train, motorcycle, bird, person, bed, bicycle, zebra, elephant, sink
```

SAM2-family image runs are point-prompt only in this benchmark:

```bash
sam-profile-coco \
  --backend sam2 \
  --external-repo external/sam2 \
  --model-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --checkpoint-path checkpoints/sam2/sam2.1_hiera_tiny.pt \
  --manifest data/manifests/coco_val2017_fixed10.jsonl \
  --prompt-mode point \
  --eval-mode both \
  --csv-output results/coco/single/sam2_tiny_fixed10/profile.csv
```

Use `--backend efficient-sam2 --external-repo external/Efficient-SAM2` for the
Efficient-SAM2 fork, or `--backend efficienttam --external-repo external/EfficientTAM`
with an EfficientTAM config such as `configs/efficienttam/efficienttam_ti.yaml`.
Use `--backend mobilesam --external-repo external/MobileSAM --checkpoint-path
checkpoints/mobilesam/mobile_sam.pt --mobile-sam-model-type vit_t` for the
MobileSAM point-prompt image baseline.

To run the default fixed COCO model matrix:

```bash
bash scripts/download_sam3_checkpoint.sh
sam-run-coco-suite \
  --manifest data/manifests/coco_val2017_fixed10.jsonl \
  --device cuda \
  --skip-missing \
  --output-dir results/coco/suite/manual \
  --overlay-dir overlays/coco/suite/manual
```

Suite-level comparison is written to
`results/coco/suite/<run>/coco_suite_component_summary.csv`, with one row per
model/prompt mode.

On PACE L40S:

```bash
sbatch scripts/pace_l40s_coco_suite.sbatch
```

## SA-V Fixed-Video Profiling Shape

Full SA-V is too large for this repo budget. The official download page lists
train archives at about 8 GiB each and val/test archives at about 16 GiB each.
For GT evaluation, download the official val/test archive, extract only a fixed
3-video subset for tracking, and remove the archive after extraction:

```bash
scripts/check_storage_budget.sh 300 data checkpoints external
bash scripts/download_sav_valtest_subset.sh val 3
```

SAM2-family native video profiler example:

```bash
bash scripts/download_sam2_family_checkpoints.sh
sam-profile-sav-video \
  --backend sam2 \
  --model-id sam2p1_hiera_tiny \
  --external-repo external/sam2 \
  --model-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --checkpoint-path checkpoints/sam2/sam2.1_hiera_tiny.pt \
  --manifest data/manifests/sav_val_fixed3.jsonl \
  --eval-mode both \
  --csv-output results/sav/video/manual/sam2p1_hiera_tiny/frames.csv \
  --summary-output results/sav/video/manual/sam2p1_hiera_tiny/summary.json \
  --pred-root results/sav/video/manual/sam2p1_hiera_tiny/pred \
  --overlay-root overlays/sav/video/manual/sam2p1_hiera_tiny
```

The SA-V profiler uses each manifest row's initial point prompt, propagates with
the backend's native video predictor, reports component timings and parameter
or weight sizes, computes IoU on the official val/test PNG annotations, writes
SA-V-style prediction PNGs, and writes per-video overlay MP4s for visual review.
SA-V val/test has segmentation masks but no semantic category labels, so this
benchmark uses point prompts for video tracking unless a separate text-label
source is explicitly documented.

The default SA-V target selection is official-GT-first, not saliency-first. It
selects the largest annotated object in the first GT frame of each seeded video,
which can still be a visually minor object if that is what SA-V annotated. For
POC overlays with more important-looking targets, rerun extraction with:

```bash
bash scripts/prepare_sav_salient_subset.sh
```

That writes an independent manifest at
`data/manifests/sav_val_salient_fixed3.jsonl` and a review contact sheet at
`overlays/sav/review/sav_val_salient_fixed3/contact_sheet.png`, leaving the
original `sav_val_fixed3` manifest untouched.

To manually add text prompts for a fixed SA-V manifest:

```bash
sam-sav-text-prompts init \
  --manifest data/manifests/sav_val_fixed3.jsonl \
  --review-dir overlays/sav/review/current_fixed3 \
  --output configs/datasets/sav_val_fixed3_text_prompts.json

# Fill text_prompt and instance_hint in configs/datasets/sav_val_fixed3_text_prompts.json.

sam-sav-text-prompts apply \
  --manifest data/manifests/sav_val_fixed3.jsonl \
  --prompts configs/datasets/sav_val_fixed3_text_prompts.json \
  --output data/manifests/sav_val_fixed3_text.jsonl
```

The text prompt must describe the selected official object ID shown in the
review overlay, not a more obvious unannotated object elsewhere in the frame.
When the frame has multiple same-class objects, `text_prompt` alone is
ambiguous; use `instance_hint` to document the selected GT object and report
top-1 localization separately from GT-assisted best-instance diagnostics.
The resulting `_text.jsonl` manifest can be passed to `sam-profile-yoloe-edgetam`
without `--text-prompt`; it will use the per-row prompt.

On PACE L40S:

```bash
SAV_TEXT_MANIFEST=data/manifests/sav_val_salient_fixed3_text.jsonl \
DOWNLOAD_YOLOE_EDGETAM_MOBILESAM=1 \
sbatch scripts/pace_l40s_yoloe_edgetam_sav_text.sbatch
```

On PACE L40S:

```bash
DOWNLOAD_SAM2_FAMILY_CHECKPOINTS=1 sbatch scripts/pace_l40s_sav_video_sam2_family.sbatch
```

The SA-V Slurm job writes `results/sav/video/<run>/sav_video_suite_summary.csv`
with one row per video-capable model.

## YOLOE-26M-seg + EdgeTAM POC

This POC keeps semantic grounding and temporal tracking separate:

```text
text prompt -> YOLOE-26M-seg initial mask/box -> EdgeTAM video tracking
            -> YOLOE low-frequency validation/re-grounding
```

Prepare the assets once in the project `.venv`:

```bash
bash scripts/download_yoloe_edgetam_mobilesam_assets.sh
```

Run the PACE POC on a recorded video:

```bash
TEXT_PROMPT=person YOLOE_SOURCE=videos/test1.mov \
  DOWNLOAD_YOLOE_EDGETAM_MOBILESAM=1 \
  sbatch scripts/pace_l40s_yoloe_edgetam_poc.sbatch
```

Outputs are nested by run:

```text
results/yoloe_edgetam/<run>/frames.csv
results/yoloe_edgetam/<run>/frames_summary.csv
results/yoloe_edgetam/<run>/summary.json
overlays/yoloe_edgetam/<run>/<source_id>/overlay.mp4
```

The summary records YOLOE prompt setup and initial localization latency,
EdgeTAM session/add-prompt/propagation latency, re-ground counts, CUDA memory,
and YOLOE/EdgeTAM parameter and weight sizes.

## MobileSAM COCO Baseline

MobileSAM is included as a point-prompt image segmentation baseline on the same
COCO fixed10 manifest:

```bash
DOWNLOAD_YOLOE_EDGETAM_MOBILESAM=1 sbatch scripts/pace_l40s_mobilesam_coco.sbatch
```

Outputs:

```text
results/coco/mobilesam/<run>/profile.csv
results/coco/mobilesam/<run>/summary.json
overlays/coco/mobilesam/<run>/*.png
```

## Sampled Camera-Frame Smoke

Use this before moving to Thor live camera work. It samples frame(s) from a
camera-like `cv2.VideoCapture` source, runs the same backend shim used by the
ROS-free Thor pipeline smoke, and writes nested results for visual inspection.
On PACE, pass a recorded camera video. On Thor, the source can be a camera index
such as `0` if OpenCV can open the device.

```bash
# Recorded camera/video source on PACE.
sbatch scripts/pace_l40s_sampled_camera_frame_smoke.sbatch

# Direct command, useful inside an interactive GPU allocation.
SAM_PROMPT=monitor SAM_POINT=0.5,0.5 SAM_MAX_FRAMES=1 \
  bash scripts/run_sampled_camera_frame_smoke.sh videos/test1.mov
```

The script runs:

- `efficientsam3_es3p1_weak_image_weak_text` with text prompt `SAM_PROMPT`.
- `efficient_sam2p1_hiera_tiny` with normalized point prompt `SAM_POINT`.

Outputs are written under:

```text
results/camera_sample/<run_id>/<model>/result.jsonl
results/camera_sample/<run_id>/<model>/sampled_frames/frame_000000.png
overlays/camera_sample/<run_id>/<model>/frames/frame_000000.png
overlays/camera_sample/<run_id>/<model>/overlay.mp4
```

## PACE GPU Benchmark Shape

```bash
module load python/3.12.5 cuda/12.6.1
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .

sbatch scripts/pace_l40s_benchmark.sbatch
```

Install SAM3 or EfficientSAM3 in that same Python 3.12 environment before using
the real backends.

For EfficientSAM3 component profiling on the local demo videos:

```bash
bash scripts/setup_pace_venv.sh
sbatch scripts/pace_l40s_profile_sam3.sbatch
sbatch scripts/pace_l40s_profile_efficientsam3.sbatch
```

The default prompt is `monitor`. Per-frame component timings and parameter
counts are written to `results/`; mask overlay demo videos are written to
`overlays/`.

## Thor Guides

Use these Thor-side guides:

- `docs/thor_offline_benchmark.md`: offline benchmark/profiling without ROS.
- `docs/thor_ros_camera_benchmark.md`: live ROS camera benchmark/profiling.
- `docs/thor_setup.md`: older combined setup notes and troubleshooting.

On Thor, build the ROS package after the backend package is installed:

```bash
python3 -m pip install -e .
cd ros_ws
colcon build --symlink-install
source install/setup.bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args -p backend:=sam3
```

The ROS wrapper publishes result JSON, optional overlay images, per-frame CSV
records, aggregate summaries, and overlay MP4 demos through separate recorder
nodes. Use `docs/thor_ros_camera_benchmark.md` for the current camera command
sequence.
