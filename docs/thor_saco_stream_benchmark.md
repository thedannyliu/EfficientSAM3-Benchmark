# Jetson Thor SA-Co/VEval Stream Benchmark

This guide runs the new SA-Co/VEval-SAV stream benchmark on Jetson Thor. It
uses the same repository checkout and Thor Python/ROS environment as
`docs/thor_offline_benchmark.md`, then adds the SA-Co/VEval stream data,
checkpoints, suite commands, and overlay outputs.

The benchmark writes quantitative CSV/JSON results and one overlay MP4 per
model/video. Large assets stay under ignored local directories in this repo
unless you explicitly override the asset root.

The benchmark now has two execution layers:

```text
offline SA-Co:
  quality metrics plus offline timing on SA-Co/SA-V frames

ROS video stream:
  video_stream_node -> backend node -> result/overlay recorders
  deployment timing on the existing ROS video pipeline
```

Offline SA-Co has two model families of runs:

```text
video:
  offline native video tracking where supported, otherwise offline bbox-chain
  stream tracking

image_per_frame:
  independent image inference on each SA-Co frame, with no previous-frame bbox
  state carried forward
```

Latency/FPS columns use these definitions:

```text
latency_ms       model itself
end_to_end_ms    full per-frame benchmark step
effective_fps    1000 / mean_end_to_end_ms
```

## Quick One-Command Run

Use this path when Thor already has the venv, external repos, checkpoints, and
SA-Co/SA-V assets from earlier setup. The script will reuse existing assets,
download missing lightweight pieces, run a null smoke test, then run:

```text
1. offline SA-Co video + image_per_frame suite
2. ROS video stream pipeline timing on a materialized SA-Co clip
```

```bash
cd ~/EfficientSAM3-Benchmark
git checkout main
git pull

export THOR_VENV=~/venvs/effisam3_venv_ros
export SAM3_SOURCE=~/efficientsam3/sam3
export THOR_ROS_SETUP=/opt/ros/jazzy/setup.bash
unset SAM_BENCH_SCRATCH

bash scripts/run_thor_saco_video_and_image_per_frame.sh
```

The one-command output goes to:

```text
results/thor/saco_video_image_per_frame/<run_id>/saco_stream_suite_summary.csv
results/thor/saco_video_image_per_frame/<run_id>/<model_id>/frames.csv
results/thor/saco_video_image_per_frame/<run_id>/<model_id>/frames_summary.csv
overlays/thor/saco_video_image_per_frame/<run_id>/<model_id>/<source_id>/overlay.mp4

results/thor/ros_saco_stream/<run_id>/ros_saco_stream_summary.csv
results/thor/ros_saco_stream/<run_id>/<model_id>/results.csv
results/thor/ros_saco_stream/<run_id>/<model_id>/summary.csv
overlays/thor/ros_saco_stream/<run_id>/<model_id>/overlay.mp4
```

For a command-only check without loading models:

```bash
DRY_RUN=1 bash scripts/run_thor_saco_video_and_image_per_frame.sh
```

For a smaller first pass:

```bash
SACO_MODELS="sam3_ref_native sam3_ref_image_per_frame sam1_vit_h_bbox_chain sam1_vit_h_image_per_frame mobilesam_vit_t_bbox_chain mobilesam_vit_t_image_per_frame" \
ROS_MODELS="sam3_ref_native sam1_vit_h_bbox_chain mobilesam_vit_t_bbox_chain" \
MAX_FRAMES=60 \
bash scripts/run_thor_saco_video_and_image_per_frame.sh
```

Useful one-command switches:

```text
RUN_OFFLINE=1
RUN_ROS=1
DRY_RUN=0
SACO_MODELS="sam3_ref_native sam3_ref_image_per_frame"
ROS_MODELS="mobilesam_vit_t_bbox_chain sam1_vit_h_bbox_chain sam3_ref_native"
ROS_VIDEO_PATH=/path/to/existing/video.mp4
ROS_PROMPT=monitor
ROS_INITIAL_POINT_X=0.5
ROS_INITIAL_POINT_Y=0.5
```

## 1. Get The Repository

```bash
git clone git@github.com:thedannyliu/EfficientSAM3-Benchmark.git
cd EfficientSAM3-Benchmark
git fetch origin
git checkout main
```

If the repo already exists on Thor:

```bash
cd EfficientSAM3-Benchmark
git checkout main
git pull
```

## 2. Use One Thor Environment

Install JetPack and the NVIDIA-provided PyTorch/torchvision wheels that match
the Thor JetPack release first. Do not let generic PyPI replace them.
This guide uses the shared Thor helper from `docs/thor_setup.md`; it expects
ROS Jazzy at `THOR_ROS_SETUP` even for offline runs.

```bash
sudo apt update
sudo apt install -y python3-opencv

python3 -m venv --system-site-packages ~/venvs/effisam3_venv_ros
export THOR_VENV=~/venvs/effisam3_venv_ros
export SAM3_SOURCE=~/efficientsam3/sam3
export THOR_ROS_SETUP=/opt/ros/jazzy/setup.bash
source scripts/source_thor_ros_env.sh

python -m pip install -U pip

# Follow NVIDIA's Jetson PyTorch page for the exact wheel URLs for this Thor.
# Then verify CUDA before installing this repo:
python - <<'PY'
import torch
print(torch.__version__)
print("cuda:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

If this environment already exists from `docs/thor_offline_benchmark.md`, reuse
it instead of recreating it:

```bash
cd EfficientSAM3-Benchmark
export THOR_VENV=~/venvs/effisam3_venv_ros
export SAM3_SOURCE=~/efficientsam3/sam3
export THOR_ROS_SETUP=/opt/ros/jazzy/setup.bash
source scripts/source_thor_ros_env.sh
```

## 3. Use Repo-Local Asset Directories

By default, the Thor setup script stores checkpoints, external repos, SA-Co
annotations, media symlinks, results, and overlays under this repository. These
directories are ignored by git.

```bash
cd EfficientSAM3-Benchmark
```

Expected layout:

```text
EfficientSAM3-Benchmark/
├── checkpoints/
├── external/
├── data/
│   ├── annotation/
│   └── media/saco_sav/JPEGImages_24fps/
├── results/
└── overlays/
```

If full SA-V frames already exist somewhere else from the old setup, pass that
path as `SAV_JPEG_ROOT`. The setup script will symlink it into the local
`data/media/saco_sav/` layout:

```bash
SAV_JPEG_ROOT=/path/to/full/JPEGImages_24fps \
  bash scripts/setup_thor_saco_stream_benchmark.sh
```

The old `sav_val_fixed10` subset is useful for smoke tests, but it is not
enough for the default fixed20 SA-Co/VEval benchmark unless all selected videos
happen to overlap.

If `SAV_JPEG_ROOT` is not provided and `data/media/saco_sav/JPEGImages_24fps`
does not exist yet, the setup script downloads the official SA-V split archive
and extracts only the videos selected by the SA-Co fixed manifest. The archive
is cached under `data/sa-v/_archives` while extracting and removed afterward
unless `KEEP_SAV_ARCHIVE=1` is set.

If you intentionally want assets outside the repo, set `SAM_BENCH_SCRATCH` to
that external root before running the setup script. This is optional on Thor.
If an old shell still exports an unwritable `/storage/...` path, clear it first:

```bash
unset SAM_BENCH_SCRATCH
```

The setup/download scripts also fall back to the repo-local asset root when
`SAM_BENCH_SCRATCH` is set but not writable.

## 4. One-Command Setup

The setup script reuses the active Thor venv, installs only missing Python
packages without replacing Jetson PyTorch, downloads the new stream benchmark
assets, downloads SA-Co/VEval-SAV annotations, builds a fixed manifest, and
downloads the selected SA-V frames if needed, then runs a null-backend smoke
test with overlay output.

```bash
bash scripts/setup_thor_saco_stream_benchmark.sh
```

Useful environment knobs:

```text
SAM_BENCH_SCRATCH=$PWD
SACO_SPLIT=val
SACO_COUNT=20
SACO_SEED=20260617
SACO_SAV_MEDIA_ROOT=$SAM_BENCH_SCRATCH/data/media/saco_sav/JPEGImages_24fps
SAV_JPEG_ROOT=/path/to/existing/JPEGImages_24fps
INSTALL_DEPS=1
DOWNLOAD_ASSETS=1
DOWNLOAD_SACO_ANNOTATION=1
DOWNLOAD_SACO_MEDIA=1
PREPARE_MANIFEST=1
RUN_NULL_SMOKE=1
```

Key outputs:

```text
data/manifests/saco_veval_sav_fixed20.jsonl
results/thor/saco_stream/smoke/null/frames.csv
overlays/thor/saco_stream/smoke/null/<source_id>/overlay.mp4
```

## 5. Dry-Run The Full Suite

Print commands without loading models:

```bash
RUN_SUITE=1 DRY_RUN=1 bash scripts/setup_thor_saco_stream_benchmark.sh
```

Restrict to a subset:

```bash
SACO_MODELS="sam2p1_hiera_tiny_native sam3_ref_text_bbox_chain efficientsam3_ev_m_text_bbox_chain" \
RUN_SUITE=1 DRY_RUN=1 \
bash scripts/setup_thor_saco_stream_benchmark.sh
```

## 6. Run The Full Offline Stream Suite

Run all available models and save overlay MP4s:

```bash
RUN_SUITE=1 DRY_RUN=0 bash scripts/setup_thor_saco_stream_benchmark.sh
```

The setup script defaults to `SACO_MODE_SET=video`. To run only independent
image-per-frame models through the same setup script:

```bash
SACO_MODE_SET=image_per_frame RUN_SUITE=1 DRY_RUN=0 \
  bash scripts/setup_thor_saco_stream_benchmark.sh
```

Run offline video modes, offline image-per-frame modes, and the ROS video stream
pipeline:

```bash
bash scripts/run_thor_saco_video_and_image_per_frame.sh
```

Dry-run the one-command offline + ROS suite:

```bash
DRY_RUN=1 bash scripts/run_thor_saco_video_and_image_per_frame.sh
```

The one-command script accepts the same common knobs:

```text
SACO_COUNT=20
MAX_FRAMES=120
INPUT_FPS=30.0
SACO_MODELS="sam3_ref_native sam3_ref_image_per_frame"
ROS_MODELS="mobilesam_vit_t_bbox_chain sam1_vit_h_bbox_chain sam3_ref_native"
OUTPUT_DIR=results/thor/saco_video_image_per_frame/<run_id>
OVERLAY_DIR=overlays/thor/saco_video_image_per_frame/<run_id>
ROS_OUTPUT_DIR=results/thor/ros_saco_stream/<run_id>
ROS_OVERLAY_DIR=overlays/thor/ros_saco_stream/<run_id>
ROS_VIDEO_PATH=/path/to/existing/video.mp4
```

Or run a smaller first pass:

```bash
SACO_MODELS="mobilesam_vit_t_bbox_chain sam2p1_hiera_tiny_native sam3_ref_text_bbox_chain efficientsam3_ev_m_text_bbox_chain" \
ROS_MODELS="mobilesam_vit_t_bbox_chain sam1_vit_h_bbox_chain sam3_ref_native" \
MAX_FRAMES=60 \
bash scripts/run_thor_saco_video_and_image_per_frame.sh
```

Important outputs:

```text
results/thor/saco_stream/<run_id>/saco_stream_suite_summary.csv
results/thor/saco_stream/<run_id>/<model_id>/frames.csv
results/thor/saco_stream/<run_id>/<model_id>/frames_summary.csv
results/thor/saco_stream/<run_id>/<model_id>/saco_veval_preds.json
results/thor/saco_stream/<run_id>/<model_id>/saco_veval_eval_res.json
overlays/thor/saco_stream/<run_id>/<model_id>/<source_id>/overlay.mp4
```

The summary reports stream mode, effective FPS, latency, mIoU, mask F1/AP-style
threshold summaries, presence accuracy, CUDA memory, and links to overlay and
prediction artifacts.

## 7. Model Matrix

The default suite includes:

```text
mobilesam_vit_t_bbox_chain
sam1_vit_b_bbox_chain
sam1_vit_l_bbox_chain
sam1_vit_h_bbox_chain
sam2p1_hiera_tiny_bbox_chain
sam2p1_hiera_tiny_native
sam2p1_hiera_large_bbox_chain
sam2p1_hiera_large_native
sam3_ref_text_bbox_chain
sam3_ref_native
sam3p1_ref_native
efficientsam3_ev_m_text_bbox_chain
efficientsam3_rv_m_text_bbox_chain
efficientsam3_tv_m_text_bbox_chain
```

The image-per-frame suite includes:

```text
mobilesam_vit_t_image_per_frame
sam1_vit_b_image_per_frame
sam1_vit_l_image_per_frame
sam1_vit_h_image_per_frame
sam2p1_hiera_tiny_image_per_frame
sam2p1_hiera_large_image_per_frame
sam3_ref_image_per_frame
efficientsam3_ev_m_image_per_frame
efficientsam3_rv_m_image_per_frame
efficientsam3_tv_m_image_per_frame
```

SAM3.1 is currently video-native only in this suite because the available
`sam3.1_multiplex.pt` checkpoint is used through the native video predictor, not
the SAM3 image processor.

`sam2p1_hiera_tiny` is the smallest SAM2 replacement for the earlier SAM2-B
request. `sam3_ref` uses the available `sam3.pt`; `sam3p1_ref` uses
`sam3.1_multiplex.pt` when Hugging Face access is available.

## 8. EfficientSAM3 Full-Model Load Settings

The `efficientsam3_ft/*.pt` checkpoints are full EfficientSAM3 models with a
lightweight image encoder and MobileCLIP-S0 LiteText encoder. They must be
loaded with the matching text encoder settings, otherwise text prompts can
produce no masks because the checkpoint language keys are loaded against the
standard SAM3 text encoder.

The suite passes these settings automatically:

```text
efficientsam3_ev_m_text_bbox_chain: efficientvit / b1 / MobileCLIP-S0 / ctx16
efficientsam3_rv_m_text_bbox_chain: repvit / m1.1 / MobileCLIP-S0 / ctx16
efficientsam3_tv_m_text_bbox_chain: tinyvit / 11m / MobileCLIP-S0 / ctx16
```

For a direct EV-M smoke test on Thor, use the same equivalent builder settings:

```bash
python -m sam_backend.thor_pipeline_smoke \
  --backend efficientsam3 \
  --external-repo external/efficientsam3 \
  --checkpoint-path checkpoints/efficientsam3_ft/efficientsam3_efficientvit.pt \
  --device cuda \
  --backbone-type efficientvit \
  --model-name b1 \
  --text-encoder-type MobileCLIP-S0 \
  --text-encoder-context-length 16 \
  --text-encoder-pos-embed-table-size 16 \
  --prompt monitor \
  --video videos/test1.mov \
  --max-frames 60 \
  --output-jsonl results/thor/saco_stream/ev_m_smoke.jsonl \
  --overlay-output overlays/thor/saco_stream/ev_m_smoke.mp4
```

## 9. Recorded ROS Stream Timing

For end-to-end ROS timing, use the same manifest and publish one selected SA-Co
clip at 30 FPS with `video_stream_node`. The runner materializes the first
manifest video into an MP4 unless `ROS_VIDEO_PATH` points to an existing video.
It then starts one backend at a time and records result/overlay topics.

Run the default ROS video-stream set:

```bash
bash scripts/run_thor_ros_saco_stream_suite.sh data/manifests/saco_veval_sav_fixed20.jsonl
```

The default ROS set runs every currently supported ROS video-stream model:

```text
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
```

Pass an explicit subset for a faster first pass:

```bash
bash scripts/run_thor_ros_saco_stream_suite.sh \
  data/manifests/saco_veval_sav_fixed20.jsonl \
  mobilesam_vit_t_bbox_chain sam1_vit_h_bbox_chain sam3_ref_native
```

Recorders write:

```text
results/thor/ros_saco_stream/<run_id>/ros_saco_stream_summary.csv
results/thor/ros_saco_stream/<run_id>/<model_id>/results.csv
results/thor/ros_saco_stream/<run_id>/<model_id>/summary.csv
overlays/thor/ros_saco_stream/<run_id>/<model_id>/overlay.mp4
```

For SAM1-H and MobileSAM, the ROS runner uses `mobile_sam_interactive_node` in
headless auto-start mode: the first frame gets a point prompt at
`ROS_INITIAL_POINT_X`, `ROS_INITIAL_POINT_Y`, and later frames use
mask-to-bounding-box chaining. SAM3 uses `sam3_native_clip_node`, which captures
`ROS_CLIP_FRAMES` frames from the ROS stream and then runs native tracking.

Use the offline stream suite for mIoU/AP/accuracy. The ROS path measures
transport/callback overhead and deployment timing on the ROS video pipeline.
