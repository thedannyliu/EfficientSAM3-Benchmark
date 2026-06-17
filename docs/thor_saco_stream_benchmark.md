# Jetson Thor SA-Co/VEval Stream Benchmark

This guide runs the new SA-Co/VEval-SAV stream benchmark on Jetson Thor. It
uses the same repository checkout and Thor Python/ROS environment as
`docs/thor_offline_benchmark.md`, then adds the SA-Co/VEval stream data,
checkpoints, suite commands, and overlay outputs.

The benchmark writes quantitative CSV/JSON results and one overlay MP4 per
model/video. Large assets should stay on scratch, not in the Git workspace.

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

## 3. Use Scratch For Large Assets

Default scratch root:

```bash
export SAM_BENCH_SCRATCH=/storage/scratch1/9/eliu354/efficientsam3-benchmark
```

Expected layout:

```text
$SAM_BENCH_SCRATCH/
├── checkpoints/
├── external/
└── data/
    ├── annotation/
    └── media/saco_sav/JPEGImages_24fps/
```

If full SA-V frames already exist somewhere else from the old setup, pass that
path as `SAV_JPEG_ROOT`. The setup script will symlink it into the scratch
layout:

```bash
SAV_JPEG_ROOT=/path/to/full/JPEGImages_24fps \
  bash scripts/setup_thor_saco_stream_benchmark.sh
```

The old `sav_val_fixed10` subset is useful for smoke tests, but it is not
enough for the default fixed20 SA-Co/VEval benchmark unless all selected videos
happen to overlap.

## 4. One-Command Setup

The setup script reuses the active Thor venv, installs only missing Python
packages without replacing Jetson PyTorch, downloads the new stream benchmark
assets to scratch, downloads SA-Co/VEval-SAV annotations, builds a fixed
manifest, and runs a null-backend smoke test with overlay output.

```bash
bash scripts/setup_thor_saco_stream_benchmark.sh
```

Useful environment knobs:

```text
SAM_BENCH_SCRATCH=/storage/scratch1/9/eliu354/efficientsam3-benchmark
SACO_SPLIT=val
SACO_COUNT=20
SACO_SEED=20260617
SACO_SAV_MEDIA_ROOT=$SAM_BENCH_SCRATCH/data/media/saco_sav/JPEGImages_24fps
SAV_JPEG_ROOT=/path/to/existing/JPEGImages_24fps
INSTALL_DEPS=1
DOWNLOAD_ASSETS=1
DOWNLOAD_SACO_ANNOTATION=1
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

Or run a smaller first pass:

```bash
SACO_MODELS="mobilesam_vit_t_bbox_chain sam2p1_hiera_tiny_native sam3_ref_text_bbox_chain efficientsam3_ev_m_text_bbox_chain" \
MAX_FRAMES=60 \
RUN_SUITE=1 DRY_RUN=0 \
bash scripts/setup_thor_saco_stream_benchmark.sh
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
  --external-repo "$SAM_BENCH_SCRATCH/external/efficientsam3" \
  --checkpoint-path "$SAM_BENCH_SCRATCH/checkpoints/efficientsam3_ft/efficientsam3_efficientvit.pt" \
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

For end-to-end ROS timing, use the same manifest and publish one selected video
at 30 FPS with `video_stream_node`. Then start the matching backend and record
result/overlay topics.

The helper prints recorder commands:

```bash
bash scripts/run_thor_ros_saco_stream_suite.sh \
  data/manifests/saco_veval_sav_fixed20.jsonl \
  sam3_ref_text_bbox_chain
```

Recorders write:

```text
results/thor/ros_saco/<model_id>/results.csv
results/thor/ros_saco/<model_id>/summary.csv
overlays/thor/ros_saco/<model_id>/overlay.mp4
```

Use the offline stream suite first. The ROS path measures transport/callback
overhead and should be treated as deployment timing, not the primary quality
evaluation.
