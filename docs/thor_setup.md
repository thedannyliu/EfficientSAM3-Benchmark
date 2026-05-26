# Jetson Thor Benchmark and ROS Guide

Last updated: 2026-05-26.

This guide is the Thor-side procedure for running real EfficientSAM3 profiling,
checking overlay video quality, and then moving the same backend into ROS 2.
PACE is useful for backend development, but TensorRT, camera pipelines, ROS
latency, and final deployment behavior must be validated on Jetson Thor.

Default benchmark target:

```text
prompt: monitor
videos: videos/test1.mov, videos/test2.mov
EfficientSAM3 checkpoint: checkpoints/effsam3/efficient_sam3_efficientvit_s_sa_1b_1p.pt
EfficientSAM3 config: backend=efficientsam3, backbone_type=efficientvit, model_name=b0
```

The EfficientSAM3 checkpoint above is the current working Thor example. It is
an EfficientViT-S image-encoder checkpoint. Do not pass `--text-encoder-type`
for this checkpoint.

## 1. System Check

Run this on Thor:

```bash
cat /etc/os-release
uname -m
nvidia-smi
python3 --version
```

Expected shape:

```text
OS: Ubuntu 24.04
arch: aarch64
GPU: NVIDIA Thor
Python: 3.12.x
```

Use JetPack/NVIDIA packages for CUDA, TensorRT, and Jetson GPU support. Do not
install Ubuntu's generic `nvidia-cuda-toolkit` as a replacement for JetPack.

## 2. Get the Repository

```bash
git clone git@github.com:thedannyliu/EfficientSAM3-Benchmark.git
cd ~/EfficientSAM3-Benchmark
```

If the repo already exists:

```bash
cd ~/EfficientSAM3-Benchmark
git pull
```

Large local inputs and outputs stay on Thor and are not committed:

```text
videos/
checkpoints/
results/
overlays/
logs/
```

## 3. Use One Thor Environment

Use one venv consistently for non-ROS and ROS work:

```text
venv: ~/venvs/effisam3_venv_ros
EfficientSAM3 source: ~/efficientsam3/sam3
ROS: /opt/ros/jazzy/setup.bash
```

The repo provides a helper that sources ROS, activates the venv, and fixes
`PYTHONPATH` so `/usr/bin/python3` ROS entrypoints can still see:

- this repo's `sam_backend`
- venv packages such as `torch`
- editable EfficientSAM3 source package `sam3`
- ROS Jazzy packages such as `rclpy` and `cv_bridge`

Use it in every Thor terminal:

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh
```

If your paths differ, set them before sourcing:

```bash
export THOR_VENV=/path/to/venv
export SAM3_SOURCE=/path/to/efficientsam3/sam3
export THOR_ROS_SETUP=/opt/ros/jazzy/setup.bash
source scripts/source_thor_ros_env.sh
```

## 4. Install Python Dependencies

First install the JetPack-compatible PyTorch and torchvision for Thor/aarch64
using the NVIDIA-supported wheel or container for your JetPack version.

Then install repo dependencies without letting pip replace core ROS/JetPack
packages unexpectedly:

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

python -m pip install -U pip
python -m pip install "numpy>=1.26,<2" opencv-python-headless pillow pyyaml huggingface_hub
python -m pip install timm tqdm ftfy==6.1.1 regex iopath typing_extensions psutil
python -m pip install -e . --no-deps
```

The `numpy<2` pin matters. ROS `cv_bridge` on this Thor stack is built against
NumPy 1.x and can fail with a NumPy 2.x ABI error.

Verify imports:

```bash
python -m sam_backend.env_probe
python -c "import cv2, rclpy, cv_bridge, torch, sam3, sam_backend; print('ok', torch.__version__, torch.cuda.is_available())"
```

If NumPy was accidentally upgraded to 2.x:

```bash
source scripts/source_thor_ros_env.sh
python -m pip install --force-reinstall "numpy>=1.26,<2"
python -m pip install -e . --no-deps
```

## 5. Hugging Face and Local Inputs

```bash
source scripts/source_thor_ros_env.sh
hf auth login
hf auth whoami
```

Expected local files for this guide:

```text
videos/test1.mov
videos/test2.mov
checkpoints/effsam3/efficient_sam3_efficientvit_s_sa_1b_1p.pt
```

Check them:

```bash
ls -lh videos/test1.mov videos/test2.mov
ls -lh checkpoints/effsam3/efficient_sam3_efficientvit_s_sa_1b_1p.pt
```

## 6. Minimal Pipeline Check

This is only a plumbing check. Do not use these numbers as model benchmark
results.

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh
mkdir -p results/thor_pipeline_smoke overlays/thor_pipeline_smoke

python -m sam_backend.thor_pipeline_smoke \
  --backend null \
  --device cpu \
  --prompt monitor \
  --video videos/test1.mov \
  --max-frames 5 \
  --output-jsonl results/thor_pipeline_smoke/null-test1.jsonl \
  --overlay-output overlays/thor_pipeline_smoke/null-test1.mp4
```

Check outputs:

```bash
head results/thor_pipeline_smoke/null-test1.jsonl
ls -lh overlays/thor_pipeline_smoke/null-test1.mp4
```

Do not run `scripts/run_pace_thor_pipeline_smoke.sh` on Thor. That wrapper is
PACE-specific and calls `module load`.

## 7. EfficientSAM3 Performance Profiling

This is the real EfficientSAM3 benchmark path for Thor. It writes per-frame
latency CSVs and overlay MP4s.

Run `test1`:

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh
mkdir -p results overlays

python -m sam_backend.profile_video \
  --model-id esam3-efficientvit-s-sa1b1p-test1 \
  --backend efficientsam3 \
  --checkpoint-path checkpoints/effsam3/efficient_sam3_efficientvit_s_sa_1b_1p.pt \
  --device cuda \
  --backbone-type efficientvit \
  --model-name b0 \
  --prompt monitor \
  --video videos/test1.mov \
  --max-frames 300 \
  --csv-output results/esam3-efficientvit-s-sa1b1p-test1.csv \
  --overlay-output overlays/esam3-efficientvit-s-sa1b1p-test1.mp4
```

Run `test2`:

```bash
python -m sam_backend.profile_video \
  --model-id esam3-efficientvit-s-sa1b1p-test2 \
  --backend efficientsam3 \
  --checkpoint-path checkpoints/effsam3/efficient_sam3_efficientvit_s_sa_1b_1p.pt \
  --device cuda \
  --backbone-type efficientvit \
  --model-name b0 \
  --prompt monitor \
  --video videos/test2.mov \
  --max-frames 300 \
  --csv-output results/esam3-efficientvit-s-sa1b1p-test2.csv \
  --overlay-output overlays/esam3-efficientvit-s-sa1b1p-test2.mp4
```

`profile_video` prints a JSON summary. `mean_total_ms` is the mean per-frame
latency for the profiled frames.

The CSV contains:

```text
total_ms
image_encoder_ms
text_encoder_ms
grounding_ms
other_ms
cuda_allocated_mb
cuda_reserved_mb
params_total
```

## 8. Summarize Latency

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

python - <<'PY'
import csv
from pathlib import Path

paths = [
    Path("results/esam3-efficientvit-s-sa1b1p-test1.csv"),
    Path("results/esam3-efficientvit-s-sa1b1p-test2.csv"),
]

for path in paths:
    rows = list(csv.DictReader(path.open()))
    totals = sorted(float(r["total_ms"]) for r in rows)
    if not totals:
        print(path, "no rows")
        continue
    mean = sum(totals) / len(totals)
    p50 = totals[len(totals) // 2]
    p95 = totals[int((len(totals) - 1) * 0.95)]
    print(f"{path}: frames={len(totals)} mean={mean:.2f}ms p50={p50:.2f}ms p95={p95:.2f}ms")
PY
```

Record the JSON summary from `profile_video` and the latency summary above in
your notes.

## 9. Check Overlay Outputs

First confirm the MP4s exist and are non-empty:

```bash
ls -lh overlays/esam3-efficientvit-s-sa1b1p-test1.mp4
ls -lh overlays/esam3-efficientvit-s-sa1b1p-test2.mp4
```

Then confirm OpenCV can decode at least one frame:

```bash
python - <<'PY'
import cv2
from pathlib import Path

for path in [
    Path("overlays/esam3-efficientvit-s-sa1b1p-test1.mp4"),
    Path("overlays/esam3-efficientvit-s-sa1b1p-test2.mp4"),
]:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    print(path, "ok=", ok, "shape=", None if frame is None else frame.shape)
    cap.release()
PY
```

Finally, open the overlay videos on Thor or copy them to a machine with a video
player. Check whether the masks/boxes visually align with `monitor`. A fast
model run with empty or incorrect overlays is not a successful benchmark.

## 10. Optional SAM3 and SAM3-LiteText Checks

### Single-image text prompt check

Before comparing video overlays, it is useful to test SAM3 and EfficientSAM3 on
the same still image with a simple prompt. Put a cat image at:

```text
images/cats.jpg
```

Run SAM3 with `prompt=cats`:

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh
mkdir -p results/image_checks overlays/image_checks

python -m sam_backend.profile_image \
  --model-id sam3-cats-image \
  --backend sam3 \
  --device cuda \
  --prompt cats \
  --image images/cats.jpg \
  --json-output results/image_checks/sam3-cats.json \
  --overlay-output overlays/image_checks/sam3-cats.png
```

Run EfficientSAM3 with the current EfficientViT-S checkpoint:

```bash
python -m sam_backend.profile_image \
  --model-id esam3-efficientvit-s-sa1b1p-cats-image \
  --backend efficientsam3 \
  --checkpoint-path checkpoints/effsam3/efficient_sam3_efficientvit_s_sa_1b_1p.pt \
  --device cuda \
  --backbone-type efficientvit \
  --model-name b0 \
  --prompt cats \
  --image images/cats.jpg \
  --json-output results/image_checks/esam3-efficientvit-s-sa1b1p-cats.json \
  --overlay-output overlays/image_checks/esam3-efficientvit-s-sa1b1p-cats.png
```

Compare:

```bash
cat results/image_checks/sam3-cats.json
cat results/image_checks/esam3-efficientvit-s-sa1b1p-cats.json
ls -lh overlays/image_checks/*cats*.png
```

Open both overlay PNGs and confirm the cat masks/boxes are visually correct.
For text-prompt quality checks, a model that runs but produces empty or wrong
cat overlays should be treated as a failed quality check.

### SAM3 video baseline

SAM3 baseline:

```bash
python -m sam_backend.profile_video \
  --model-id sam3-test1 \
  --backend sam3 \
  --device cuda \
  --prompt monitor \
  --video videos/test1.mov \
  --max-frames 300 \
  --csv-output results/sam3-test1.csv \
  --overlay-output overlays/sam3-test1.mp4
```

SAM3-LiteText is not EfficientSAM3. Use `--backend sam3`, not
`--backend efficientsam3`.

Fixed ctx16 LiteText checkpoints should use:

```text
--text-encoder-context-length 16
--text-encoder-pos-embed-table-size 16
```

If the checkpoint reports a positional embedding shape of `77`, use legacy
mode:

```text
--text-encoder-context-length 16
--text-encoder-pos-embed-table-size 77
--interpolate-pos-embed
```

## 11. Install ROS 2 Jazzy

Use the official ROS 2 Jazzy APT flow for Ubuntu 24.04. Then install:

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-ros-base \
  ros-jazzy-cv-bridge \
  ros-jazzy-sensor-msgs \
  ros-jazzy-std-msgs \
  python3-colcon-common-extensions
```

Verify:

```bash
source /opt/ros/jazzy/setup.bash
ros2 --help
python3 -c "import rclpy, sensor_msgs, std_msgs; print('ros ok')"
```

## 12. Build the ROS Workspace

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

python -c "import cv2, rclpy, cv_bridge, torch, sam3, sam_backend; print('ros python ok')"

cd ros_ws
colcon build --symlink-install
cd ..
source scripts/source_thor_ros_env.sh
```

Check the ROS entrypoint:

```bash
head -1 ros_ws/install/sam_benchmark_ros/lib/sam_benchmark_ros/sam_backend_node
```

It may show:

```text
#!/usr/bin/python3
```

That is acceptable when `scripts/source_thor_ros_env.sh` has been sourced,
because it adds the venv site-packages and EfficientSAM3 source to `PYTHONPATH`.

## 13. ROS Video Pipeline

Terminal A, publish video frames:

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh
cd ros_ws

ros2 run sam_benchmark_ros video_stream_node --ros-args \
  -p video_path:=../videos/test1.mov \
  -p image_topic:=/image \
  -p fps:=15.0
```

Terminal B, start null backend first:

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh
cd ros_ws

ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=null \
  -p device:=cpu \
  -p prompt:=monitor \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json
```

Terminal C, inspect topics:

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh
cd ros_ws

ros2 topic echo /sam/result_json
ros2 topic hz /image
```

When `backend:=null` works, switch Terminal B to SAM3 or EfficientSAM3.

## 14. ROS EfficientSAM3

The current ROS wrapper exposes `backend`, `checkpoint_path`, `device`, `prompt`,
and topic parameters. It does not yet expose `backbone_type`, `model_name`, or
text-encoder parameters. For the current working EfficientViT-S checkpoint,
that is fine because the defaults are:

```text
backbone_type=efficientvit
model_name=b0
```

Run:

```bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=efficientsam3 \
  -p checkpoint_path:=../checkpoints/effsam3/efficient_sam3_efficientvit_s_sa_1b_1p.pt \
  -p device:=cuda \
  -p prompt:=monitor \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json
```

The ROS node currently publishes JSON summaries only. It does not write overlay
videos and does not publish full mask images yet. Use non-ROS `profile_video`
for the authoritative latency CSV and overlay MP4 benchmark.

## 15. Camera Test

After video-file ROS plumbing works, switch the video node to OpenCV camera
device `0`:

```bash
ros2 run sam_benchmark_ros video_stream_node --ros-args \
  -p video_path:="" \
  -p image_topic:=/image \
  -p fps:=15.0
```

If Thor camera input requires a GStreamer/NVIDIA camera pipeline, add a
camera-specific node. Do not force a GStreamer pipeline string into the existing
smoke video node.

## 16. TensorRT and ONNX

TensorRT engine building and final TensorRT latency validation must happen on
Thor because TensorRT engines are tied to hardware, CUDA, TensorRT, and driver
versions.

```text
export ONNX: PACE or Thor
build TensorRT engine: Thor
validate TensorRT latency: Thor
```

## 17. Troubleshooting

### `module: command not found`

You are running a PACE script on Thor. Do not use scripts that call
`module load` on Thor.

### `cv_bridge` NumPy ABI Error

Downgrade NumPy in the Thor venv:

```bash
source scripts/source_thor_ros_env.sh
python -m pip install --force-reinstall "numpy>=1.26,<2"
python -m pip install -e . --no-deps
```

### `ModuleNotFoundError: sam_backend`, `torch`, or `sam3` in ROS

Use the unified environment script in that terminal:

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh
python -c "import torch, sam3, sam_backend; print('ok')"
```

### EfficientSAM3 Checkpoint Mismatch

For `efficient_sam3_efficientvit_s_sa_1b_1p.pt`, use:

```text
--backend efficientsam3
--backbone-type efficientvit
--model-name b0
```

Do not add LiteText `--text-encoder-type` parameters for this checkpoint.

### SAM3-LiteText Positional Embedding Mismatch

If the checkpoint shape says `77` but the current model shape says `16`, use:

```text
--text-encoder-pos-embed-table-size 77
--interpolate-pos-embed
```

and keep `--backend sam3`.

## 18. What to Record

For each real benchmark run, record:

```text
date
git commit
JetPack version
Ubuntu version
Python version
PyTorch version
CUDA available
GPU
backend
checkpoint
video source
prompt
max_frames
CSV path
overlay path
frames profiled
mean_total_ms
p50_total_ms
p95_total_ms
overlay visual quality notes
errors/log path
```
