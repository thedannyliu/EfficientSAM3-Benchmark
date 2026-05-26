# Jetson Thor 測試與部署指南

最後更新：2026-05-26。

目標是在 Thor 上驗證 PACE 已開發的同一套 backend，並逐步接上 ROS 2、相機、TensorRT 與實際延遲量測。

官方相容性重點：

- Jetson AGX Thor 使用 JetPack 7 / Ubuntu 24.04 系列。
- ROS 2 Jazzy 官方支援 Ubuntu 24.04 Noble 的 amd64 與 aarch64。
- TensorRT engine、Jetson camera pipeline、真實 ROS latency 必須在 Thor 上驗證，不要只看 PACE。

參考：

- NVIDIA Jetson AGX Thor Quick Start: https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/quick_start.html
- NVIDIA JetPack: https://developer.nvidia.com/embedded/jetpack
- ROS 2 Jazzy installation: https://docs.ros.org/en/jazzy/Installation.html

## 1. 系統檢查

在 Thor 上先確認 OS、GPU、CUDA、Python：

```bash
cat /etc/os-release
uname -m
nvidia-smi
python3 --version
```

期望：

```text
OS: Ubuntu 24.04
arch: aarch64
GPU: NVIDIA Thor
Python: 3.12.x
```

如果 CUDA/TensorRT 尚未安裝，優先使用 JetPack / NVIDIA 官方 APT 套件，不要混用一般 Ubuntu 的 `nvidia-cuda-toolkit`。

## 2. 取得程式碼

```bash
git clone git@github.com:thedannyliu/EfficientSAM3-Benchmark.git
cd EfficientSAM3-Benchmark
```

本 repo 不追蹤影片、checkpoint、engine、logs、results。這些都應該留在 Thor 本地。

## 3. 建立 Python venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

如果已經有 Thor 專用 venv，例如 `/home/ril-thor/venvs/effisam3_venv_ros`，
後續所有 `source .venv/bin/activate` 請改成該 venv 的 activate 路徑。

先不要直接照 PACE 的 `requirements.txt` 安裝 PyTorch wheel，因為 PACE 是 x86_64，Thor 是 aarch64。Thor 上 PyTorch/CUDA wheel 需要依 NVIDIA/JetPack 支援方式安裝。

建議順序：

```bash
# 1. 先安裝 Thor/JetPack 對應的 PyTorch + torchvision
# 依 NVIDIA 當前 Jetson PyTorch 指南或容器版本選擇。

# 2. 再裝 repo 的非 torch dependency
python -m pip install "numpy>=1.26,<2" opencv-python-headless pillow pyyaml huggingface_hub
python -m pip install timm tqdm ftfy==6.1.1 regex iopath typing_extensions psutil

# 3. 安裝本 repo
python -m pip install -e . --no-deps
```

確認：

```bash
python -m sam_backend.env_probe
```

## 4. Hugging Face 登入與 checkpoint

```bash
source .venv/bin/activate  # 若使用外部 venv，改成該 venv 的 activate 路徑
hf auth login
hf auth whoami
```

SAM3 官方 checkpoint 會下載到 Hugging Face cache。EfficientSAM3 runner 會下載到本地：

```text
checkpoints/
```

建議 Thor 上也保留這個目錄，不要 commit。

## 5. 先跑非 ROS pipeline smoke test

這一步確認 Thor 上最核心的資料路徑可以跑：

```text
video -> backend.predict(...) -> result JSONL -> overlay MP4
```

先用 null backend：

```bash
mkdir -p videos
# 放入測試影片，例如 videos/test1.mov

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

注意：不要在 Thor 上直接使用 `scripts/run_pace_thor_pipeline_smoke.sh`。
該 wrapper 會呼叫 PACE 專用的 `module load`，Thor 預設沒有 `module` 指令。

檢查輸出：

```bash
ls -lh results/thor_pipeline_smoke/null-test1.jsonl
ls -lh overlays/thor_pipeline_smoke/null-test1.mp4
head results/thor_pipeline_smoke/null-test1.jsonl
```

如果 null backend 失敗，先修 Python/OpenCV/路徑問題，不要進 ROS。

## 6. 跑 SAM3 backend smoke test

在 Thor 上有 GPU/PyTorch CUDA 後：

```bash
source .venv/bin/activate  # 若使用外部 venv，改成該 venv 的 activate 路徑

python -m sam_backend.profile_video \
  --model-id sam3-thor-smoke \
  --backend sam3 \
  --device cuda \
  --prompt monitor \
  --video videos/test1.mov \
  --max-frames 5 \
  --csv-output results/sam3-thor-smoke.csv \
  --overlay-output overlays/sam3-thor-smoke.mp4
```

檢查：

```bash
head results/sam3-thor-smoke.csv
ls -lh overlays/sam3-thor-smoke.mp4
```

## 7. 跑 EfficientSAM3 backend smoke test

先使用已知 checkpoint，例如：

```text
checkpoints/stage1_sam3p1/efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt
```

命令：

```bash
python -m sam_backend.profile_video \
  --model-id esam3-thor-smoke \
  --backend efficientsam3 \
  --checkpoint-path checkpoints/stage1_sam3p1/efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt \
  --device cuda \
  --backbone-type efficientvit \
  --model-name b0 \
  --text-encoder-type MobileCLIP-S0 \
  --text-encoder-context-length 16 \
  --text-encoder-pos-embed-table-size 16 \
  --prompt monitor \
  --video videos/test1.mov \
  --max-frames 5 \
  --csv-output results/esam3-thor-smoke.csv \
  --overlay-output overlays/esam3-thor-smoke.mp4
```

如果使用 MobileCLIP-S1 舊 checkpoint，可能需要：

```bash
--text-encoder-context-length 16 \
--text-encoder-pos-embed-table-size 77
```

## 8. 安裝 ROS 2 Jazzy

在 Ubuntu 24.04 Thor 上使用官方 ROS 2 Jazzy APT 安裝流程。安裝後確認：

```bash
source /opt/ros/jazzy/setup.bash
ros2 --help
python3 -c "import rclpy, sensor_msgs, std_msgs; print('ros ok')"
```

需要的 ROS 套件：

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-ros-base \
  ros-jazzy-cv-bridge \
  ros-jazzy-sensor-msgs \
  ros-jazzy-std-msgs \
  python3-colcon-common-extensions
```

如果要用 USB/CSI camera，還需要依實際相機補 `v4l2`、GStreamer 或 NVIDIA camera stack。

## 9. Build ROS workspace

Thor 上 ROS console script 可能仍使用 `/usr/bin/python3` shebang。為了讓它同時看到：

- repo root 的 `sam_backend`
- venv 裡的 `torch`
- editable EfficientSAM3 repo 裡的 `sam3`
- ROS Jazzy 的 `rclpy` / `cv_bridge`

請使用 repo 提供的 Thor ROS 環境腳本。預設假設：

```text
venv: ~/venvs/effisam3_venv_ros
EfficientSAM3 source: ~/efficientsam3/sam3
ROS: /opt/ros/jazzy/setup.bash
```

若路徑不同，先設定：

```bash
export THOR_VENV=/path/to/venv
export SAM3_SOURCE=/path/to/efficientsam3/sam3
```

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

python -m pip install "numpy>=1.26,<2"
python -m pip install -e . --no-deps
python -c "import cv2, rclpy, cv_bridge, torch, sam3, sam_backend; print('ros python ok')"

cd ros_ws
colcon build --symlink-install
cd ..
source scripts/source_thor_ros_env.sh
```

如果 `cv_bridge` 或 Python path 出問題，先確認：

```bash
which python
python -c "import cv2, rclpy, cv_bridge, torch, sam3, sam_backend; print('ok')"
head -1 ros_ws/install/sam_benchmark_ros/lib/sam_benchmark_ros/sam_backend_node
```

若 `head -1` 顯示 `#!/usr/bin/python3`，屬於正常狀況；`scripts/source_thor_ros_env.sh`
會把 venv site-packages 與 EfficientSAM3 source 加到 `PYTHONPATH`。

若先前曾執行 `python -m pip install -e .` 並把 NumPy 升到 2.x，`cv_bridge`
可能會出現 NumPy ABI error。修復：

```bash
source scripts/source_thor_ros_env.sh
python -m pip install --force-reinstall "numpy>=1.26,<2"
python -m pip install -e . --no-deps
```

## 10. 用影片測 ROS nodes

Terminal A：

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh
cd ros_ws

ros2 run sam_benchmark_ros video_stream_node --ros-args \
  -p video_path:=../videos/test1.mov \
  -p image_topic:=/image \
  -p fps:=15.0
```

Terminal B：

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

Terminal C：

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh
cd ros_ws

ros2 topic echo /sam/result_json
ros2 topic hz /image
```

先用 `backend:=null` 確認 ROS plumbing，再換 `sam3` 或 `efficientsam3`。

以下 ROS backend 指令假設沿用 Terminal B 的 shell。若開新 terminal，先重新執行：

```bash
cd ~/EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh
cd ros_ws
```

## 11. 換成 SAM3 ROS backend

```bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=sam3 \
  -p device:=cuda \
  -p prompt:=monitor \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json
```

目前 ROS node 只發布 JSON summary，不發布完整 mask image。先確認 latency 與穩定性，再加 mask topic。

## 12. 換成 EfficientSAM3 ROS backend

目前 ROS wrapper 只暴露 `backend`、`checkpoint_path`、`device`、`prompt` 與 topic
相關參數，尚未把 `backbone_type`、`model_name`、`text_encoder_type`、
`text_encoder_context_length`、`text_encoder_pos_embed_table_size` 暴露成 ROS parameters。

因此不要直接把第 7 節的 S0 ctx16 checkpoint 套到 ROS node；那個 checkpoint
通常需要 `text_encoder_context_length=16` 與 `text_encoder_pos_embed_table_size=16`。
在補齊 ROS parameters 前，EfficientSAM3 請先用第 7 節的非 ROS `profile_video`
驗證。

只有在 checkpoint 與 `BackendConfig` 預設值相容時，才使用以下 ROS 形式：

```bash
ros2 run sam_benchmark_ros sam_backend_node --ros-args \
  -p backend:=efficientsam3 \
  -p checkpoint_path:=../checkpoints/path/to/default-compatible-efficientsam3.pt \
  -p device:=cuda \
  -p prompt:=monitor \
  -p image_topic:=/image \
  -p result_topic:=/sam/result_json
```

## 13. 真實 camera 測試

影片測通後再換 camera：

```bash
ros2 run sam_benchmark_ros video_stream_node --ros-args \
  -p video_path:="" \
  -p image_topic:=/image \
  -p fps:=15.0
```

這會用 OpenCV device `0`。若 Thor camera 需要 GStreamer pipeline，應新增 camera-specific node，不要把 pipeline 字串硬塞進現有 smoke node。

## 14. TensorRT / ONNX

PACE 可以用來驗證 PyTorch backend 與 benchmark code。TensorRT engine 必須在 Thor 上 build：

```text
export ONNX: 可以在 PACE 或 Thor
build TensorRT engine: Thor
validate engine latency: Thor
```

原因是 TensorRT engine 與硬體、CUDA、TensorRT 版本強相關。

## 15. 建議驗收順序

1. `sam_backend.env_probe` 成功。
2. null backend pipeline smoke 成功。
3. SAM3 / EfficientSAM3 非 ROS `profile_video` 成功。
4. ROS `video_stream_node + sam_backend_node backend:=null` 成功。
5. ROS `backend:=sam3` 成功。
6. ROS `backend:=efficientsam3` 成功。
7. 換真實 camera。
8. 加 mask topic / visualization。
9. 做 TensorRT。
10. 做長時間穩定性測試。

## 16. 常見問題

### PyTorch CUDA 不可用

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

若是 `False`，先修 JetPack/PyTorch，不要進 ROS。

### ROS Python 與 venv 衝突

症狀：`rclpy`、`cv_bridge` 找不到，或 ABI 錯。

處理：

- 先用 system Python 跑 ROS node。
- backend 可先獨立成非 ROS process。
- 等 ROS plumbing 穩定後，再整理 venv 與 ROS Python path。

### EfficientSAM3 checkpoint shape mismatch

看 text encoder positional embedding shape：

- S0 ctx16 checkpoint 通常用 `pos_embed_table_size=16`。
- 部分 S1 checkpoint 需要 `pos_embed_table_size=77`。

### Overlay 有輸出但 mask 為空

這代表 code path 可跑，但 prompt 或模型沒有偵測到物件。先確認：

- prompt 是否為 `monitor`
- 影片中是否真的有 monitor
- score threshold 是否太高

## 17. Thor 上要回報的資訊

每次測試請記錄：

```text
date
git commit
JetPack version
Ubuntu version
Python version
PyTorch version
CUDA available
backend
checkpoint
video/camera source
prompt
CSV path
overlay path
mean latency
errors/log path
```
