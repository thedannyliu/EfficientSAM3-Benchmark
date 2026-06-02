# RTX 5090 Offline Benchmark and Profiling

This guide reproduces the offline benchmark environment on an RTX 5090
workstation. It mirrors `docs/thor_offline_benchmark.md`, but uses a standard
x86_64 Python virtual environment and the CUDA PyTorch wheels pinned in
`requirements.txt`.

It does not use ROS. Use this path to compare RTX 5090 results against Jetson
Thor for:

- COCO fixed10 single-image SAM-family benchmarks
- YOLO COCO mask baselines
- SA-V fixed10 native video tracking
- SA-V fixed10 frame-by-frame image profiling
- YOLOE + EdgeTAM text-prompt video tracking
- pipeline bottleneck profiling

Default output roots:

```text
results/rtx5090/offline/
overlays/rtx5090/offline/
```

## 1. System Requirements

Expected host:

```text
OS: Ubuntu or another x86_64 Linux workstation
GPU: NVIDIA RTX 5090
Python: 3.12
venv: ~/venvs/effisam3_venv_ros
NVIDIA driver: new enough for the CUDA 12.8 PyTorch wheels in requirements.txt
```

Check the machine:

```bash
nvidia-smi
python3.12 --version || python3 --version
git --version
```

Install basic tools first:

```bash
sudo apt update
sudo apt install -y git git-lfs wget curl
```

Then provide Python 3.12 with one of the options below.

Option A, Ubuntu 22.04 with deadsnakes PPA:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
python3.12 --version
```

Option B, Ubuntu 24.04 or any system whose apt repository provides Python 3.12:

```bash
sudo apt install -y python3.12 python3.12-venv python3.12-dev
python3.12 --version
```

Option C, conda or mamba when apt does not provide Python 3.12 or when you do
not want to add a PPA:

```bash
conda create -y -n effisam3-5090 python=3.12
conda activate effisam3-5090
PYTHON_BIN="$(which python)" bash scripts/setup_5090_offline_benchmark.sh
```

Option D, any existing Python 3.12 installation:

```bash
/path/to/python3.12 --version
PYTHON_BIN=/path/to/python3.12 bash scripts/setup_5090_offline_benchmark.sh
```

The setup script intentionally refuses Python versions other than 3.12 because
`pyproject.toml` requires `>=3.12,<3.13`.

## 2. Get The Repository

```bash
git clone git@github.com:thedannyliu/EfficientSAM3-Benchmark.git
cd EfficientSAM3-Benchmark
git fetch origin
git checkout main
git pull
git rev-parse HEAD
```

If the repo already exists:

```bash
cd EfficientSAM3-Benchmark
git checkout main
git pull
git rev-parse HEAD
```

Generated data, checkpoints, external repos, results, and overlays are ignored.
Do not commit local datasets, checkpoints, TensorRT engines, or videos.

## 3. Recommended Workstation Pipeline

Use this path for the company RTX 5090 workstation. The workstation has shown
that `hf auth login` can fail with `SSL: CERTIFICATE_VERIFY_FAILED`, while
plain Hugging Face `git` access works. Treat git/git-lfs as the primary
checkpoint download path and do not spend time debugging `hf auth login` unless
git also fails.

Assumptions:

- workstation is Ubuntu 22.04
- GPU is RTX 5090
- shell is at the repo root
- Python 3.12 is available as `python3.12`
- Hugging Face read token has access to `facebook/sam3` and
  `Simon7108528/EfficientSAM3`

### 3.1 Create The Python Environment

Create the virtual environment and install the repo first. Skip checkpoints,
datasets, and smoke tests until the environment is verified:

```bash
PYTHON_BIN=python3.12 \
  DOWNLOAD_CHECKPOINTS=0 \
  PREPARE_DATASETS=0 \
  PREPARE_SAV_TEXT=0 \
  RUN_SMOKE=0 \
  bash scripts/setup_5090_offline_benchmark.sh
```

Activate the environment:

```bash
source ~/venvs/effisam3_venv_ros/bin/activate
```

Verify Python and CUDA:

```bash
which python
python --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

Expected:

```text
python: .../venvs/effisam3_venv_ros/bin/python
python version: 3.12.x
cuda: True
gpu: NVIDIA GeForce RTX 5090
```

### 3.2 Test Hugging Face Git Access With SAM3

Use `facebook/sam3` as the smallest useful test. This does not download model
weights; it only checks whether git can read the repo HEAD with your token.

```bash
read -rsp "HF token: " HF_TOKEN
echo

GIT_TERMINAL_PROMPT=0 \
  git -c "http.extraHeader=Authorization: Bearer ${HF_TOKEN}" \
  ls-remote https://huggingface.co/facebook/sam3 HEAD

echo "exit_code=$?"
unset HF_TOKEN
```

Success looks like:

```text
<commit_sha>	HEAD
exit_code=0
```

If this returns `exit_code=0`, continue to checkpoint download. If it fails
with `401`, `403`, or a gated-repo message, check that the token has read
access and that the Hugging Face account has accepted access for:

```text
https://huggingface.co/facebook/sam3
https://huggingface.co/Simon7108528/EfficientSAM3
```

If it fails with `SSL: CERTIFICATE_VERIFY_FAILED`, git is also blocked by the
company TLS environment. Stop there and get the workstation root CA from IT or
the machine administrator before trying checkpoint download.

### 3.3 Download Hugging Face Checkpoints With Git-LFS

Install git-lfs and use the repository helper. This avoids `hf auth login`,
clones into `external/hf-checkpoints/`, pulls only the required LFS files, and
copies them into `checkpoints/`.

```bash
sudo apt install -y git-lfs
git lfs install
git lfs version

read -rsp "HF token: " HF_TOKEN
echo

HF_TOKEN="${HF_TOKEN}" \
  HF_GIT_TIMEOUT=900 \
  bash scripts/download_hf_checkpoints_via_git.sh

unset HF_TOKEN
```

Verify the Hugging Face checkpoints:

```bash
ls -lh checkpoints/sam3/
ls -lh checkpoints/stage1_sam3p1/
ls -lh checkpoints/stage1_all_converted/

python - <<'PY'
from pathlib import Path

required = [
    "checkpoints/sam3/config.json",
    "checkpoints/sam3/sam3.pt",
    "checkpoints/stage1_sam3p1/efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt",
    "checkpoints/stage1_sam3p1/efficient_sam3p1_efficientvit_l_mobileclip_s0_ctx16.pt",
    "checkpoints/stage1_all_converted/efficient_sam3_efficientvit-b0_mobileclip_s1.pth",
    "checkpoints/stage1_all_converted/efficient_sam3_efficientvit-b2_mobileclip_s1.pth",
]

for item in required:
    path = Path(item)
    if not path.exists():
        raise SystemExit(f"missing: {path}")
    size = path.stat().st_size
    print(path, size)
    if size < 1024 * 1024 and path.suffix != ".json":
        raise SystemExit(f"too small, likely incomplete: {path}")
PY
```

Do not continue until this verification passes.

### 3.4 Download Remaining Model Assets

After SAM3 and EfficientSAM3 are in place, fetch the remaining checkpoints:

```bash
bash scripts/download_sam2_family_checkpoints.sh
bash scripts/download_yoloe_edgetam_mobilesam_assets.sh
bash scripts/check_storage_budget.sh 300 data checkpoints external
```

### 3.5 Prepare Datasets And Run Smoke

Finish setup with checkpoint download disabled, because the Hugging Face
checkpoints were already downloaded through git-lfs:

```bash
PYTHON_BIN=python3.12 \
  DOWNLOAD_CHECKPOINTS=0 \
  PREPARE_DATASETS=1 \
  PREPARE_SAV_TEXT=1 \
  RUN_SMOKE=1 \
  bash scripts/setup_5090_offline_benchmark.sh
```

### 3.6 Reuse After A Failed Attempt

If a previous git-lfs checkpoint download was interrupted, remove only the
partial Hugging Face clones and rerun section 3.3:

```bash
rm -rf external/hf-checkpoints/facebook__sam3
rm -rf external/hf-checkpoints/Simon7108528__EfficientSAM3
```

If an earlier attempt created a broken `.venv` with Ubuntu 22.04's default
`python3`, remove it and rerun section 3.1:

```bash
rm -rf .venv
```

## 4. Manual Environment Setup

Use this only if you do not want the setup script.

```bash
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
mkdir -p ~/venvs
"${PYTHON_BIN}" -m venv ~/venvs/effisam3_venv_ros
source ~/venvs/effisam3_venv_ros/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt` uses the PyTorch CUDA 12.8 wheel index and pins:

```text
torch==2.10.0
torchvision==0.25.0
torchaudio==2.10.0
```

Do not use the Thor package policy here. RTX 5090 should use the normal
workstation CUDA PyTorch packages from `requirements.txt`.

## 5. Install Model Source Repositories

```bash
source ~/venvs/effisam3_venv_ros/bin/activate
bash scripts/setup_model_repos.sh
```

This creates ignored editable checkouts:

```text
external/sam3
external/efficientsam3
external/sam2
external/Efficient-SAM2
external/EfficientTAM
external/EdgeTAM
external/MobileSAM
```

## 6. Download Checkpoints

For the company RTX 5090 workstation, use git-lfs for SAM3 and EfficientSAM3.
This path does not require `hf auth login`, which can fail under the company
SSL environment.

```bash
source ~/venvs/effisam3_venv_ros/bin/activate
mkdir -p logs/rtx5090_debug
set -o pipefail
```

### 6.1 Verify SAM3 Git Access

Use `facebook/sam3` as the first test. This is the same access path used by the
download script, but it does not download LFS files yet.

```bash
read -rsp "HF token: " HF_TOKEN
echo

GIT_TERMINAL_PROMPT=0 \
  git -c "http.extraHeader=Authorization: Bearer ${HF_TOKEN}" \
  ls-remote https://huggingface.co/facebook/sam3 HEAD 2>&1 \
  | tee logs/rtx5090_debug/git_ls_remote_sam3.log

echo "exit_code=${PIPESTATUS[0]}"
unset HF_TOKEN
```

Continue only if `exit_code=0`.

### 6.2 Download SAM3 And EfficientSAM3

```bash
sudo apt install -y git-lfs
git lfs install
git lfs version

read -rsp "HF token: " HF_TOKEN
echo

HF_TOKEN="${HF_TOKEN}" \
  HF_GIT_TIMEOUT=900 \
  bash scripts/download_hf_checkpoints_via_git.sh 2>&1 \
  | tee logs/rtx5090_debug/download_hf_git_lfs.log

unset HF_TOKEN
```

The script clones these repos under `external/hf-checkpoints/`, pulls only the
required LFS files, and copies them to the benchmark checkpoint paths:

```text
external/hf-checkpoints/facebook__sam3
external/hf-checkpoints/Simon7108528__EfficientSAM3
```

```bash
ls -lh checkpoints/sam3/
ls -lh checkpoints/stage1_sam3p1/
ls -lh checkpoints/stage1_all_converted/
python - <<'PY'
from pathlib import Path

required = [
    "checkpoints/sam3/config.json",
    "checkpoints/sam3/sam3.pt",
    "checkpoints/stage1_sam3p1/efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt",
    "checkpoints/stage1_sam3p1/efficient_sam3p1_efficientvit_l_mobileclip_s0_ctx16.pt",
    "checkpoints/stage1_all_converted/efficient_sam3_efficientvit-b0_mobileclip_s1.pth",
    "checkpoints/stage1_all_converted/efficient_sam3_efficientvit-b2_mobileclip_s1.pth",
]

for path_str in required:
    path = Path(path_str)
    if not path.exists():
        raise SystemExit(f"missing: {path}")
    size = path.stat().st_size
    print(path, size)
    if size < 1024 * 1024 and path.suffix != ".json":
        raise SystemExit(f"too small, likely incomplete: {path}")
PY
```

Do not continue until this verification passes.

### 6.3 Retry After An Interrupted Git-LFS Download

Remove only partial Hugging Face clones from previous failed git-lfs attempts:

```bash
rm -rf external/hf-checkpoints/facebook__sam3
rm -rf external/hf-checkpoints/Simon7108528__EfficientSAM3
```

Then rerun section 6.2. If the git-lfs log says `401`, `403`, or a gated-repo
message, fix Hugging Face token access. If it says `SSL: CERTIFICATE_VERIFY_FAILED`,
git is also blocked by the company TLS environment; get the workstation root CA
before retrying.

### 6.4 Download The Remaining Checkpoints

After SAM3 and EfficientSAM3 pass verification, download the remaining model
assets:

```bash
bash scripts/download_sam2_family_checkpoints.sh 2>&1 \
  | tee logs/rtx5090_debug/download_sam2_family.log

bash scripts/download_yoloe_edgetam_mobilesam_assets.sh 2>&1 \
  | tee logs/rtx5090_debug/download_yoloe_edgetam_mobilesam.log

bash scripts/check_storage_budget.sh 300 data checkpoints external
```

Expected key files:

```text
checkpoints/sam3/sam3.pt
checkpoints/stage1_sam3p1/efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt
checkpoints/stage1_sam3p1/efficient_sam3p1_efficientvit_l_mobileclip_s0_ctx16.pt
checkpoints/stage1_all_converted/efficient_sam3_efficientvit-b0_mobileclip_s1.pth
checkpoints/stage1_all_converted/efficient_sam3_efficientvit-b2_mobileclip_s1.pth
checkpoints/sam2/sam2.1_hiera_tiny.pt
checkpoints/sam2/sam2.1_hiera_small.pt
checkpoints/sam2/sam2.1_hiera_base_plus.pt
checkpoints/sam2/sam2.1_hiera_large.pt
checkpoints/efficient-sam2/sam2.1_hiera_tiny.pt
checkpoints/efficienttam/efficienttam_ti.pt
checkpoints/efficienttam/efficienttam_s.pt
checkpoints/yoloe/yoloe-26m-seg.pt
checkpoints/edgetam/edgetam.pt
checkpoints/mobilesam/mobile_sam.pt
```

## 7. Prepare Fixed Datasets

COCO fixed10:

```bash
bash scripts/prepare_coco_fixed_subset.sh 10
```

Outputs:

```text
data/manifests/coco_val2017_fixed10.jsonl
data/manifests/coco_val2017_fixed10_selection.json
configs/datasets/coco_val2017_fixed10_prompts.json
```

SA-V fixed10:

```bash
bash scripts/prepare_sav_fixed10_subset.sh
```

Outputs:

```text
data/sa-v/sav_val_fixed10/
data/manifests/sav_val_fixed10.jsonl
data/manifests/sav_val_fixed10_selection.json
configs/datasets/sav_val_fixed10_text_prompts.json
data/manifests/sav_val_fixed10_text.jsonl
```

Only 10 SA-V videos should remain in the fixed10 root:

```bash
find data/sa-v/sav_val_fixed10/JPEGImages_24fps -mindepth 1 -maxdepth 1 -type d | wc -l
find data/sa-v/sav_val_fixed10/Annotations_6fps -mindepth 1 -maxdepth 1 -type d | wc -l
wc -l data/manifests/sav_val_fixed10.jsonl
```

The expected counts are all `10`. The downloaded `sav_val.tar` archive is
removed by default unless `KEEP_SAV_ARCHIVE=1` is set.

## 8. Run The COCO Fixed10 Image Suite

```bash
source ~/venvs/effisam3_venv_ros/bin/activate
RUN_ID="$(date +%Y%m%d-%H%M%S)"

python -m sam_backend.coco_suite \
  --manifest data/manifests/coco_val2017_fixed10.jsonl \
  --device cuda \
  --eval-mode both \
  --output-dir "results/rtx5090/offline/coco/${RUN_ID}" \
  --overlay-dir "overlays/rtx5090/offline/coco/${RUN_ID}" \
  --skip-missing
```

Read first:

```text
results/rtx5090/offline/coco/<run_id>/coco_suite_model_summary.csv
results/rtx5090/offline/coco/<run_id>/coco_suite_component_summary.csv
```

For a quick smoke:

```bash
python -m sam_backend.coco_suite \
  --manifest data/manifests/coco_val2017_fixed10.jsonl \
  --device cuda \
  --models sam3 sam2p1_hiera_tiny mobilesam_vit_t \
  --limit 1 \
  --eval-mode profile \
  --output-dir results/rtx5090/offline/smoke/coco \
  --skip-missing
```

## 9. Run YOLO COCO Mask Suite

Fast smoke:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
PREPARE_COCO=1 DOWNLOAD_WEIGHTS=1 LIMIT=1 YOLO_PRESET=quick \
  OUTPUT_DIR="results/rtx5090/offline/yolo_coco/${RUN_ID}" \
  OVERLAY_DIR="overlays/rtx5090/offline/yolo_coco/${RUN_ID}" \
  bash scripts/run_thor_yolo_coco_suite.sh
```

Full YOLO sweep:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
PREPARE_COCO=1 DOWNLOAD_WEIGHTS=1 LIMIT=0 YOLO_PRESET=all \
  OUTPUT_DIR="results/rtx5090/offline/yolo_coco/${RUN_ID}" \
  OVERLAY_DIR="overlays/rtx5090/offline/yolo_coco/${RUN_ID}" \
  bash scripts/run_thor_yolo_coco_suite.sh
```

Read first:

```text
results/rtx5090/offline/yolo_coco/<run_id>/yolo_coco_model_summary.csv
results/rtx5090/offline/yolo_coco/<run_id>/yolo_coco_component_summary.csv
```

## 10. Run All COCO Image Models And Merge Summary

Smoke:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
PREPARE_COCO=1 DOWNLOAD_YOLO=1 DOWNLOAD_SAM=1 LIMIT=1 YOLO_PRESET=quick \
  SAM_MODELS="sam3 sam2p1_hiera_tiny mobilesam_vit_t" \
  OUTPUT_ROOT="results/rtx5090/offline/coco_all/${RUN_ID}" \
  OVERLAY_ROOT="overlays/rtx5090/offline/coco_all/${RUN_ID}" \
  bash scripts/run_thor_coco_all_benchmarks.sh
```

Full matrix:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
PREPARE_COCO=1 DOWNLOAD_YOLO=1 DOWNLOAD_SAM=1 LIMIT=0 YOLO_PRESET=all \
  OUTPUT_ROOT="results/rtx5090/offline/coco_all/${RUN_ID}" \
  OVERLAY_ROOT="overlays/rtx5090/offline/coco_all/${RUN_ID}" \
  bash scripts/run_thor_coco_all_benchmarks.sh
```

Read first:

```text
results/rtx5090/offline/coco_all/<run_id>/coco_all_model_summary.csv
```

## 11. Run SA-V Fixed10 Native Video Tracking

SAM2.1 tiny point-prompt tracking:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
python -m sam_backend.profile_sav_video \
  --model-id sam2p1_hiera_tiny \
  --backend sam2 \
  --external-repo external/sam2 \
  --checkpoint-path checkpoints/sam2/sam2.1_hiera_tiny.pt \
  --model-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda \
  --manifest data/manifests/sav_val_fixed10.jsonl \
  --eval-mode both \
  --max-frames 120 \
  --autocast-bfloat16 \
  --csv-output "results/rtx5090/offline/sav/${RUN_ID}/sam2p1_hiera_tiny/frames.csv" \
  --summary-output "results/rtx5090/offline/sav/${RUN_ID}/sam2p1_hiera_tiny/summary.json" \
  --overlay-root "overlays/rtx5090/offline/sav/${RUN_ID}/sam2p1_hiera_tiny"
```

Efficient-SAM2.1 tiny mask initialization:

```bash
python -m sam_backend.profile_sav_video \
  --model-id efficient_sam2p1_hiera_tiny \
  --backend efficient-sam2 \
  --external-repo external/Efficient-SAM2 \
  --checkpoint-path checkpoints/efficient-sam2/sam2.1_hiera_tiny.pt \
  --model-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda \
  --manifest data/manifests/sav_val_fixed10.jsonl \
  --eval-mode both \
  --max-frames 120 \
  --init-prompt mask \
  --autocast-bfloat16 \
  --csv-output "results/rtx5090/offline/sav/${RUN_ID}/efficient_sam2p1_hiera_tiny/frames.csv" \
  --summary-output "results/rtx5090/offline/sav/${RUN_ID}/efficient_sam2p1_hiera_tiny/summary.json" \
  --overlay-root "overlays/rtx5090/offline/sav/${RUN_ID}/efficient_sam2p1_hiera_tiny"
```

Summarize:

```bash
python -m sam_backend.sav_video_report \
  --root "results/rtx5090/offline/sav/${RUN_ID}" \
  --output "results/rtx5090/offline/sav/${RUN_ID}/sav_video_suite_summary.csv"
```

## 12. Run SA-V Fixed10 Frame-By-Frame Image Profiling

Point prompt mode for point-only backends:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
python -m sam_backend.profile_sav_frames \
  --manifest data/manifests/sav_val_fixed10.jsonl \
  --model-id mobilesam_vit_t_sav_frames_point \
  --backend mobilesam \
  --checkpoint-path checkpoints/mobilesam/mobile_sam.pt \
  --external-repo external/MobileSAM \
  --mobile-sam-model-type vit_t \
  --device cuda \
  --prompt-mode point \
  --max-frames 30 \
  --csv-output "results/rtx5090/offline/sav_frames/${RUN_ID}/mobilesam_vit_t/frames.csv" \
  --summary-output "results/rtx5090/offline/sav_frames/${RUN_ID}/mobilesam_vit_t/summary.json" \
  --overlay-dir "overlays/rtx5090/offline/sav_frames/${RUN_ID}/mobilesam_vit_t"
```

Text and point prompt mode for SAM3-style backends:

```bash
python -m sam_backend.profile_sav_frames \
  --manifest data/manifests/sav_val_fixed10_text.jsonl \
  --model-id sam3_sav_frames_text_point \
  --backend sam3 \
  --checkpoint-path checkpoints/sam3/sam3.pt \
  --external-repo external/sam3 \
  --device cuda \
  --prompt-mode both \
  --max-frames 30 \
  --csv-output "results/rtx5090/offline/sav_frames/${RUN_ID}/sam3/frames.csv" \
  --summary-output "results/rtx5090/offline/sav_frames/${RUN_ID}/sam3/summary.json" \
  --overlay-dir "overlays/rtx5090/offline/sav_frames/${RUN_ID}/sam3"
```

Read:

```text
results/rtx5090/offline/sav_frames/<run_id>/<model_id>/frames.csv
results/rtx5090/offline/sav_frames/<run_id>/<model_id>/frames_summary.csv
```

## 13. Run YOLOE-26M-seg + EdgeTAM Text-Prompt Tracking

SA-V fixed10 text-prompt tracking:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
python -m sam_backend.profile_yoloe_edgetam \
  --manifest data/manifests/sav_val_fixed10_text.jsonl \
  --device cuda \
  --max-frames 240 \
  --yoloe-interval 20 \
  --autocast-bfloat16 \
  --csv-output "results/rtx5090/offline/yoloe_edgetam_sav/${RUN_ID}/frames.csv" \
  --summary-output "results/rtx5090/offline/yoloe_edgetam_sav/${RUN_ID}/summary.json" \
  --overlay-root "overlays/rtx5090/offline/yoloe_edgetam_sav/${RUN_ID}" \
  --work-dir "results/rtx5090/offline/yoloe_edgetam_sav/${RUN_ID}/work"
```

Recorded video POC:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
python -m sam_backend.profile_yoloe_edgetam \
  --video-path videos/test1.mov \
  --source-id test1 \
  --text-prompt monitor \
  --device cuda \
  --max-frames 240 \
  --yoloe-interval 20 \
  --autocast-bfloat16 \
  --csv-output "results/rtx5090/offline/yoloe_edgetam/${RUN_ID}/frames.csv" \
  --summary-output "results/rtx5090/offline/yoloe_edgetam/${RUN_ID}/summary.json" \
  --overlay-root "overlays/rtx5090/offline/yoloe_edgetam/${RUN_ID}" \
  --work-dir "results/rtx5090/offline/yoloe_edgetam/${RUN_ID}/work"
```

## 14. Run Pipeline Bottleneck Profiling

Use this to compare the RTX 5090 against Jetson Thor when small models appear
to hit a fixed FPS ceiling.

```bash
RUN_ID="rtx5090-$(date +%Y%m%d-%H%M%S)"
LIMIT=10 \
WARMUP=5 \
REPEAT=5 \
INPUT_MODE=preload \
WITH_GT=0 \
TORCH_PROFILER=0 \
IMGSZ_LIST="320 640 1024" \
OUTPUT_ROOT="results/rtx5090/offline/bottleneck/${RUN_ID}" \
bash scripts/run_pipeline_bottleneck_matrix.sh
```

Short diagnostic run with PyTorch profiler:

```bash
RUN_ID="rtx5090-prof-$(date +%Y%m%d-%H%M%S)"
LIMIT=3 \
WARMUP=2 \
REPEAT=1 \
INPUT_MODE=preload \
WITH_GT=0 \
TORCH_PROFILER=1 \
IMGSZ_LIST="640" \
OUTPUT_ROOT="results/rtx5090/offline/bottleneck/${RUN_ID}" \
bash scripts/run_pipeline_bottleneck_matrix.sh
```

Read:

```text
results/rtx5090/offline/bottleneck/<run_id>/bottleneck_matrix_summary.csv
```

Compare these columns against Thor:

```text
effective_pipeline_fps
mean_total_pipeline_ms
mean_predict_wall_ms
mean_predict_cuda_window_ms
mean_predict_torch_cuda_kernel_ms
gpu_time_source
gpu_time_fraction_of_pipeline
mean_predict_cpu_gap_ms
bottleneck_hint
```

## 15. Record The Run

For each RTX 5090 run, record:

```text
git branch and commit
OS and kernel
nvidia-smi output
Python version
torch/torchvision versions
GPU name and driver version
model IDs and checkpoint paths
dataset manifest path
result directory
overlay directory
mean latency/FPS
mIoU where GT exists
CUDA peak memory
any Nsight trace path
```

Suggested environment record:

```bash
mkdir -p results/rtx5090/offline/run_notes
{
  git rev-parse HEAD
  uname -a
  nvidia-smi
  python -m sam_backend.env_probe
} > "results/rtx5090/offline/run_notes/$(date +%Y%m%d-%H%M%S)-env.txt"
```

Generated `results/`, `overlays/`, `data/`, `checkpoints/`, `external/`,
TensorRT engines, and local videos should not be committed.
