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

## 3. One-Command Setup

The one-command setup does the repeatable workstation work:

- creates `~/venvs/effisam3_venv_ros`
- installs `requirements.txt`
- installs this repo editable
- clones and installs external model repos under `external/`
- downloads checkpoints under `checkpoints/`
- prepares COCO fixed10 and SA-V fixed10
- reapplies tracked SA-V fixed10 text prompts
- runs a lightweight null-backend smoke check

Checkpoint download requires Hugging Face auth for SAM3 and EfficientSAM3.
For a first-time setup, create a read token at
`https://huggingface.co/settings/tokens`, then either export it for this shell:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"
PYTHON_BIN=python3.12 bash scripts/setup_5090_offline_benchmark.sh
```

Or create the environment first, log in interactively, and rerun setup:

```bash
PYTHON_BIN=python3.12 DOWNLOAD_CHECKPOINTS=0 PREPARE_DATASETS=0 RUN_SMOKE=0 bash scripts/setup_5090_offline_benchmark.sh
source ~/venvs/effisam3_venv_ros/bin/activate
hf auth login
hf auth whoami
PYTHON_BIN=python3.12 bash scripts/setup_5090_offline_benchmark.sh
```

If `hf auth whoami` fails with `CERTIFICATE_VERIFY_FAILED` or
`self-signed certificate in certificate chain`, the workstation is behind an
HTTPS-inspecting proxy or has a local root CA that Python does not trust yet.
Get the workstation's trusted root CA from IT or the machine administrator,
save it as a PEM file, then rerun with:

```bash
mkdir -p ~/certs
# Put the organization/root CA PEM at ~/certs/workstation-root-ca.pem first.
openssl x509 -in ~/certs/workstation-root-ca.pem -noout -subject -issuer -dates

HF_CA_BUNDLE=~/certs/workstation-root-ca.pem \
  PYTHON_BIN=python3.12 \
  bash scripts/setup_5090_offline_benchmark.sh
```

`HF_CA_BUNDLE` is propagated to `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, and
`CURL_CA_BUNDLE` for Python, Hugging Face, curl, and related download helpers.

Run:

```bash
bash scripts/setup_5090_offline_benchmark.sh
```

Useful overrides:

```bash
PYTHON_BIN=python3.12 bash scripts/setup_5090_offline_benchmark.sh
VENV_DIR=.venv bash scripts/setup_5090_offline_benchmark.sh
PYTHON_BIN="$(which python)" bash scripts/setup_5090_offline_benchmark.sh
DOWNLOAD_CHECKPOINTS=0 PREPARE_DATASETS=0 bash scripts/setup_5090_offline_benchmark.sh
HF_CA_BUNDLE=~/certs/workstation-root-ca.pem bash scripts/setup_5090_offline_benchmark.sh
CHECK_HF_AUTH=0 bash scripts/setup_5090_offline_benchmark.sh
RUN_SMOKE=0 bash scripts/setup_5090_offline_benchmark.sh
STORAGE_LIMIT_GIB=300 bash scripts/setup_5090_offline_benchmark.sh
```

After setup, activate the environment in every terminal:

```bash
source ~/venvs/effisam3_venv_ros/bin/activate
```

If an earlier attempt created a broken `.venv` with Ubuntu 22.04's default
`python3`, it can be removed. The default RTX 5090 setup now uses the same
venv path as Thor, `~/venvs/effisam3_venv_ros`, unless `VENV_DIR` is set:

```bash
rm -rf .venv
PYTHON_BIN=python3.12 bash scripts/setup_5090_offline_benchmark.sh
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("cuda:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
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

```bash
source ~/venvs/effisam3_venv_ros/bin/activate

bash scripts/download_sam3_checkpoint.sh
bash scripts/download_efficientsam3_checkpoints.sh
bash scripts/download_sam2_family_checkpoints.sh
bash scripts/download_yoloe_edgetam_mobilesam_assets.sh
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

SAM3 and EfficientSAM3 downloads use Hugging Face. If auth fails:

```bash
source ~/venvs/effisam3_venv_ros/bin/activate
hf auth login
hf auth whoami
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
