# Jetson Thor Offline Benchmark and Profiling

This guide runs offline image/video benchmarks directly on Jetson Thor. It does
not use ROS. Use this path to measure model latency, component timing,
parameters, weight size, CUDA memory, IoU where ground truth is available, and
overlay artifacts.

Authoritative upstream install references:

- NVIDIA PyTorch for Jetson: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
- Jetson AGX Thor JetPack setup: https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_jetpack.html

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

Use this same helper in every Thor terminal, including non-ROS benchmark
terminals. It sources ROS, activates the venv, and updates `PYTHONPATH` so
Python can find this repo, venv packages, and the local EfficientSAM3 source.

If your paths differ, set them before sourcing:

```bash
export THOR_VENV=/path/to/venv
export SAM3_SOURCE=/path/to/efficientsam3/sam3
export THOR_ROS_SETUP=/opt/ros/jazzy/setup.bash
source scripts/source_thor_ros_env.sh
```

Install repo dependencies without replacing the already installed Jetson PyTorch
or ROS packages:

```bash
python -m pip install "numpy>=1.26,<2" opencv-python-headless pillow pyyaml huggingface_hub
python -m pip install timm tqdm ftfy==6.1.1 regex iopath typing_extensions psutil
python -m pip install -e . --no-deps
```

Do not use `requirements.txt` on Thor unless you intentionally want to manage
PyTorch yourself; it pins the PACE CUDA PyTorch packages.

If a missing package error appears later, install that package explicitly in the
same venv. Re-check `torch.cuda.is_available()` after any dependency change.

## 3. Install Model Source Repositories

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh
bash scripts/setup_model_repos.sh
```

This creates ignored editable checkouts under `external/`:

```text
external/sam3
external/efficientsam3
external/sam2
external/Efficient-SAM2
external/EfficientTAM
external/EdgeTAM
external/MobileSAM
```

## 4. Download Checkpoints

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

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
checkpoints/efficient-sam2/sam2.1_hiera_tiny.pt
checkpoints/efficienttam/efficienttam_ti.pt
checkpoints/efficienttam/efficienttam_s.pt
checkpoints/yoloe/yoloe-26m-seg.pt
checkpoints/edgetam/edgetam.pt
checkpoints/mobilesam/mobile_sam.pt
```

## 5. Prepare Fixed Datasets

COCO fixed10 image benchmark:

```bash
bash scripts/prepare_coco_fixed_subset.sh 10
```

Outputs:

```text
data/manifests/coco_val2017_fixed10.jsonl
data/manifests/coco_val2017_fixed10_selection.json
configs/datasets/coco_val2017_fixed10_prompts.json
```

SA-V fixed3 video benchmark:

```bash
bash scripts/download_sav_valtest_subset.sh val 3
```

Outputs:

```text
data/manifests/sav_val_fixed3.jsonl
data/manifests/sav_val_fixed3_selection.json
```

For visually clearer demos, also prepare the salient SA-V subset:

```bash
bash scripts/prepare_sav_salient_subset.sh
```

Keep `data/`, `checkpoints/`, and `external/` under the agreed 300 GiB cap.

## 6. Run The COCO Fixed10 Image Suite

This is the main single-image benchmark. It runs SAM3, EfficientSAM3 variants,
SAM2.1 tiny, Efficient-SAM2.1 tiny, EfficientTAM-Ti/S, and MobileSAM where
their checkpoints/repos exist.

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
python -m sam_backend.coco_suite \
  --manifest data/manifests/coco_val2017_fixed10.jsonl \
  --device cuda \
  --eval-mode both \
  --output-dir "results/thor/offline/coco/${RUN_ID}" \
  --overlay-dir "overlays/thor/offline/coco/${RUN_ID}" \
  --skip-missing
```

Important outputs:

```text
results/thor/offline/coco/<run_id>/coco_suite_summary.csv
results/thor/offline/coco/<run_id>/coco_suite_component_summary.csv
results/thor/offline/coco/<run_id>/<model_id>/profile.csv
results/thor/offline/coco/<run_id>/<model_id>/summary.json
overlays/thor/offline/coco/<run_id>/<model_id>/*.png
```

Read `coco_suite_component_summary.csv` first. It contains:

```text
mean_total_ms
effective_fps
miou_best
miou_merged
mean_cuda_peak_allocated_mb
mean_image_encoder_ms
mean_text_encoder_ms
mean_prompt_encoder_ms
mean_mask_decoder_ms
mean_grounding_ms
mean_detector_ms
params_*
weight_*_bytes
```

EfficientSAM3's upstream `external/efficientsam3/eval/eval_coco.py` evaluates
COCO with each annotation's ground-truth bounding box:

```text
model.predict_inst(..., box=gt_box, multimask_output=False)
```

That is an interactive box-prompt segmentation check, not text-prompt object
discovery. To reproduce that protocol for one EfficientSAM3 variant:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
python -m sam_backend.profile_coco \
  --manifest data/manifests/coco_val2017_fixed10.jsonl \
  --model-id es3_weak_image_strong_available_text_box \
  --backend efficientsam3 \
  --checkpoint-path checkpoints/stage1_all_converted/efficient_sam3_efficientvit-b0_mobileclip_s1.pth \
  --external-repo external/efficientsam3 \
  --backbone-type efficientvit \
  --model-name b0 \
  --text-encoder-type MobileCLIP-S1 \
  --text-encoder-context-length 16 \
  --text-encoder-pos-embed-table-size 77 \
  --prompt-mode box \
  --device cuda \
  --eval-mode both \
  --csv-output "results/thor/offline/coco_box/${RUN_ID}/profile.csv" \
  --summary-output "results/thor/offline/coco_box/${RUN_ID}/summary.json" \
  --overlay-dir "overlays/thor/offline/coco_box/${RUN_ID}"
```

## 7. Run A Smaller Image Sanity Check

Use this before the full suite when changing the environment:

```bash
python -m sam_backend.coco_suite \
  --manifest data/manifests/coco_val2017_fixed10.jsonl \
  --device cuda \
  --models sam3 es3p1_weak_image_weak_text sam2p1_hiera_tiny mobilesam_vit_t \
  --limit 1 \
  --eval-mode both \
  --output-dir results/thor/offline/smoke/coco \
  --overlay-dir overlays/thor/offline/smoke/coco \
  --skip-missing
```

This should produce one overlay per selected model and no failed rows in
`coco_suite_summary.csv`.

## 8. Run SA-V Point-Prompt Video Tracking

This uses official SA-V masks for IoU and overlay videos. SAM2-family models use
point prompts derived from the selected object's first available GT mask.

SAM2.1 tiny:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
python -m sam_backend.profile_sav_video \
  --model-id sam2p1_hiera_tiny \
  --backend sam2 \
  --external-repo external/sam2 \
  --checkpoint-path checkpoints/sam2/sam2.1_hiera_tiny.pt \
  --model-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda \
  --manifest data/manifests/sav_val_fixed3.jsonl \
  --eval-mode both \
  --max-frames 120 \
  --autocast-bfloat16 \
  --csv-output "results/thor/offline/sav/${RUN_ID}/sam2p1_hiera_tiny/frames.csv" \
  --summary-output "results/thor/offline/sav/${RUN_ID}/sam2p1_hiera_tiny/summary.json" \
  --overlay-root "overlays/thor/offline/sav/${RUN_ID}/sam2p1_hiera_tiny"
```

Efficient-SAM2.1 tiny:

```bash
python -m sam_backend.profile_sav_video \
  --model-id efficient_sam2p1_hiera_tiny \
  --backend efficient-sam2 \
  --external-repo external/Efficient-SAM2 \
  --checkpoint-path checkpoints/efficient-sam2/sam2.1_hiera_tiny.pt \
  --model-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda \
  --manifest data/manifests/sav_val_fixed3.jsonl \
  --eval-mode both \
  --max-frames 120 \
  --autocast-bfloat16 \
  --csv-output "results/thor/offline/sav/${RUN_ID}/efficient_sam2p1_hiera_tiny/frames.csv" \
  --summary-output "results/thor/offline/sav/${RUN_ID}/efficient_sam2p1_hiera_tiny/summary.json" \
  --overlay-root "overlays/thor/offline/sav/${RUN_ID}/efficient_sam2p1_hiera_tiny"
```

EfficientTAM-Ti:

```bash
python -m sam_backend.profile_sav_video \
  --model-id efficienttam_ti \
  --backend efficienttam \
  --external-repo external/EfficientTAM \
  --checkpoint-path checkpoints/efficienttam/efficienttam_ti.pt \
  --model-config configs/efficienttam/efficienttam_ti.yaml \
  --device cuda \
  --manifest data/manifests/sav_val_fixed3.jsonl \
  --eval-mode both \
  --max-frames 120 \
  --autocast-bfloat16 \
  --csv-output "results/thor/offline/sav/${RUN_ID}/efficienttam_ti/frames.csv" \
  --summary-output "results/thor/offline/sav/${RUN_ID}/efficienttam_ti/summary.json" \
  --overlay-root "overlays/thor/offline/sav/${RUN_ID}/efficienttam_ti"
```

On Thor, `sam_backend.profile_sav_video` disables EfficientTAM image encoder
`torch.compile` through `++model.compile_image_encoder=False`. The default
EfficientTAM builder enables compilation on GPUs with compute capability 8 or
newer, but the bundled Triton `ptxas` in the current Thor venv does not
recognize Thor's `sm_110a` target. Leaving compilation enabled causes a fatal
`ptxas fatal: Value 'sm_110a' is not defined for option 'gpu-name'` error and a
large PTX repro dump. Disabling compile is slower than the upstream optimized
path, but it is the stable Thor benchmark path until the Triton/ptxas stack
supports `sm_110a`.

Summarize all completed SA-V runs:

```bash
python -m sam_backend.sav_video_report \
  --root "results/thor/offline/sav/${RUN_ID}" \
  --output "results/thor/offline/sav/${RUN_ID}/sav_video_suite_summary.csv"
```

## 9. Run YOLOE-26M-seg + EdgeTAM Text-Prompt Tracking

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
  --csv-output "results/thor/offline/yoloe_edgetam/${RUN_ID}/frames.csv" \
  --summary-output "results/thor/offline/yoloe_edgetam/${RUN_ID}/summary.json" \
  --overlay-root "overlays/thor/offline/yoloe_edgetam/${RUN_ID}" \
  --work-dir "results/thor/offline/yoloe_edgetam/${RUN_ID}/work"
```

SA-V manually labeled text prompts:

```bash
python -m sam_backend.profile_yoloe_edgetam \
  --manifest data/manifests/sav_val_salient_fixed3_text.jsonl \
  --device cuda \
  --max-frames 240 \
  --yoloe-interval 20 \
  --autocast-bfloat16 \
  --csv-output "results/thor/offline/yoloe_edgetam_sav/${RUN_ID}/frames.csv" \
  --summary-output "results/thor/offline/yoloe_edgetam_sav/${RUN_ID}/summary.json" \
  --overlay-root "overlays/thor/offline/yoloe_edgetam_sav/${RUN_ID}" \
  --work-dir "results/thor/offline/yoloe_edgetam_sav/${RUN_ID}/work"
```

Check `frames_summary.csv` for first-mask latency, tracking FPS, YOLOE
validation latency, re-ground count, and top-1 versus GT-assisted localization
diagnostics.

EdgeTAM's Hydra builder searches configs under the `sam2` package. The default
config is therefore `configs/edgetam.yaml`; older repo-relative values such as
`external/EdgeTAM/sam2/configs/edgetam.yaml` are normalized by the benchmark
entrypoint before calling EdgeTAM.

## 10. Record The Run

For each Thor offline run, save these in your notes or PR:

```text
git branch and commit
JetPack/L4T version
Python version
torch/torchvision versions
GPU power mode
model IDs and checkpoint paths
dataset manifest path
result directory
overlay directory
mean latency/FPS
mIoU when GT exists
CUDA peak memory
```

To collect all completed Thor offline summaries into per-task CSVs and a model
storage/component table:

```bash
python -m sam_backend.thor_offline_report \
  --root results/thor/offline \
  --output-dir results/thor/offline/reports
```

This writes:

```text
results/thor/offline/reports/thor_offline_coco_summary.csv
results/thor/offline/reports/thor_offline_sav_summary.csv
results/thor/offline/reports/thor_offline_yoloe_edgetam_summary.csv
results/thor/offline/reports/thor_offline_all_summary.csv
results/thor/offline/reports/thor_offline_model_storage_components.csv
```

The task summaries keep the existing `params_*` and `weight_*` component
columns from the benchmark outputs and add checkpoint/asset storage totals. The
storage component CSV records each model asset path, existence, and file size.

Generated `results/`, `overlays/`, `data/`, `checkpoints/`, and `external/`
contents are intentionally ignored and should not be committed.
