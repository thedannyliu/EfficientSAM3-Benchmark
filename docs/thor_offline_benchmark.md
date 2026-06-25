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
checkpoints/efficientsam3_ft/efficient_sam3_tinyvit21_stage1_e32_h200_full_sam3.pt
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

SA-V fixed10 video/frame benchmark:

```bash
bash scripts/prepare_sav_fixed10_subset.sh
```

Outputs:

```text
data/manifests/sav_val_fixed10.jsonl
data/manifests/sav_val_fixed10_selection.json
```

For visually clearer demos, also prepare the salient SA-V subset:

```bash
bash scripts/prepare_sav_salient_subset.sh
```

Keep `data/`, `checkpoints/`, and `external/` under the agreed 300 GiB cap.

For YOLO COCO baselines, the script can also prepare the COCO fixed manifest and
Ultralytics weights before benchmarking. This avoids discovering missing data or
weights halfway through a run:

```bash
PREPARE_COCO=1 DOWNLOAD_WEIGHTS=1 LIMIT=0 YOLO_PRESET=quick EVAL_MODE=profile \
  bash scripts/run_thor_yolo_coco_suite.sh --dry-run
```

`PREPARE_COCO=1` calls `scripts/prepare_coco_fixed_subset.sh`, which downloads
COCO val2017/images plus annotations if needed and writes
`data/manifests/coco_val2017_fixed${COCO_COUNT}.jsonl`. The default
`COCO_COUNT` is `10`.

Use `YOLO_PRESET=small` or `YOLO_PRESET=all` to prepare the larger YOLOE/YOLO11
weights. `checkpoints/yoloe/yoloe-26m-seg.pt` is used when present; other
Ultralytics weights are resolved by model name and cached by Ultralytics.

## 6. Run The COCO Fixed10 Image Suite

This is the main SAM-family single-image benchmark. It runs SAM3,
EfficientSAM3 variants, SAM2.1 tiny/small/base-plus/large,
Efficient-SAM2.1 tiny/small/base-plus/large, EfficientTAM-Ti/S, and
MobileSAM registry variants `vit_t/vit_b/vit_l/vit_h` where their
checkpoints/repos exist. Official SAM3 currently has one image checkpoint in
this benchmark; the size sweep comes from EfficientSAM3, SAM2-family, and
MobileSAM registry variants.

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
results/thor/offline/coco/<run_id>/coco_suite_model_summary.csv
results/thor/offline/coco/<run_id>/coco_suite_component_summary.csv
results/thor/offline/coco/<run_id>/<model_id>/profile.csv
results/thor/offline/coco/<run_id>/<model_id>/summary.json
overlays/thor/offline/coco/<run_id>/<model_id>/*.png
```

Read `coco_suite_model_summary.csv` first. It is the concise model/prompt table
for comparing mIoU, FPS, latency, CUDA memory, parameter count, and model size
across the full run. Text and point prompts are separate rows. The summary
columns include:

```text
mean_total_ms
effective_fps
prompt_mode
miou_best
miou_merged
mean_cuda_peak_allocated_mb
mean_image_encoder_ms
mean_text_encoder_ms
mean_prompt_encoder_ms
mean_mask_decoder_ms
mean_grounding_ms
mean_detector_ms
params_*_m
weight_*_mb
checkpoint_file_mb
```

`effective_fps` is `1000 / mean_total_ms` for the profiled prediction rows.
It excludes model construction, checkpoint loading, image `cv2.imread`,
ground-truth mask decoding, and overlay writing. For each COCO row, `total_ms`
starts immediately before `backend.predict(frame_rgb, prompt)` and includes the
single-image model path, such as SAM3 `set_image` plus the selected text, point,
or box prompt.

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

## 7. Run YOLO COCO Mask Suite

This is the COCO image benchmark for YOLO mask baselines. It currently runs:

```text
YOLOE-seg: open-vocabulary text prompt -> instance masks
YOLO11-seg: closed-set COCO class prediction -> target-class masks
```

YOLO-World is intentionally not included in this script yet because it is an
open-vocabulary detector baseline, not a mask model. Add it later as a separate
box-IoU path or a YOLO-World + SAM2 hybrid mask path.

Fastest smoke run, starting from the smallest models:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
PREPARE_COCO=1 DOWNLOAD_WEIGHTS=1 LIMIT=1 YOLO_PRESET=quick \
  bash scripts/run_thor_yolo_coco_suite.sh
```

The quick preset runs:

```text
yoloe_26n_seg
yolo11n_seg
```

To include the small/medium models after the smoke run:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
LIMIT=0 YOLO_PRESET=small bash scripts/run_thor_yolo_coco_suite.sh
```

Run every YOLOE segmentation variant plus all YOLO11 segmentation baselines:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
PREPARE_COCO=1 DOWNLOAD_WEIGHTS=1 LIMIT=0 YOLO_PRESET=all \
  bash scripts/run_thor_yolo_coco_suite.sh
```

The `small` preset runs the quick models plus `yoloe_11s_seg`,
`yoloe_v8s_seg`, `yoloe_26s_seg`, `yolo11s_seg`, and `yolo11m_seg`.
The `all` preset adds every YOLOE segmentation variant currently listed by
Ultralytics:

```text
yoloe_11s_seg   yoloe-11s-seg.pt
yoloe_11m_seg   yoloe-11m-seg.pt
yoloe_11l_seg   yoloe-11l-seg.pt
yoloe_v8s_seg   yoloe-v8s-seg.pt
yoloe_v8m_seg   yoloe-v8m-seg.pt
yoloe_v8l_seg   yoloe-v8l-seg.pt
yoloe_26n_seg   yoloe-26n-seg.pt
yoloe_26s_seg   yoloe-26s-seg.pt
yoloe_26m_seg   checkpoints/yoloe/yoloe-26m-seg.pt
yoloe_26l_seg   yoloe-26l-seg.pt
yoloe_26x_seg   yoloe-26x-seg.pt
```

The `all` preset also keeps the YOLO11 segmentation closed-set baselines:

```text
yolo11n_seg
yolo11s_seg
yolo11m_seg
yolo11l_seg
yolo11x_seg
```

Important outputs:

```text
results/thor/offline/yolo_coco/<run_id>/yolo_coco_suite_summary.csv
results/thor/offline/yolo_coco/<run_id>/yolo_coco_model_summary.csv
results/thor/offline/yolo_coco/<run_id>/yolo_coco_component_summary.csv
results/thor/offline/yolo_coco/<run_id>/<model_id>/profile.csv
results/thor/offline/yolo_coco/<run_id>/<model_id>/summary.json
overlays/thor/offline/yolo_coco/<run_id>/<model_id>/*.png
```

Read `yolo_coco_model_summary.csv` first. It is the concise one-row-per-model
table for comparing mIoU, FPS, latency, CUDA memory, parameter count, and model
size across the full run. Use `yolo_coco_component_summary.csv` when you need
the full component/storage breakdown. The summary columns include:

```text
effective_fps
miou_best
miou_merged
mean_best_box_iou
mean_target_detection_count
mean_set_classes_ms
mean_predict_ms
mean_postprocess_ms
params_*
weight_*_bytes
params_total_m / weight_total_mb / checkpoint_file_mb
params_yolo_backbone / weight_yolo_backbone_bytes
params_yolo_neck / weight_yolo_neck_bytes
params_yolo_head / weight_yolo_head_bytes
yolo_backbone_layers / yolo_neck_layers / yolo_head_layers
checkpoint_file_bytes
```

`miou_best` is the mean IoU of the best predicted target mask against the
selected COCO annotation mask. `miou_merged` unions all target detections before
computing IoU. Overlays are written for every evaluated sample when
`EVAL_MODE=both` or `EVAL_MODE=overlay`.

For comparing model sizes, use `params_total_m`, `weight_total_mb`, and
`checkpoint_file_mb` in `yolo_coco_model_summary.csv`. The component summary
also includes `params_yolo_backbone_m`, `params_yolo_neck_m`,
`params_yolo_head_m`, and the matching `weight_yolo_*_mb` fields.

For Ultralytics YOLO models, component storage is reported with a pragmatic
split:

```text
params_yolo_backbone / weight_yolo_backbone_bytes = layers before the first neck Upsample/Concat layer
params_yolo_neck / weight_yolo_neck_bytes = neck/FPN/PAN layers before the final head
params_yolo_head / weight_yolo_head_bytes = final detect/segment head layer
params_segmentation_head / weight_segmentation_head_bytes = same final YOLO head
params_detector / weight_detector_bytes = whole YOLO model
checkpoint_file_bytes = local .pt file size when the path is discoverable
```

The summary also stores `yolo_backbone_layers`, `yolo_neck_layers`, and
`yolo_head_layers` so the split can be audited for each Ultralytics model.

Use lower confidence while debugging open-vocabulary localization:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
LIMIT=1 YOLO_PRESET=quick CONF=0.05 bash scripts/run_thor_yolo_coco_suite.sh
```

Run a single model directly:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
python -m sam_backend.profile_yolo_coco \
  --manifest data/manifests/coco_val2017_fixed10.jsonl \
  --model-id yolo11n_seg \
  --family yolo-seg \
  --weights yolo11n-seg.pt \
  --device cuda \
  --limit 1 \
  --eval-mode both \
  --csv-output "results/thor/offline/yolo_coco/${RUN_ID}/yolo11n_seg/profile.csv" \
  --summary-output "results/thor/offline/yolo_coco/${RUN_ID}/yolo11n_seg/summary.json" \
  --overlay-dir "overlays/thor/offline/yolo_coco/${RUN_ID}/yolo11n_seg"
```

## 8. Run All COCO Image Models And Merge Summary

Use this when you want one command that runs the YOLOE/YOLO11 COCO mask suite,
the SAM-family COCO suite, and then writes a single comparison CSV:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
PREPARE_COCO=1 DOWNLOAD_YOLO=1 DOWNLOAD_SAM=1 LIMIT=1 YOLO_PRESET=quick \
  SAM_MODELS="sam3 es3p1_weak_image_weak_text sam2p1_hiera_tiny efficient_sam2p1_hiera_tiny mobilesam_vit_t" \
  bash scripts/run_thor_coco_all_benchmarks.sh
```

After the smoke run, run the full matrix. This includes all YOLOE-seg sizes,
YOLO11 segmentation sizes, SAM2.1 sizes, Efficient-SAM2.1 sizes,
EfficientSAM3 variants, EfficientTAM-Ti/S, and MobileSAM `vit_t/vit_b/vit_l/vit_h`
when the checkpoints are available:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
PREPARE_COCO=1 DOWNLOAD_YOLO=1 DOWNLOAD_SAM=1 LIMIT=0 YOLO_PRESET=all \
  bash scripts/run_thor_coco_all_benchmarks.sh
```

Important outputs:

```text
results/thor/offline/coco_all/<run_id>/coco_all_model_summary.csv
results/thor/offline/coco_all/<run_id>/sam/coco_suite_component_summary.csv
results/thor/offline/coco_all/<run_id>/yolo/yolo_coco_model_summary.csv
overlays/thor/offline/coco_all/<run_id>/sam/<model_id>/*.png
overlays/thor/offline/coco_all/<run_id>/yolo/<model_id>/*.png
```

`coco_all_model_summary.csv` is the first table to inspect after the run. It
keeps one row per model/prompt mode and includes `suite`, `model_id`, `family`,
`backend`, `prompt_mode`, `effective_fps`, `mean_total_ms`, `miou_best`,
`miou_merged`, CUDA peak memory, total/component parameter counts, total/component
weight sizes, `checkpoint_file_mb`, and `source_csv`.

Optional model subsets are space-separated:

```bash
YOLO_MODELS="yoloe_26n_seg yolo11n_seg" \
SAM_MODELS="sam3 sam2p1_hiera_tiny mobilesam_vit_t" \
LIMIT=1 YOLO_PRESET=all bash scripts/run_thor_coco_all_benchmarks.sh
```

Use `DRY_RUN=1` to verify the expanded command matrix without loading models:

```bash
DRY_RUN=1 YOLO_PRESET=quick LIMIT=1 bash scripts/run_thor_coco_all_benchmarks.sh
```

## 9. Run A Smaller Image Sanity Check

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

## 10. Run SA-V Video Tracking

This uses official SA-V masks for IoU and overlay videos. SAM2-family models use
point prompts derived from the selected object's first available GT mask by
default. For VOS-style validation, use `--init-prompt mask`; Efficient-SAM2's
upstream SA-V/VOS inference initializes objects with `add_new_mask`, not a
single point. If the Efficient-SAM2 overlay looks like the first mask stays
fixed in image coordinates, rerun with `--init-prompt mask` before treating it
as a model or propagation bug.

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
  --manifest data/manifests/sav_val_fixed10.jsonl \
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
  --manifest data/manifests/sav_val_fixed10.jsonl \
  --eval-mode both \
  --max-frames 120 \
  --init-prompt mask \
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
  --manifest data/manifests/sav_val_fixed10.jsonl \
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

## 10b. Run SA-V Frame-By-Frame Image Profiling

This treats each annotated SA-V frame as an independent image segmentation
sample while reusing the same fixed video/object selection. Use this to compare
SA-V against COCO-style single-frame behavior.

Point prompt mode derives a fresh positive point from the selected object's GT
mask on each annotated frame:

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
  --csv-output "results/thor/offline/sav_frames/${RUN_ID}/mobilesam_vit_t/frames.csv" \
  --summary-output "results/thor/offline/sav_frames/${RUN_ID}/mobilesam_vit_t/summary.json" \
  --overlay-dir "overlays/thor/offline/sav_frames/${RUN_ID}/mobilesam_vit_t"
```

For text prompt mode, first create and fill the manual SA-V text prompt record,
then merge it into a text-enabled manifest:

```bash
python -m sam_backend.sav_text_prompts init \
  --manifest data/manifests/sav_val_fixed10.jsonl \
  --review-dir overlays/sav/review/sav_val_fixed10 \
  --output configs/datasets/sav_val_fixed10_text_prompts.json

# Edit configs/datasets/sav_val_fixed10_text_prompts.json, then:
python -m sam_backend.sav_text_prompts apply \
  --manifest data/manifests/sav_val_fixed10.jsonl \
  --prompts configs/datasets/sav_val_fixed10_text_prompts.json \
  --output data/manifests/sav_val_fixed10_text.jsonl
```

SAM3/EfficientSAM3 can then run text and point prompts from the same fixed SA-V
objects:

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
  --csv-output "results/thor/offline/sav_frames/${RUN_ID}/sam3/frames.csv" \
  --summary-output "results/thor/offline/sav_frames/${RUN_ID}/sam3/summary.json" \
  --overlay-dir "overlays/thor/offline/sav_frames/${RUN_ID}/sam3"
```

`frames_summary.csv` groups rows by video/object/prompt mode and reports mean
IoU, latency, FPS, CUDA memory, and component timing. SAM2-family and MobileSAM
image backends are point-only in this frame benchmark.

## 10c. Run SA-Co/VEval-SAV Per-Image Segmentation

This runs the SA-Co/VEval-SAV stream manifest as independent image
segmentation samples with `--stream-mode image_per_frame`. Point prompt mode
derives a fresh positive point from the current frame GT mask. Text prompt mode
uses the SA-Co noun phrase in the manifest.

Prepare or refresh the fixed SA-Co/VEval-SAV assets first:

```bash
RUN_SUITE=0 RUN_NULL_SMOKE=1 bash scripts/setup_thor_saco_stream_benchmark.sh
```

Run the distilled EfficientSAM3 TinyViT-21M checkpoint in both point and text
prompt modes, plus the existing EffiSAM-TV point prompt baseline:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="results/thor/saco_video_image_per_frame/${RUN_ID}"
OVERLAY_DIR="overlays/thor/saco_video_image_per_frame/${RUN_ID}"
THREE_MODEL_SUMMARY="${OUTPUT_DIR}/tinyvit21_three_model_summary.csv"

test -f checkpoints/efficientsam3_ft/efficient_sam3_tinyvit21_stage1_e32_h200_full_sam3.pt

sam-run-saco-stream-suite \
  --manifest data/manifests/saco_veval_sav_fixed20.jsonl \
  --gt-annotation-file "${SAM_BENCH_SCRATCH:-/storage/scratch1/9/eliu354/efficientsam3-benchmark}/data/annotation/saco_veval_sav_val.json" \
  --mode-set image_per_frame \
  --models \
    efficientsam3_tinyvit21_image_per_frame_point \
    efficientsam3_tinyvit21_image_per_frame_text \
    efficientsam3_tv_m_image_per_frame_point \
  --device cuda \
  --max-frames 120 \
  --output-dir "${OUTPUT_DIR}" \
  --overlay-dir "${OVERLAY_DIR}" \
  --skip-missing

python -m sam_backend.saco_model_summary \
  --offline-root "${OUTPUT_DIR}" \
  --offline-only \
  --models \
    efficientsam3_tinyvit21_image_per_frame_point \
    efficientsam3_tinyvit21_image_per_frame_text \
    efficientsam3_tv_m_image_per_frame_point \
  --output "${THREE_MODEL_SUMMARY}"
```

The fixed manifest is `data/manifests/saco_veval_sav_fixed20.jsonl`. With
`--max-frames 120`, this evaluates up to 2400 frame rows across 20 videos, with
the final count lower when selected videos have fewer valid frames. Use
`--max-frames 30` only for a quick first pass; fixed20 then produces at most
600 frame rows.

Command-only check without loading models:

```bash
DRY_RUN_ID="$(date +%Y%m%d-%H%M%S)-dryrun"
sam-run-saco-stream-suite \
  --manifest data/manifests/saco_veval_sav_fixed20.jsonl \
  --mode-set image_per_frame \
  --models \
    efficientsam3_tinyvit21_image_per_frame_point \
    efficientsam3_tinyvit21_image_per_frame_text \
    efficientsam3_tv_m_image_per_frame_point \
  --device cuda \
  --max-frames 1 \
  --output-dir "results/thor/saco_video_image_per_frame/${DRY_RUN_ID}" \
  --overlay-dir "overlays/thor/saco_video_image_per_frame/${DRY_RUN_ID}" \
  --dry-run
```

Model IDs:

```text
efficientsam3_tinyvit21_image_per_frame_point  distilled TinyViT-21M image encoder, point prompt
efficientsam3_tinyvit21_image_per_frame_text   distilled TinyViT-21M image encoder, text prompt
efficientsam3_tv_m_image_per_frame_point       existing EffiSAM-TV 11M, point prompt
```

The TinyViT-21M checkpoint is loaded through
`build_efficientsam3_image_model(..., backbone_type="tinyvit",
model_name="21m", load_from_HF=False)`. Its point prompt result is the main
quality signal right now. Text prompt runs are included for completeness, but
text IoU can be 0 until text/prompt KD is trained.

If the distilled source checkout differs from `external/efficientsam3` on Thor,
put or symlink that Thor-local checkout at `external/efficientsam3` before
running the suite. Do not use the PACE `/storage/home/...` project path on Thor.

Outputs:

```text
results/thor/saco_video_image_per_frame/<run_id>/saco_stream_suite_summary.csv
results/thor/saco_video_image_per_frame/<run_id>/tinyvit21_three_model_summary.csv
results/thor/saco_video_image_per_frame/<run_id>/<model_id>/frames.csv
results/thor/saco_video_image_per_frame/<run_id>/<model_id>/frames_summary.csv
results/thor/saco_video_image_per_frame/<run_id>/<model_id>/summary.json
results/thor/saco_video_image_per_frame/<run_id>/<model_id>/saco_veval_preds.json
results/thor/saco_video_image_per_frame/<run_id>/<model_id>/saco_veval_eval_res.json
overlays/thor/saco_video_image_per_frame/<run_id>/<model_id>/<source_id>/overlay.mp4
```

## 11. Run YOLOE-26M-seg + EdgeTAM Text-Prompt Tracking

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

By default this tracks the top-1 YOLOE detection to preserve older POC
behavior. For multiple screens/monitors, register multiple detections with
unique EdgeTAM object IDs:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
python -m sam_backend.profile_yoloe_edgetam \
  --video-path videos/test1.mov \
  --source-id test1 \
  --text-prompt monitor \
  --device cuda \
  --yoloe-conf 0.05 \
  --yoloe-iou 0.75 \
  --yoloe-max-det 50 \
  --no-yoloe-agnostic-nms \
  --max-objects 10 \
  --edgetam-init-prompt mask \
  --max-frames 240 \
  --yoloe-interval 20 \
  --autocast-bfloat16 \
  --csv-output "results/thor/offline/yoloe_edgetam_multi/${RUN_ID}/frames.csv" \
  --summary-output "results/thor/offline/yoloe_edgetam_multi/${RUN_ID}/summary.json" \
  --overlay-root "overlays/thor/offline/yoloe_edgetam_multi/${RUN_ID}" \
  --work-dir "results/thor/offline/yoloe_edgetam_multi/${RUN_ID}/work"
```

Use `frames_summary.csv` columns `yoloe_initial_detection_count`,
`yoloe_initial_tracked_count`, `track_count`, and `object_rows` to confirm that
YOLOE found multiple instances and EdgeTAM registered them. If
`yoloe_initial_detection_count` is still `1`, tune the YOLOE prompt, confidence,
IoU, NMS, or image size; EdgeTAM only tracks objects that were prompted.

SA-V manually labeled text prompts:

```bash
python -m sam_backend.profile_yoloe_edgetam \
  --manifest data/manifests/sav_val_fixed10_text.jsonl \
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

## 12. Record The Run

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
