# Pipeline Bottleneck Profiling on Jetson Thor and RTX 5090

This guide is for diagnosing the observed latency shape where larger models are
slower, but the smallest models still appear capped near a fixed FPS. It adds a
pipeline profiler that separates fixed per-frame cost from model GPU time.

The profiler answers this first question:

```text
Is the frame time mostly GPU model work, or mostly fixed CPU/wrapper/copy/postprocess work?
```

It writes:

```text
profile.csv       per-sample pipeline timings
summary.json      mean timings, FPS, and a bottleneck_hint
```

Key fields:

```text
total_pipeline_ms             read + color convert + GT + prompt + predict + postprocess
pipeline_without_read_ms      same as above, excluding cv2.imread
predict_wall_ms               wall-clock model call time
predict_cuda_window_ms        CUDA stream window around the model call
predict_torch_cuda_kernel_ms  optional PyTorch profiler CUDA kernel self time
predict_cpu_gap_ms            predict_wall_ms - best available GPU timing
postprocess_ms                mask/box extraction, filtering, IoU if enabled
gpu_time_fraction_of_pipeline GPU timing / total_pipeline_ms
bottleneck_hint               coarse diagnosis from the timing split
```

`predict_cuda_window_ms` is only available when CUDA is active. It is a stream
window around a black-box model call, not pure kernel busy time. For a short
diagnostic run, set `TORCH_PROFILER=1` or pass `--with-torch-profiler`; then
`predict_torch_cuda_kernel_ms` becomes the preferred GPU-time estimate.

If the best available GPU timing is much smaller than `predict_wall_ms`, the
model call is spending significant time in CPU wrappers, synchronization,
copies, Python overhead, or framework overhead.

## Diagnosis Rules

Use the summary as a first-pass classifier, not a final proof:

```text
gpu_bound_compute_or_memory
  GPU timing dominates the pipeline. Use Nsight Compute to separate compute
  saturation from memory bandwidth pressure.

cpu_wrapper_sync_or_copy_bound
  Wall time is much larger than GPU timing. Inspect Python/framework
  overhead, cudaMemcpy, synchronization, and CPU preprocessing inside wrappers.

postprocess_bound
  Mask extraction, resizing, filtering, or IoU dominates. Deployment latency
  should be measured with GT disabled.

preprocess_or_fixed_pipeline_bound
  Non-model stages dominate. Focus on decode, color conversion, resize,
  tensor conversion, and pipeline serialization.

mixed_or_kernel_launch_bound
  No single stage dominates. Run resolution and batch sweeps, then inspect with
  Nsight Systems.
```

If the smallest model is capped near 10 FPS, focus on these comparisons:

```text
Thor current path vs Thor bottleneck profiler
Thor read-each-time vs Thor preload
Thor 320/640/1024 image size sweep
Thor PyTorch YOLO vs TensorRT YOLO if available
Thor vs RTX 5090 with the same commit, manifest, models, warmup, repeat
```

## Common Setup

Use the same git commit on both machines:

```bash
git clone git@github.com:thedannyliu/EfficientSAM3-Benchmark.git
cd EfficientSAM3-Benchmark
git fetch origin
git checkout main
git pull
git rev-parse HEAD
```

Prepare the fixed COCO manifest and assets as in
`docs/thor_offline_benchmark.md`:

```bash
bash scripts/setup_model_repos.sh
bash scripts/download_sam2_family_checkpoints.sh
bash scripts/download_yoloe_edgetam_mobilesam_assets.sh
bash scripts/prepare_coco_fixed_subset.sh 10
```

Expected manifest:

```text
data/manifests/coco_val2017_fixed10.jsonl
```

Recommended first-pass models:

```text
null_pipeline        harness floor
mobilesam_vit_t      smallest SAM-family path
sam2p1_hiera_tiny    small SAM2 path
sam2p1_hiera_large   larger SAM2 path
yolo11n_seg          smallest YOLO segmentation path
yolo11s_seg          next YOLO segmentation path
```

The matrix wrapper skips missing checkpoints/configs.

## Jetson Thor From Scratch

Follow the Thor environment policy from `docs/thor_offline_benchmark.md`.
Use the NVIDIA JetPack-compatible PyTorch/torchvision wheels. Do not install
the generic PACE CUDA PyTorch pins on Thor.

```bash
cd EfficientSAM3-Benchmark

python3 -m venv --system-site-packages ~/venvs/effisam3_venv_ros
export THOR_VENV=~/venvs/effisam3_venv_ros
export SAM3_SOURCE=~/efficientsam3/sam3
export THOR_ROS_SETUP=/opt/ros/jazzy/setup.bash
source scripts/source_thor_ros_env.sh

python -m pip install -U pip
python -m pip install -r requirements-thor.txt
python -m pip install opencv-python-headless
python -m pip install -e . --no-deps

python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

Before running, record platform state:

```bash
git rev-parse HEAD
cat /etc/nv_tegra_release || true
uname -a
python -m sam_backend.env_probe
nvidia-smi || true
tegrastats --interval 1000
```

Run the matrix:

```bash
cd EfficientSAM3-Benchmark
source scripts/source_thor_ros_env.sh

RUN_ID="thor-$(date +%Y%m%d-%H%M%S)"
LIMIT=10 \
WARMUP=5 \
REPEAT=5 \
INPUT_MODE=preload \
WITH_GT=0 \
TORCH_PROFILER=0 \
IMGSZ_LIST="320 640 1024" \
OUTPUT_ROOT="results/bottleneck/${RUN_ID}" \
bash scripts/run_pipeline_bottleneck_matrix.sh
```

Then run a deployment-like disk-read variant:

```bash
RUN_ID="thor-read-$(date +%Y%m%d-%H%M%S)"
LIMIT=10 \
WARMUP=5 \
REPEAT=5 \
INPUT_MODE=read-each-time \
WITH_GT=0 \
TORCH_PROFILER=0 \
IMGSZ_LIST="640" \
OUTPUT_ROOT="results/bottleneck/${RUN_ID}" \
bash scripts/run_pipeline_bottleneck_matrix.sh
```

Optional GT/postprocess-heavy variant:

```bash
RUN_ID="thor-gt-$(date +%Y%m%d-%H%M%S)"
LIMIT=10 \
WARMUP=5 \
REPEAT=3 \
INPUT_MODE=preload \
WITH_GT=1 \
TORCH_PROFILER=0 \
IMGSZ_LIST="640" \
OUTPUT_ROOT="results/bottleneck/${RUN_ID}" \
bash scripts/run_pipeline_bottleneck_matrix.sh
```

Primary output to compare:

```text
results/bottleneck/<run_id>/bottleneck_matrix_summary.csv
```

If `bottleneck_hint` is `cpu_wrapper_sync_or_copy_bound`, capture an Nsight
Systems trace for one small model:

```bash
nsys profile \
  -o "results/bottleneck/${RUN_ID}/nsys_yolo11n" \
  --trace=cuda,nvtx,osrt \
  python -m sam_backend.pipeline_bottleneck_profile \
    --manifest data/manifests/coco_val2017_fixed10.jsonl \
    --suite yolo \
    --model-id yolo11n_seg_imgsz640_nsys \
    --family yolo-seg \
    --weights yolo11n-seg.pt \
    --device cuda \
    --imgsz 640 \
    --limit 3 \
    --warmup 2 \
    --repeat 1 \
    --input-mode preload \
    --with-torch-profiler \
    --csv-output "results/bottleneck/${RUN_ID}/nsys_yolo11n/profile.csv" \
    --summary-output "results/bottleneck/${RUN_ID}/nsys_yolo11n/summary.json"
```

## RTX 5090 Workstation From Scratch

Use a project-local venv and the CUDA PyTorch wheels from `requirements.txt`.
This is the comparison machine for separating repo/framework fixed cost from
Thor-specific platform limits.

```bash
cd EfficientSAM3-Benchmark

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .

python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

Prepare the same assets:

```bash
bash scripts/setup_model_repos.sh
bash scripts/download_sam2_family_checkpoints.sh
bash scripts/download_yoloe_edgetam_mobilesam_assets.sh
bash scripts/prepare_coco_fixed_subset.sh 10
```

Record platform state:

```bash
git rev-parse HEAD
uname -a
nvidia-smi
python -m sam_backend.env_probe
```

Run the same matrix:

```bash
RUN_ID="rtx5090-$(date +%Y%m%d-%H%M%S)"
LIMIT=10 \
WARMUP=5 \
REPEAT=5 \
INPUT_MODE=preload \
WITH_GT=0 \
TORCH_PROFILER=0 \
IMGSZ_LIST="320 640 1024" \
OUTPUT_ROOT="results/bottleneck/${RUN_ID}" \
bash scripts/run_pipeline_bottleneck_matrix.sh
```

Run the same disk-read variant:

```bash
RUN_ID="rtx5090-read-$(date +%Y%m%d-%H%M%S)"
LIMIT=10 \
WARMUP=5 \
REPEAT=5 \
INPUT_MODE=read-each-time \
WITH_GT=0 \
TORCH_PROFILER=0 \
IMGSZ_LIST="640" \
OUTPUT_ROOT="results/bottleneck/${RUN_ID}" \
bash scripts/run_pipeline_bottleneck_matrix.sh
```

Optional Nsight Systems trace:

```bash
RUN_ID="rtx5090-nsys-$(date +%Y%m%d-%H%M%S)"
mkdir -p "results/bottleneck/${RUN_ID}/nsys_yolo11n"
nsys profile \
  -o "results/bottleneck/${RUN_ID}/nsys_yolo11n/trace" \
  --trace=cuda,nvtx,osrt \
  python -m sam_backend.pipeline_bottleneck_profile \
    --manifest data/manifests/coco_val2017_fixed10.jsonl \
    --suite yolo \
    --model-id yolo11n_seg_imgsz640_nsys \
    --family yolo-seg \
    --weights yolo11n-seg.pt \
    --device cuda \
    --imgsz 640 \
    --limit 3 \
    --warmup 2 \
    --repeat 1 \
    --input-mode preload \
    --with-torch-profiler \
    --csv-output "results/bottleneck/${RUN_ID}/nsys_yolo11n/profile.csv" \
    --summary-output "results/bottleneck/${RUN_ID}/nsys_yolo11n/summary.json"
```

## How To Compare Thor And RTX 5090

Copy or keep both summary CSVs:

```text
results/bottleneck/thor-*/bottleneck_matrix_summary.csv
results/bottleneck/rtx5090-*/bottleneck_matrix_summary.csv
```

Compare these columns first:

```text
model_id
effective_pipeline_fps
mean_total_pipeline_ms
mean_predict_wall_ms
mean_predict_cuda_window_ms
mean_predict_torch_cuda_kernel_ms
gpu_time_source
gpu_time_fraction_of_pipeline
mean_predict_cpu_gap_ms
mean_postprocess_ms
bottleneck_hint
```

Interpretation:

```text
Thor small model near 10 FPS, RTX 5090 also near 10 FPS
  Fixed repo/framework/Python/postprocess cost is likely. Inspect common code.

Thor small model near 10 FPS, RTX 5090 much faster
  Thor platform path is likely. Inspect clocks, shared memory bandwidth,
  Jetson PyTorch kernels, CPU-side wrapper cost, and hardware decode/preprocess.

best GPU timing close to mean_total_pipeline_ms
  GPU model work dominates. Use Nsight Compute for compute vs DRAM bandwidth.

best GPU timing much lower than predict_wall_ms
  CPU wrapper, synchronization, copy, or framework overhead dominates.

preload much faster than read-each-time
  Decode or disk read matters for deployment. Use hardware decode and pipeline
  overlap before judging model FPS.

320/640/1024 scaling is strong
  Pixel-dependent model/preprocess/memory traffic matters.

320/640/1024 scaling is weak
  Fixed per-frame overhead or launch overhead matters.
```

## Notes For PR Or Lab Records

Record:

```text
git commit
machine: Jetson Thor or RTX 5090
OS / JetPack or driver version
Python / torch / torchvision versions
GPU power mode and clocks where available
manifest path
RUN_ID and result directory
models that were skipped due to missing checkpoints
bottleneck_matrix_summary.csv path
any Nsight trace path
```

Generated `results/`, traces, checkpoints, and local datasets are ignored and
should not be committed.
