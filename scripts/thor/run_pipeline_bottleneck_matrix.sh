#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${MANIFEST:-data/manifests/coco_val2017_fixed10.jsonl}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-10}"
WARMUP="${WARMUP:-5}"
REPEAT="${REPEAT:-5}"
INPUT_MODE="${INPUT_MODE:-preload}"
WITH_GT="${WITH_GT:-0}"
TORCH_PROFILER="${TORCH_PROFILER:-0}"
IMGSZ_LIST="${IMGSZ_LIST:-320 640 1024}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/bottleneck/${RUN_ID}}"

gt_args=()
if [[ "${WITH_GT}" == "1" ]]; then
  gt_args=(--with-gt)
fi
profiler_args=()
if [[ "${TORCH_PROFILER}" == "1" ]]; then
  profiler_args=(--with-torch-profiler)
fi

common_args=(
  --manifest "${MANIFEST}"
  --device "${DEVICE}"
  --limit "${LIMIT}"
  --warmup "${WARMUP}"
  --repeat "${REPEAT}"
  --input-mode "${INPUT_MODE}"
  "${gt_args[@]}"
  "${profiler_args[@]}"
)

run_sam() {
  local model_id="$1"
  local backend="$2"
  local prompt_mode="$3"
  local checkpoint="$4"
  local model_config="$5"
  local external_repo="$6"
  shift 6

  if [[ "${checkpoint}" != "-" && ! -f "${checkpoint}" ]]; then
    printf 'skip %s: missing checkpoint %s\n' "${model_id}" "${checkpoint}"
    return 0
  fi
  if [[ "${model_config}" != "-" && ! -f "${model_config}" ]]; then
    printf 'skip %s: missing config %s\n' "${model_id}" "${model_config}"
    return 0
  fi

  local out_dir="${OUTPUT_ROOT}/${model_id}"
  mkdir -p "${out_dir}"
  local cmd=(
    python -m sam_backend.pipeline_bottleneck_profile
    "${common_args[@]}"
    --suite sam
    --model-id "${model_id}"
    --backend "${backend}"
    --prompt-mode "${prompt_mode}"
    --csv-output "${out_dir}/profile.csv"
    --summary-output "${out_dir}/summary.json"
  )
  if [[ "${checkpoint}" != "-" ]]; then
    cmd+=(--checkpoint-path "${checkpoint}")
  fi
  if [[ "${model_config}" != "-" ]]; then
    cmd+=(--model-config "${model_config}")
  fi
  if [[ "${external_repo}" != "-" ]]; then
    cmd+=(--external-repo "${external_repo}")
  fi
  cmd+=("$@")
  "${cmd[@]}"
}

run_yolo() {
  local model_id="$1"
  local family="$2"
  local weights="$3"
  local imgsz="$4"
  local out_dir="${OUTPUT_ROOT}/${model_id}_imgsz${imgsz}"
  mkdir -p "${out_dir}"
  python -m sam_backend.pipeline_bottleneck_profile \
    "${common_args[@]}" \
    --suite yolo \
    --model-id "${model_id}_imgsz${imgsz}" \
    --family "${family}" \
    --weights "${weights}" \
    --imgsz "${imgsz}" \
    --csv-output "${out_dir}/profile.csv" \
    --summary-output "${out_dir}/summary.json"
}

mkdir -p "${OUTPUT_ROOT}"

run_sam null_pipeline null point - - -
run_sam mobilesam_vit_t mobilesam point checkpoints/mobilesam/mobile_sam.pt - external/MobileSAM --mobile-sam-model-type vit_t
run_sam sam2p1_hiera_tiny sam2 point checkpoints/sam2/sam2.1_hiera_tiny.pt configs/sam2.1/sam2.1_hiera_t.yaml external/sam2
run_sam sam2p1_hiera_large sam2 point checkpoints/sam2/sam2.1_hiera_large.pt configs/sam2.1/sam2.1_hiera_l.yaml external/sam2

for imgsz in ${IMGSZ_LIST}; do
  run_yolo yolo11n_seg yolo-seg yolo11n-seg.pt "${imgsz}"
  run_yolo yolo11s_seg yolo-seg yolo11s-seg.pt "${imgsz}"
done

python - <<'PY' "${OUTPUT_ROOT}"
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/summary.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "model_id": data.get("model_id", ""),
        "samples": data.get("samples", ""),
        "rows": data.get("rows", ""),
        "mean_total_pipeline_ms": data.get("mean_total_pipeline_ms", ""),
        "effective_pipeline_fps": data.get("effective_pipeline_fps", ""),
        "mean_predict_wall_ms": data.get("mean_predict_wall_ms", ""),
        "mean_predict_cuda_window_ms": data.get("mean_predict_cuda_window_ms", ""),
        "mean_predict_torch_cuda_kernel_ms": data.get("mean_predict_torch_cuda_kernel_ms", ""),
        "gpu_time_source": data.get("gpu_time_source", ""),
        "gpu_time_fraction_of_pipeline": data.get("gpu_time_fraction_of_pipeline", ""),
        "mean_predict_cpu_gap_ms": data.get("mean_predict_cpu_gap_ms", ""),
        "mean_postprocess_ms": data.get("mean_postprocess_ms", ""),
        "bottleneck_hint": data.get("bottleneck_hint", ""),
        "summary": str(path),
    })
out = root / "bottleneck_matrix_summary.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["model_id"])
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY
