#!/usr/bin/env bash
set -euo pipefail

COCO_COUNT="${COCO_COUNT:-10}"
COCO_DIR="${COCO_DIR:-data/coco}"
MANIFEST="${MANIFEST:-data/manifests/coco_val2017_fixed${COCO_COUNT}.jsonl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
YOLO_PRESET="${YOLO_PRESET:-quick}"
LIMIT="${LIMIT:-1}"
DEVICE="${DEVICE:-cuda}"
IMGSZ="${IMGSZ:-640}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.7}"
MAX_DET="${MAX_DET:-100}"
EVAL_MODE="${EVAL_MODE:-both}"
OUTPUT_DIR="${OUTPUT_DIR:-results/thor/offline/yolo_coco/${RUN_ID}}"
OVERLAY_DIR="${OVERLAY_DIR:-overlays/thor/offline/yolo_coco/${RUN_ID}}"
DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-0}"
PREPARE_COCO="${PREPARE_COCO:-0}"

if [[ "${PREPARE_COCO}" == "1" ]]; then
  bash scripts/prepare_coco_fixed_subset.sh "${COCO_COUNT}" "${COCO_DIR}"
fi

if [[ "${DOWNLOAD_WEIGHTS}" == "1" ]]; then
  python - <<PY
from pathlib import Path
import shutil

from sam_backend.yolo_coco_suite import weight_names_for_preset

preset = "${YOLO_PRESET}"
weights = weight_names_for_preset(preset)
if not weights:
    raise SystemExit(f"no YOLO weights for preset {preset}")

from ultralytics import YOLO, YOLOE

for weight in weights:
    path = Path(weight)
    if path.exists():
        print(f"YOLO weight ready: {path}")
        continue
    cls = YOLOE if "yoloe" in path.name else YOLO
    print(f"Downloading/preparing YOLO weight: {weight}")
    model = cls(path.name if path.parent != Path(".") else weight)
    if path.parent != Path(".") and not path.exists():
        candidates = [Path(path.name)]
        for obj in (model, getattr(model, "model", None)):
            for attr in ("ckpt_path", "pt_path"):
                value = getattr(obj, attr, None) if obj is not None else None
                if value:
                    candidates.append(Path(str(value)))
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, path)
                print(f"YOLO weight copied to: {path}")
                break
PY
fi

python -m sam_backend.yolo_coco_suite \
  --manifest "${MANIFEST}" \
  --device "${DEVICE}" \
  --preset "${YOLO_PRESET}" \
  --limit "${LIMIT}" \
  --imgsz "${IMGSZ}" \
  --conf "${CONF}" \
  --iou "${IOU}" \
  --max-det "${MAX_DET}" \
  --eval-mode "${EVAL_MODE}" \
  --output-dir "${OUTPUT_DIR}" \
  --overlay-dir "${OVERLAY_DIR}" \
  "$@"
