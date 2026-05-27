#!/usr/bin/env bash
set -euo pipefail

COUNT="${1:-10}"
OUT_DIR="${2:-data/coco}"
SEED="${COCO_SEED:-20260527}"
MIN_AREA="${COCO_MIN_AREA:-1024}"
MANIFEST="data/manifests/coco_val2017_fixed${COUNT}.jsonl"
RECORD="${OUT_DIR}/coco_val2017_fixed${COUNT}_selection.json"
TRACKED_RECORD="data/manifests/coco_val2017_fixed${COUNT}_selection.json"
PYTHON_BIN="${PYTHON:-python}"

mkdir -p data/manifests

bash scripts/download_coco_val2017.sh "${OUT_DIR}"

"${PYTHON_BIN}" -m sam_backend.coco_manifest \
  --annotations "${OUT_DIR}/annotations/instances_val2017.json" \
  --image-dir "${OUT_DIR}/images/val2017" \
  --output "${MANIFEST}" \
  --count "${COUNT}" \
  --seed "${SEED}" \
  --min-area "${MIN_AREA}"

"${PYTHON_BIN}" - "${MANIFEST}" "${RECORD}" "${TRACKED_RECORD}" "${COUNT}" "${SEED}" "${MIN_AREA}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
record = Path(sys.argv[2])
tracked_record = Path(sys.argv[3])
count = int(sys.argv[4])
seed = int(sys.argv[5])
min_area = float(sys.argv[6])
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
payload = {
    "source": "COCO val2017 instances",
    "count": count,
    "seed": seed,
    "min_area": min_area,
    "manifest": str(manifest),
    "selection": "random_image_seeded_largest_non_crowd_object",
    "prompt_protocol": {
        "text": "COCO category name of selected annotation",
        "point": "centroid of selected annotation mask",
    },
    "samples": [
        {
            "sample_id": row["sample_id"],
            "image_id": row["image_id"],
            "annotation_id": row["annotation_id"],
            "file_name": row["file_name"],
            "category_name": row["category_name"],
            "text_prompt": row["text_prompt"],
            "point": row["point"],
            "point_label": row["point_label"],
            "area": row["area"],
            "width": row["width"],
            "height": row["height"],
        }
        for row in rows
    ],
}
for path in (record, tracked_record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(record)
print(tracked_record)
PY

echo "${MANIFEST}"
