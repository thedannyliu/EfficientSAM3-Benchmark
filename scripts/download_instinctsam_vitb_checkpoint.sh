#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-checkpoints}"
PYTHON_BIN="${PYTHON:-python}"
TRUNK_FILENAME="${INSTINCTSAM_TRUNK_FILENAME:-concept_vitb_trunk_step6000.pt}"
TEACHER_CHECKPOINT="${INSTINCTSAM_TEACHER_CHECKPOINT:-${OUT_DIR}/sam3/sam3.pt}"
TEXT_CHECKPOINT="${INSTINCTSAM_TEXT_CHECKPOINT:-${OUT_DIR}/stage1_all_converted/efficient_sam3_efficientvit-b0_mobileclip_s1.pth}"
TARGET_CHECKPOINT="${INSTINCTSAM_TARGET_CHECKPOINT:-${OUT_DIR}/instinctsam/instinctsam_vitb_concept.pt}"
BUILD_DEVICE="${INSTINCTSAM_BUILD_DEVICE:-cpu}"

if [[ ! -f "${TEACHER_CHECKPOINT}" ]]; then
  echo "ERROR: missing SAM3 teacher checkpoint: ${TEACHER_CHECKPOINT}" >&2
  echo "Run scripts/download_sam3_checkpoint.sh first." >&2
  exit 2
fi
if [[ ! -f "${TEXT_CHECKPOINT}" ]]; then
  echo "ERROR: missing MobileCLIP-S1 source checkpoint: ${TEXT_CHECKPOINT}" >&2
  echo "Run scripts/download_efficientsam3_checkpoints.sh first." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}/instinctsam"

"${PYTHON_BIN}" - "${OUT_DIR}/instinctsam" "${TRUNK_FILENAME}" <<'PY'
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


out_dir = Path(sys.argv[1])
filename = sys.argv[2]
dst = out_dir / filename
if dst.exists():
    print(f"exists: {dst}")
else:
    src = Path(hf_hub_download(repo_id="GM717/InstinctSAM-ViT-B", filename=filename))
    shutil.copy2(src, dst)
    print(dst)
PY

"${PYTHON_BIN}" -m sam_backend.instinctsam \
  --teacher "${TEACHER_CHECKPOINT}" \
  --trunk "${OUT_DIR}/instinctsam/${TRUNK_FILENAME}" \
  --text "${TEXT_CHECKPOINT}" \
  --out "${TARGET_CHECKPOINT}" \
  --external-repo external/efficientsam3 \
  --device "${BUILD_DEVICE}"
