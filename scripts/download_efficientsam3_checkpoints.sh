#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-checkpoints}"
PYTHON_BIN="${PYTHON:-python}"

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


repo_id = "Simon7108528/EfficientSAM3"
out_dir = Path(sys.argv[1])
filenames = [
    "stage1_sam3p1/efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt",
    "stage1_sam3p1/efficient_sam3p1_efficientvit_l_mobileclip_s0_ctx16.pt",
    "stage1_all_converted/efficient_sam3_efficientvit-b0_mobileclip_s1.pth",
    "stage1_all_converted/efficient_sam3_efficientvit-b2_mobileclip_s1.pth",
]

for filename in filenames:
    dst = out_dir / filename
    if dst.exists():
        print(f"exists: {dst}")
        continue
    src = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(dst)
PY
