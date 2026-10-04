#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-checkpoints/sam2_distill/tinyvit}"
PYTHON_BIN="${PYTHON:-python}"

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


out_dir = Path(sys.argv[1])
models = {
    "timm/tiny_vit_21m_512.dist_in22k_ft_in1k": "tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors",
    "timm/tiny_vit_11m_224.dist_in22k_ft_in1k": "tiny_vit_11m_224.dist_in22k_ft_in1k.safetensors",
    "timm/tiny_vit_5m_224.dist_in22k_ft_in1k": "tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors",
}

for repo_id, output_name in models.items():
    dst = out_dir / output_name
    if dst.exists():
        print(f"exists: {dst}")
        continue
    src = Path(hf_hub_download(repo_id=repo_id, filename="model.safetensors"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"downloaded: {dst}")
PY

cat <<EOF
TinyViT init weights ready:
  ${OUT_DIR}/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
  ${OUT_DIR}/tiny_vit_11m_224.dist_in22k_ft_in1k.safetensors
  ${OUT_DIR}/tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors
EOF
