#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-checkpoints/sam3}"
PYTHON_BIN="${PYTHON:-python}"

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)
repo_id = "facebook/sam3"
for filename in ("config.json", "sam3.pt"):
    dst = out_dir / filename
    if dst.exists():
        print(f"exists: {dst}")
        continue
    src = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    shutil.copy2(src, dst)
    print(dst)
PY
