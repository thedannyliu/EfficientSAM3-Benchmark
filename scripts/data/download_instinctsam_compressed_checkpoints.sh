#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-checkpoints/instinctsam}"
PYTHON_BIN="${PYTHON:-python}"

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


out_dir = Path(sys.argv[1])
for filename in ("gitext_large_v4.pt", "hiera_large_concept_trunk.pt"):
    destination = out_dir / filename
    if destination.exists():
        print(f"exists: {destination}")
        continue
    source = Path(
        hf_hub_download(
            repo_id="GM717/InstinctSAM-ViT-B",
            filename=filename,
        )
    )
    shutil.copy2(source, destination)
    print(destination)
PY
