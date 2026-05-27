#!/usr/bin/env bash
set -euo pipefail

YOLOE_WEIGHTS="${YOLOE_WEIGHTS:-checkpoints/yoloe/yoloe-26m-seg.pt}"
EDGETAM_REPO="${EDGETAM_REPO:-external/EdgeTAM}"
MOBILESAM_REPO="${MOBILESAM_REPO:-external/MobileSAM}"
EDGETAM_CHECKPOINT="${EDGETAM_CHECKPOINT:-checkpoints/edgetam/edgetam.pt}"
MOBILESAM_CHECKPOINT="${MOBILESAM_CHECKPOINT:-checkpoints/mobilesam/mobile_sam.pt}"
STORAGE_LIMIT_GIB="${STORAGE_LIMIT_GIB:-300}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f ".venv/bin/activate" ]]; then
    # Keep dependency installs in the project environment required by AGENTS.md.
    source .venv/bin/activate
  else
    echo "ERROR: activate the project .venv before running this script." >&2
    echo "Expected: source .venv/bin/activate" >&2
    exit 2
  fi
fi

mkdir -p external checkpoints/edgetam checkpoints/mobilesam checkpoints/yoloe

if [[ ! -d "${EDGETAM_REPO}/.git" ]]; then
  git clone https://github.com/facebookresearch/EdgeTAM.git "${EDGETAM_REPO}"
fi

if [[ ! -d "${MOBILESAM_REPO}/.git" ]]; then
  git clone https://github.com/ChaoningZhang/MobileSAM.git "${MOBILESAM_REPO}"
fi

python -m pip install -U ultralytics
python -m pip install -e "${EDGETAM_REPO}" --no-deps
python -m pip install -e "${MOBILESAM_REPO}" --no-deps

python - <<PY
from pathlib import Path
import shutil
from ultralytics import YOLOE

weights = "${YOLOE_WEIGHTS}"
weights_path = Path(weights)
if weights_path.exists():
    YOLOE(str(weights_path))
else:
    downloaded = Path("yoloe-26m-seg.pt")
    YOLOE(downloaded.name)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if downloaded.resolve() != weights_path.resolve():
        shutil.move(str(downloaded), str(weights_path))
print(f"YOLOE weights ready via Ultralytics cache/name: {weights}")
PY

if [[ ! -f "${EDGETAM_CHECKPOINT}" ]]; then
  if command -v wget >/dev/null 2>&1; then
    wget -O "${EDGETAM_CHECKPOINT}" \
      https://huggingface.co/spaces/facebook/EdgeTAM/resolve/main/checkpoints/edgetam.pt
  else
    curl -L -o "${EDGETAM_CHECKPOINT}" \
      https://huggingface.co/spaces/facebook/EdgeTAM/resolve/main/checkpoints/edgetam.pt
  fi
fi

if [[ ! -f "${MOBILESAM_CHECKPOINT}" ]]; then
  if command -v wget >/dev/null 2>&1; then
    wget -O "${MOBILESAM_CHECKPOINT}" \
      https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
  else
    curl -L -o "${MOBILESAM_CHECKPOINT}" \
      https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
  fi
fi

bash scripts/check_storage_budget.sh "${STORAGE_LIMIT_GIB}" data checkpoints external

cat <<EOF
YOLOE weights:       ${YOLOE_WEIGHTS}
EdgeTAM repo:        ${EDGETAM_REPO}
EdgeTAM checkpoint:  ${EDGETAM_CHECKPOINT}
MobileSAM repo:      ${MOBILESAM_REPO}
MobileSAM checkpoint:${MOBILESAM_CHECKPOINT}
Storage budget:      ${STORAGE_LIMIT_GIB} GiB
EOF
