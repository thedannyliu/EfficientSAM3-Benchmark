#!/usr/bin/env bash
set -euo pipefail

YOLOE_WEIGHTS="${YOLOE_WEIGHTS:-checkpoints/yoloe/yoloe-26m-seg.pt}"
EDGETAM_REPO="${EDGETAM_REPO:-external/EdgeTAM}"
MOBILESAM_REPO="${MOBILESAM_REPO:-external/MobileSAM}"
EDGETAM_CHECKPOINT="${EDGETAM_CHECKPOINT:-checkpoints/edgetam/edgetam.pt}"
MOBILESAM_CHECKPOINT="${MOBILESAM_CHECKPOINT:-checkpoints/mobilesam/mobile_sam.pt}"
STORAGE_LIMIT_GIB="${STORAGE_LIMIT_GIB:-300}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  fallback_venv="${VENV_DIR:-}"
  if [[ -n "${fallback_venv}" && -f "${fallback_venv}/bin/activate" ]]; then
    source "${fallback_venv}/bin/activate"
  elif [[ -f ".venv/bin/activate" ]]; then
    source ".venv/bin/activate"
  elif [[ -f "${HOME}/venvs/effisam3_venv_ros/bin/activate" ]]; then
    source "${HOME}/venvs/effisam3_venv_ros/bin/activate"
  else
    echo "ERROR: activate the benchmark venv before running this script." >&2
    echo "Expected: source ~/venvs/effisam3_venv_ros/bin/activate" >&2
    exit 2
  fi
fi

mkdir -p external checkpoints/edgetam checkpoints/mobilesam checkpoints/yoloe

download_if_missing() {
  local url="$1"
  local output="$2"
  if [[ -f "${output}" ]]; then
    echo "${output}"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -O "${output}" "${url}"
  else
    curl -L -o "${output}" "${url}"
  fi
}

if [[ ! -d "${EDGETAM_REPO}/.git" ]]; then
  git clone https://github.com/facebookresearch/EdgeTAM.git "${EDGETAM_REPO}"
fi

if [[ ! -d "${MOBILESAM_REPO}/.git" ]]; then
  git clone https://github.com/ChaoningZhang/MobileSAM.git "${MOBILESAM_REPO}"
fi

python -m pip install -U "numpy>=1.26,<2" "ultralytics>=8.4.56"
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

python -m pip install --force-reinstall --no-deps "numpy>=1.26,<2"

if [[ ! -f "${EDGETAM_CHECKPOINT}" ]]; then
  download_if_missing \
    https://huggingface.co/spaces/facebook/EdgeTAM/resolve/main/checkpoints/edgetam.pt \
    "${EDGETAM_CHECKPOINT}"
fi

if [[ ! -f "${MOBILESAM_CHECKPOINT}" ]]; then
  download_if_missing \
    https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt \
    "${MOBILESAM_CHECKPOINT}"
fi

SAM_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything"
download_if_missing "${SAM_BASE_URL}/sam_vit_b_01ec64.pth" "checkpoints/mobilesam/sam_vit_b_01ec64.pth"
download_if_missing "${SAM_BASE_URL}/sam_vit_l_0b3195.pth" "checkpoints/mobilesam/sam_vit_l_0b3195.pth"
download_if_missing "${SAM_BASE_URL}/sam_vit_h_4b8939.pth" "checkpoints/mobilesam/sam_vit_h_4b8939.pth"

bash scripts/check_storage_budget.sh "${STORAGE_LIMIT_GIB}" data checkpoints external

cat <<EOF
YOLOE weights:       ${YOLOE_WEIGHTS}
EdgeTAM repo:        ${EDGETAM_REPO}
EdgeTAM checkpoint:  ${EDGETAM_CHECKPOINT}
MobileSAM repo:      ${MOBILESAM_REPO}
MobileSAM checkpoint:${MOBILESAM_CHECKPOINT}
MobileSAM SAM ViT-B: checkpoints/mobilesam/sam_vit_b_01ec64.pth
MobileSAM SAM ViT-L: checkpoints/mobilesam/sam_vit_l_0b3195.pth
MobileSAM SAM ViT-H: checkpoints/mobilesam/sam_vit_h_4b8939.pth
Storage budget:      ${STORAGE_LIMIT_GIB} GiB
EOF
