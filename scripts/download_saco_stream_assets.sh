#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SAM_BENCH_SCRATCH="${SAM_BENCH_SCRATCH:-${repo_root}}"
SCRATCH_ROOT="${SAM_BENCH_SCRATCH}"
CHECKPOINT_DIR="${SCRATCH_ROOT}/checkpoints"
EXTERNAL_DIR="${SCRATCH_ROOT}/external"

mkdir -p "${CHECKPOINT_DIR}/sam1" "${CHECKPOINT_DIR}/sam2" "${CHECKPOINT_DIR}/sam3p1" \
  "${CHECKPOINT_DIR}/efficientsam3_ft" "${CHECKPOINT_DIR}/mobilesam" "${EXTERNAL_DIR}"

download_if_missing() {
  local url="$1"
  local output="$2"
  if [[ -f "${output}" ]]; then
    echo "exists: ${output}"
    return
  fi
  mkdir -p "$(dirname "${output}")"
  if command -v wget >/dev/null 2>&1; then
    wget -c --progress=dot:giga "${url}" -O "${output}"
  else
    curl -L --fail --retry 3 -C - "${url}" -o "${output}"
  fi
}

clone_if_missing() {
  local url="$1"
  local path="$2"
  if [[ ! -d "${path}/.git" ]]; then
    git clone "${url}" "${path}"
  else
    echo "exists: ${path}"
  fi
}

clone_if_missing https://github.com/facebookresearch/sam3.git "${EXTERNAL_DIR}/sam3"
clone_if_missing https://github.com/SimonZeng7108/efficientsam3.git "${EXTERNAL_DIR}/efficientsam3"
clone_if_missing https://github.com/facebookresearch/sam2 "${EXTERNAL_DIR}/sam2"
clone_if_missing https://github.com/ChaoningZhang/MobileSAM.git "${EXTERNAL_DIR}/MobileSAM"

SAM1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything"
download_if_missing "${SAM1_BASE_URL}/sam_vit_b_01ec64.pth" "${CHECKPOINT_DIR}/sam1/sam_vit_b_01ec64.pth"
download_if_missing "${SAM1_BASE_URL}/sam_vit_l_0b3195.pth" "${CHECKPOINT_DIR}/sam1/sam_vit_l_0b3195.pth"
download_if_missing "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt" "${CHECKPOINT_DIR}/mobilesam/mobile_sam.pt"

SAM2_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
download_if_missing "${SAM2_BASE_URL}/sam2.1_hiera_tiny.pt" "${CHECKPOINT_DIR}/sam2/sam2.1_hiera_tiny.pt"
download_if_missing "${SAM2_BASE_URL}/sam2.1_hiera_large.pt" "${CHECKPOINT_DIR}/sam2/sam2.1_hiera_large.pt"

download_if_missing \
  "https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_efficientvit.pt?download=true" \
  "${CHECKPOINT_DIR}/efficientsam3_ft/efficientsam3_efficientvit.pt"
download_if_missing \
  "https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_repvit.pt?download=true" \
  "${CHECKPOINT_DIR}/efficientsam3_ft/efficientsam3_repvit.pt"
download_if_missing \
  "https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_tinyvit.pt?download=true" \
  "${CHECKPOINT_DIR}/efficientsam3_ft/efficientsam3_tinyvit.pt"

python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
import os
import shutil

root = Path(os.environ["SAM_BENCH_SCRATCH"])
dst = root / "checkpoints" / "sam3p1" / "sam3.1_multiplex.pt"
if dst.exists():
    print(f"exists: {dst}")
else:
    src = Path(hf_hub_download(repo_id="facebook/sam3.1", filename="sam3.1_multiplex.pt"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(dst)
PY

cat <<EOF
SA-Co stream assets rooted at: ${SCRATCH_ROOT}
Checkpoints: ${CHECKPOINT_DIR}
External repos: ${EXTERNAL_DIR}
EOF
