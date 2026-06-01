#!/usr/bin/env bash
set -euo pipefail

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

mkdir -p external

clone_if_missing() {
  local url="$1"
  local path="$2"
  if [[ ! -d "${path}/.git" ]]; then
    git clone "${url}" "${path}"
  fi
}

clone_if_missing https://github.com/facebookresearch/sam3.git external/sam3
clone_if_missing https://github.com/SimonZeng7108/efficientsam3.git external/efficientsam3
clone_if_missing https://github.com/facebookresearch/sam2 external/sam2
clone_if_missing https://github.com/jingjing0419/Efficient-SAM2 external/Efficient-SAM2
clone_if_missing https://github.com/yformer/EfficientTAM external/EfficientTAM
clone_if_missing https://github.com/facebookresearch/EdgeTAM.git external/EdgeTAM
clone_if_missing https://github.com/ChaoningZhang/MobileSAM.git external/MobileSAM

python -m pip install -e external/sam3 --no-deps
python -m pip install -e external/efficientsam3 --no-deps
python -m pip install -e external/sam2 --no-deps
python -m pip install -e external/Efficient-SAM2 --no-deps
python -m pip install -e external/EfficientTAM --no-deps
python -m pip install -e external/EdgeTAM --no-deps
python -m pip install -e external/MobileSAM --no-deps
