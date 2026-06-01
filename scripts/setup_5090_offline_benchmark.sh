#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"
STORAGE_LIMIT_GIB="${STORAGE_LIMIT_GIB:-300}"
COCO_COUNT="${COCO_COUNT:-10}"
SAV_SPLIT="${SAV_SPLIT:-val}"
SAV_COUNT="${SAV_COUNT:-10}"

INSTALL_DEPS="${INSTALL_DEPS:-1}"
SETUP_REPOS="${SETUP_REPOS:-1}"
DOWNLOAD_CHECKPOINTS="${DOWNLOAD_CHECKPOINTS:-1}"
PREPARE_DATASETS="${PREPARE_DATASETS:-1}"
PREPARE_SAV_TEXT="${PREPARE_SAV_TEXT:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
  else
    PYTHON_BIN="python3"
  fi
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

echo "Repo: ${repo_root}"
echo "Python: ${PYTHON_BIN}"
echo "Venv: ${VENV_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
export PYTHON=python

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  python -m pip install -U pip
  python -m pip install -r requirements.txt
  python -m pip install -e .
fi

python - <<'PY'
import platform
import torch

print("python:", platform.python_version())
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY

if [[ "${SETUP_REPOS}" == "1" ]]; then
  bash scripts/setup_model_repos.sh
fi

if [[ "${DOWNLOAD_CHECKPOINTS}" == "1" ]]; then
  bash scripts/download_sam3_checkpoint.sh
  bash scripts/download_efficientsam3_checkpoints.sh
  bash scripts/download_sam2_family_checkpoints.sh
  bash scripts/download_yoloe_edgetam_mobilesam_assets.sh
fi

if [[ "${PREPARE_DATASETS}" == "1" ]]; then
  bash scripts/prepare_coco_fixed_subset.sh "${COCO_COUNT}"
  if [[ "${SAV_SPLIT}" == "val" && "${SAV_COUNT}" == "10" ]]; then
    bash scripts/prepare_sav_fixed10_subset.sh
  else
    bash scripts/download_sav_valtest_subset.sh "${SAV_SPLIT}" "${SAV_COUNT}" "data/sa-v/sav_${SAV_SPLIT}_fixed${SAV_COUNT}"
  fi
fi

if [[ "${PREPARE_SAV_TEXT}" == "1" ]]; then
  sav_manifest="data/manifests/sav_${SAV_SPLIT}_fixed${SAV_COUNT}.jsonl"
  sav_text_prompts="configs/datasets/sav_${SAV_SPLIT}_fixed${SAV_COUNT}_text_prompts.json"
  sav_text_manifest="data/manifests/sav_${SAV_SPLIT}_fixed${SAV_COUNT}_text.jsonl"
  if [[ -f "${sav_manifest}" && -f "${sav_text_prompts}" ]]; then
    python -m sam_backend.sav_text_prompts apply \
      --manifest "${sav_manifest}" \
      --prompts "${sav_text_prompts}" \
      --output "${sav_text_manifest}"
  elif [[ -f "${sav_manifest}" ]]; then
    echo "SA-V text prompt file is not present yet: ${sav_text_prompts}"
    echo "Create it with: python -m sam_backend.sav_text_prompts init --manifest ${sav_manifest} --review-dir overlays/sav/review/sav_${SAV_SPLIT}_fixed${SAV_COUNT} --output ${sav_text_prompts}"
  fi
fi

bash scripts/check_storage_budget.sh "${STORAGE_LIMIT_GIB}" data checkpoints external

if [[ "${RUN_SMOKE}" == "1" ]]; then
  python -m sam_backend.env_probe
  sav_text_manifest="data/manifests/sav_${SAV_SPLIT}_fixed${SAV_COUNT}_text.jsonl"
  sav_point_manifest="data/manifests/sav_${SAV_SPLIT}_fixed${SAV_COUNT}.jsonl"
  sav_smoke_manifest="${sav_text_manifest}"
  sav_smoke_prompt_mode="both"
  if [[ ! -f "${sav_smoke_manifest}" ]]; then
    sav_smoke_manifest="${sav_point_manifest}"
    sav_smoke_prompt_mode="point"
  fi
  if [[ -f "${sav_smoke_manifest}" ]]; then
    python -m sam_backend.profile_sav_frames \
      --manifest "${sav_smoke_manifest}" \
      --limit 1 \
      --max-frames 1 \
      --model-id null_sav${SAV_COUNT}_text_smoke \
      --backend null \
      --device cpu \
      --prompt-mode "${sav_smoke_prompt_mode}" \
      --csv-output "results/rtx5090/offline/smoke/sav_frames/null/frames.csv" \
      --summary-output "results/rtx5090/offline/smoke/sav_frames/null/summary.json"
  else
    echo "Skipping SA-V null smoke; missing manifest: ${sav_smoke_manifest}"
  fi
fi

cat <<EOF
RTX 5090 offline setup complete.

Activate:
  source ${VENV_DIR}/bin/activate

Key manifests:
  data/manifests/coco_val2017_fixed${COCO_COUNT}.jsonl
  data/manifests/sav_${SAV_SPLIT}_fixed${SAV_COUNT}.jsonl
  data/manifests/sav_${SAV_SPLIT}_fixed${SAV_COUNT}_text.jsonl

Recommended result roots:
  results/rtx5090/offline/
  overlays/rtx5090/offline/
EOF
