#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export SAM_BENCH_SCRATCH="${SAM_BENCH_SCRATCH:-/storage/scratch1/9/eliu354/efficientsam3-benchmark}"
export THOR_VENV="${THOR_VENV:-${HOME}/venvs/effisam3_venv_ros}"
export THOR_ROS_SETUP="${THOR_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
export SAM3_SOURCE="${SAM3_SOURCE:-${HOME}/efficientsam3/sam3}"

INSTALL_DEPS="${INSTALL_DEPS:-1}"
DOWNLOAD_ASSETS="${DOWNLOAD_ASSETS:-1}"
DOWNLOAD_SACO_ANNOTATION="${DOWNLOAD_SACO_ANNOTATION:-1}"
PREPARE_MANIFEST="${PREPARE_MANIFEST:-1}"
RUN_NULL_SMOKE="${RUN_NULL_SMOKE:-1}"
RUN_SUITE="${RUN_SUITE:-0}"
DRY_RUN="${DRY_RUN:-1}"

SACO_SPLIT="${SACO_SPLIT:-val}"
SACO_COUNT="${SACO_COUNT:-20}"
SACO_SEED="${SACO_SEED:-20260617}"
SACO_ANNOTATION="${SACO_ANNOTATION:-${SAM_BENCH_SCRATCH}/data/annotation/saco_veval_sav_${SACO_SPLIT}.json}"
SACO_SAV_MEDIA_ROOT="${SACO_SAV_MEDIA_ROOT:-${SAM_BENCH_SCRATCH}/data/media/saco_sav/JPEGImages_24fps}"
SACO_MANIFEST="${SACO_MANIFEST:-data/manifests/saco_veval_sav_fixed${SACO_COUNT}.jsonl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/thor/saco_stream/${RUN_ID}}"
OVERLAY_DIR="${OVERLAY_DIR:-overlays/thor/saco_stream/${RUN_ID}}"
MAX_FRAMES="${MAX_FRAMES:-120}"
INPUT_FPS="${INPUT_FPS:-30.0}"

mkdir -p "${SAM_BENCH_SCRATCH}/data/annotation" "${SAM_BENCH_SCRATCH}/data/media/saco_sav" \
  data/manifests results/thor/saco_stream overlays/thor/saco_stream

if [[ ! -f "${THOR_ROS_SETUP}" ]]; then
  echo "ERROR: THOR_ROS_SETUP does not exist: ${THOR_ROS_SETUP}" >&2
  exit 2
fi
if [[ ! -f "${THOR_VENV}/bin/activate" ]]; then
  echo "ERROR: THOR_VENV is not ready: ${THOR_VENV}" >&2
  echo "Create it following docs/thor_offline_benchmark.md, then rerun this script." >&2
  exit 2
fi

source scripts/source_thor_ros_env.sh

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  python -m pip install -U pip
  python -m pip install "numpy>=1.26,<2" opencv-python-headless pillow pyyaml huggingface_hub pycocotools
  python -m pip install timm tqdm ftfy==6.1.1 regex iopath typing_extensions psutil
  python -m pip install -e . --no-deps
fi

if [[ "${DOWNLOAD_ASSETS}" == "1" ]]; then
  bash scripts/download_saco_stream_assets.sh
fi

if [[ "${DOWNLOAD_SACO_ANNOTATION}" == "1" && ! -f "${SACO_ANNOTATION}" ]]; then
  hf download facebook/SACo-VEval "annotation/saco_veval_sav_${SACO_SPLIT}.json" \
    --repo-type dataset \
    --local-dir "${SAM_BENCH_SCRATCH}/data"
fi

if [[ ! -d "${SACO_SAV_MEDIA_ROOT}" ]]; then
  if [[ -n "${SAV_JPEG_ROOT:-}" && -d "${SAV_JPEG_ROOT}" ]]; then
    mkdir -p "$(dirname "${SACO_SAV_MEDIA_ROOT}")"
    ln -sfn "${SAV_JPEG_ROOT}" "${SACO_SAV_MEDIA_ROOT}"
  elif [[ -d "data/sa-v/sav_${SACO_SPLIT}/JPEGImages_24fps" ]]; then
    mkdir -p "$(dirname "${SACO_SAV_MEDIA_ROOT}")"
    ln -sfn "${repo_root}/data/sa-v/sav_${SACO_SPLIT}/JPEGImages_24fps" "${SACO_SAV_MEDIA_ROOT}"
  else
    cat >&2 <<EOF
ERROR: SA-Co/VEval-SAV media root is missing:
  ${SACO_SAV_MEDIA_ROOT}

Provide the full merged SA-V JPEGImages_24fps root with:
  SAV_JPEG_ROOT=/path/to/JPEGImages_24fps bash scripts/setup_thor_saco_stream_benchmark.sh

The fixed10 SA-V subset is not enough for a 20-video SA-Co/VEval manifest unless
the selected SA-Co videos happen to overlap.
EOF
    exit 2
  fi
fi

if [[ "${PREPARE_MANIFEST}" == "1" ]]; then
  sam-prepare-saco-veval-sav-subset \
    --annotation "${SACO_ANNOTATION}" \
    --media-root "${SACO_SAV_MEDIA_ROOT}" \
    --count "${SACO_COUNT}" \
    --seed "${SACO_SEED}" \
    --require-media-exists \
    --output "${SACO_MANIFEST}"

  row_count="$(wc -l < "${SACO_MANIFEST}" | tr -d ' ')"
  if [[ "${row_count}" -lt "${SACO_COUNT}" ]]; then
    echo "ERROR: only ${row_count}/${SACO_COUNT} SA-Co/VEval rows had local media." >&2
    echo "Check SACO_SAV_MEDIA_ROOT or provide a full SA-V media root." >&2
    exit 2
  fi
fi

if [[ "${RUN_NULL_SMOKE}" == "1" ]]; then
  python -m sam_backend.profile_saco_stream \
    --manifest "${SACO_MANIFEST}" \
    --limit 1 \
    --max-frames 2 \
    --model-id null_saco_stream_smoke \
    --backend null \
    --stream-mode text_bbox_chain \
    --prompt-type text \
    --device cpu \
    --csv-output "results/thor/saco_stream/smoke/null/frames.csv" \
    --summary-output "results/thor/saco_stream/smoke/null/summary.json" \
    --pred-json "results/thor/saco_stream/smoke/null/saco_veval_preds.json" \
    --overlay-root "overlays/thor/saco_stream/smoke/null"
fi

if [[ "${RUN_SUITE}" == "1" ]]; then
  suite_args=(
    --manifest "${SACO_MANIFEST}"
    --gt-annotation-file "${SACO_ANNOTATION}"
    --scratch-root "${SAM_BENCH_SCRATCH}"
    --max-frames "${MAX_FRAMES}"
    --input-fps "${INPUT_FPS}"
    --output-dir "${OUTPUT_DIR}"
    --overlay-dir "${OVERLAY_DIR}"
    --skip-missing
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    suite_args+=(--dry-run)
  fi
  if [[ -n "${SACO_MODELS:-}" ]]; then
    # shellcheck disable=SC2206
    model_args=(${SACO_MODELS})
    suite_args+=(--models "${model_args[@]}")
  fi
  sam-run-saco-stream-suite "${suite_args[@]}"
fi

cat <<EOF
Thor SA-Co stream benchmark setup complete.

Manifest:
  ${SACO_MANIFEST}

Scratch:
  ${SAM_BENCH_SCRATCH}

Run full suite:
  RUN_SUITE=1 DRY_RUN=0 bash scripts/setup_thor_saco_stream_benchmark.sh

Dry-run model commands:
  RUN_SUITE=1 DRY_RUN=1 bash scripts/setup_thor_saco_stream_benchmark.sh

Results:
  ${OUTPUT_DIR}
  ${OVERLAY_DIR}
EOF
