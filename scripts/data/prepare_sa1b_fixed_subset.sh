#!/usr/bin/env bash
set -euo pipefail

SA1B_ROOT="${SA1B_ROOT:-data/sa1b/extracted_two_tar}"
SA1B_IMAGE_ROOT="${SA1B_IMAGE_ROOT:-${SA1B_ROOT}}"
SA1B_COUNT="${SA1B_COUNT:-100}"
SA1B_SEED="${SA1B_SEED:-20260707}"
SA1B_MIN_AREA="${SA1B_MIN_AREA:-1024}"
MANIFEST="${MANIFEST:-data/manifests/sa1b_fixed${SA1B_COUNT}.jsonl}"

python -m sam_backend.sa1b_manifest \
  --annotation-root "${SA1B_ROOT}" \
  --image-root "${SA1B_IMAGE_ROOT}" \
  --count "${SA1B_COUNT}" \
  --seed "${SA1B_SEED}" \
  --min-area "${SA1B_MIN_AREA}" \
  --output "${MANIFEST}"
