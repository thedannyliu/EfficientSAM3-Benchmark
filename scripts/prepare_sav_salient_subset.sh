#!/usr/bin/env bash
set -euo pipefail

SAV_SPLIT="${SAV_SPLIT:-val}"
SAV_COUNT="${SAV_COUNT:-3}"
SAV_ROOT="${SAV_ROOT:-data/sa-v/sav_${SAV_SPLIT}_salient_fixed${SAV_COUNT}}"
SAV_MANIFEST_PREFIX="${SAV_MANIFEST_PREFIX:-sav_${SAV_SPLIT}_salient_fixed${SAV_COUNT}}"
SAV_SELECTION_POLICY="${SAV_SELECTION_POLICY:-salient_first_mask}"
SAV_MIN_AREA_RATIO="${SAV_MIN_AREA_RATIO:-0.01}"
SAV_MAX_ASPECT_RATIO="${SAV_MAX_ASPECT_RATIO:-6}"
SAV_CANDIDATE_COUNT="${SAV_CANDIDATE_COUNT:-30}"
STORAGE_LIMIT_GIB="${STORAGE_LIMIT_GIB:-300}"

export SAV_MANIFEST_PREFIX
export SAV_SELECTION_POLICY
export SAV_MIN_AREA_RATIO
export SAV_MAX_ASPECT_RATIO
export SAV_CANDIDATE_COUNT

bash scripts/check_storage_budget.sh "${STORAGE_LIMIT_GIB}" data checkpoints external
bash scripts/download_sav_valtest_subset.sh "${SAV_SPLIT}" "${SAV_COUNT}" "${SAV_ROOT}"
"${PYTHON:-python}" -m sam_backend.sav_review \
  --manifest "data/manifests/${SAV_MANIFEST_PREFIX}.jsonl" \
  --output-dir "overlays/sav/review/${SAV_MANIFEST_PREFIX}"
bash scripts/check_storage_budget.sh "${STORAGE_LIMIT_GIB}" data checkpoints external

cat <<EOF
SA-V salient root:     ${SAV_ROOT}
SA-V salient manifest: data/manifests/${SAV_MANIFEST_PREFIX}.jsonl
SA-V salient record:   data/manifests/${SAV_MANIFEST_PREFIX}_selection.json
SA-V review sheet:     overlays/sav/review/${SAV_MANIFEST_PREFIX}/contact_sheet.png
EOF
