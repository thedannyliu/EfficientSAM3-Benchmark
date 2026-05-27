#!/usr/bin/env bash
set -euo pipefail

COCO_COUNT="${COCO_COUNT:-10}"
SAV_SPLIT="${SAV_SPLIT:-val}"
SAV_COUNT="${SAV_COUNT:-3}"
SAV_ROOT="${SAV_ROOT:-data/sa-v/sav_${SAV_SPLIT}_fixed${SAV_COUNT}}"
STORAGE_LIMIT_GIB="${STORAGE_LIMIT_GIB:-300}"

bash scripts/check_storage_budget.sh "${STORAGE_LIMIT_GIB}" data checkpoints external
bash scripts/prepare_coco_fixed_subset.sh "${COCO_COUNT}"
bash scripts/download_sav_valtest_subset.sh "${SAV_SPLIT}" "${SAV_COUNT}" "${SAV_ROOT}"
bash scripts/check_storage_budget.sh "${STORAGE_LIMIT_GIB}" data checkpoints external

cat <<EOF
COCO manifest: data/manifests/coco_val2017_fixed${COCO_COUNT}.jsonl
COCO record:   data/coco/coco_val2017_fixed${COCO_COUNT}_selection.json
COCO tracked:  data/manifests/coco_val2017_fixed${COCO_COUNT}_selection.json
SA-V manifest: data/manifests/sav_${SAV_SPLIT}_fixed${SAV_COUNT}.jsonl
SA-V record:   ${SAV_ROOT}/official_subset_manifest.json
SA-V tracked:  data/manifests/sav_${SAV_SPLIT}_fixed${SAV_COUNT}_selection.json
EOF
