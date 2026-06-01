#!/usr/bin/env bash
set -euo pipefail

SPLIT="${1:-val}"
COUNT="${2:-10}"

SAV_MANIFEST_PREFIX="${SAV_MANIFEST_PREFIX:-sav_${SPLIT}_fixed${COUNT}}" \
  bash scripts/download_sav_valtest_subset.sh "${SPLIT}" "${COUNT}" "data/sa-v/sav_${SPLIT}_fixed${COUNT}"
