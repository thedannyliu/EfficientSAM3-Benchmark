#!/usr/bin/env bash
set -euo pipefail

LIMIT_GB="${1:-300}"
shift || true

PATHS=("$@")
if [[ "${#PATHS[@]}" -eq 0 ]]; then
  PATHS=(data checkpoints external results overlays logs)
fi

total_kb=0
for path in "${PATHS[@]}"; do
  if [[ -e "${path}" ]]; then
    size_kb="$(du -sk "${path}" | awk '{print $1}')"
    total_kb=$((total_kb + size_kb))
    awk -v kb="${size_kb}" -v path="${path}" 'BEGIN { printf "%8.2f GiB  %s\n", kb / 1024 / 1024, path }'
  fi
done

limit_kb=$((LIMIT_GB * 1024 * 1024))
awk -v kb="${total_kb}" 'BEGIN { printf "%8.2f GiB  total\n", kb / 1024 / 1024 }'

if (( total_kb > limit_kb )); then
  echo "storage budget exceeded: ${LIMIT_GB} GiB" >&2
  exit 1
fi
