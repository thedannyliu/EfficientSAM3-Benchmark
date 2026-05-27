#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-data/coco}"
mkdir -p "${OUT_DIR}/images" "${OUT_DIR}/annotations"

download() {
  local url="$1"
  local output="$2"
  if [[ -f "${output}" ]]; then
    echo "exists: ${output}"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -c --progress=dot:giga "${url}" -O "${output}"
  else
    curl -L --fail --retry 3 -C - "${url}" -o "${output}"
  fi
}

download "http://images.cocodataset.org/zips/val2017.zip" "${OUT_DIR}/images/val2017.zip"
download "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "${OUT_DIR}/annotations_trainval2017.zip"

if [[ ! -d "${OUT_DIR}/images/val2017" ]]; then
  unzip -q "${OUT_DIR}/images/val2017.zip" -d "${OUT_DIR}/images"
fi

if [[ ! -f "${OUT_DIR}/annotations/instances_val2017.json" ]]; then
  unzip -q "${OUT_DIR}/annotations_trainval2017.zip" -d "${OUT_DIR}"
fi

echo "${OUT_DIR}/images/val2017"
echo "${OUT_DIR}/annotations/instances_val2017.json"
