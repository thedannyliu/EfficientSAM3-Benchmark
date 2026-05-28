#!/usr/bin/env bash
set -euo pipefail

download() {
  local url="$1"
  local output="$2"
  mkdir -p "$(dirname "${output}")"
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

SAM2_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
download "${SAM2_BASE_URL}/sam2.1_hiera_tiny.pt" "checkpoints/sam2/sam2.1_hiera_tiny.pt"
download "${SAM2_BASE_URL}/sam2.1_hiera_small.pt" "checkpoints/sam2/sam2.1_hiera_small.pt"
download "${SAM2_BASE_URL}/sam2.1_hiera_base_plus.pt" "checkpoints/sam2/sam2.1_hiera_base_plus.pt"
download "${SAM2_BASE_URL}/sam2.1_hiera_large.pt" "checkpoints/sam2/sam2.1_hiera_large.pt"
download "${SAM2_BASE_URL}/sam2.1_hiera_tiny.pt" "checkpoints/efficient-sam2/sam2.1_hiera_tiny.pt"
download "${SAM2_BASE_URL}/sam2.1_hiera_small.pt" "checkpoints/efficient-sam2/sam2.1_hiera_small.pt"
download "${SAM2_BASE_URL}/sam2.1_hiera_base_plus.pt" "checkpoints/efficient-sam2/sam2.1_hiera_base_plus.pt"
download "${SAM2_BASE_URL}/sam2.1_hiera_large.pt" "checkpoints/efficient-sam2/sam2.1_hiera_large.pt"

EFFICIENTTAM_BASE_URL="https://huggingface.co/yunyangx/efficient-track-anything/resolve/main"
download "${EFFICIENTTAM_BASE_URL}/efficienttam_ti.pt" "checkpoints/efficienttam/efficienttam_ti.pt"
download "${EFFICIENTTAM_BASE_URL}/efficienttam_s.pt" "checkpoints/efficienttam/efficienttam_s.pt"
