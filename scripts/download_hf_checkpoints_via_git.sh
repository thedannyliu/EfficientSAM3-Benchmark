#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-checkpoints}"
HF_CLONE_ROOT="${HF_CLONE_ROOT:-external/hf-checkpoints}"
HF_TOKEN="${HF_TOKEN:-}"
HF_CA_BUNDLE="${HF_CA_BUNDLE:-}"
HF_GIT_TIMEOUT="${HF_GIT_TIMEOUT:-300}"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is required." >&2
  exit 2
fi

if ! git lfs version >/dev/null 2>&1; then
  echo "ERROR: git-lfs is required. Install it with: sudo apt install -y git-lfs && git lfs install" >&2
  exit 2
fi

if [[ -n "${HF_CA_BUNDLE}" ]]; then
  if [[ ! -f "${HF_CA_BUNDLE}" ]]; then
    echo "ERROR: HF_CA_BUNDLE does not exist: ${HF_CA_BUNDLE}" >&2
    exit 2
  fi
  export GIT_SSL_CAINFO="${HF_CA_BUNDLE}"
  export SSL_CERT_FILE="${HF_CA_BUNDLE}"
  export REQUESTS_CA_BUNDLE="${HF_CA_BUNDLE}"
  export CURL_CA_BUNDLE="${HF_CA_BUNDLE}"
  echo "CA bundle: ${HF_CA_BUNDLE}"
fi

repo_header_args=()
if [[ -n "${HF_TOKEN}" ]]; then
  repo_header_args=(-c "http.extraHeader=Authorization: Bearer ${HF_TOKEN}")
fi

mkdir -p "${OUT_DIR}" "${HF_CLONE_ROOT}"

run_git() {
  local label="$1"
  shift
  echo "Running: ${label} (timeout ${HF_GIT_TIMEOUT}s)"
  if ! env GIT_TERMINAL_PROMPT=0 timeout --foreground "${HF_GIT_TIMEOUT}" "$@"; then
    echo "ERROR: ${label} failed or timed out." >&2
    echo "If this is an SSL self-signed certificate error, rerun with HF_CA_BUNDLE=/path/to/root-ca.pem." >&2
    echo "If this is an authentication error, create a new read token and pass HF_TOKEN without printing it." >&2
    exit 2
  fi
}

clone_or_update() {
  local repo_id="$1"
  local dest="$2"
  local url="https://huggingface.co/${repo_id}"

  if [[ -d "${dest}/.git" ]]; then
    run_git "fetch ${repo_id}" git -C "${dest}" "${repo_header_args[@]}" fetch --progress --depth 1 origin main
    git -C "${dest}" checkout -q FETCH_HEAD
  elif [[ -e "${dest}" ]]; then
    echo "Removing incomplete Hugging Face clone: ${dest}"
    rm -rf "${dest}"
    run_git "clone ${repo_id}" env GIT_LFS_SKIP_SMUDGE=1 git "${repo_header_args[@]}" clone --progress --depth 1 "${url}" "${dest}"
  else
    run_git "clone ${repo_id}" env GIT_LFS_SKIP_SMUDGE=1 git "${repo_header_args[@]}" clone --progress --depth 1 "${url}" "${dest}"
  fi
}

lfs_pull() {
  local dest="$1"
  local include_csv="$2"
  run_git "git-lfs pull ${dest}" git -C "${dest}" "${repo_header_args[@]}" lfs pull --include "${include_csv}" --exclude ""
}

copy_file() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "${dst}")"
  cp -f "${src}" "${dst}"
  echo "${dst}"
}

sam3_repo="${HF_CLONE_ROOT}/facebook__sam3"
clone_or_update "facebook/sam3" "${sam3_repo}"
lfs_pull "${sam3_repo}" "config.json,sam3.pt"
copy_file "${sam3_repo}/config.json" "${OUT_DIR}/sam3/config.json"
copy_file "${sam3_repo}/sam3.pt" "${OUT_DIR}/sam3/sam3.pt"

eff_repo="${HF_CLONE_ROOT}/Simon7108528__EfficientSAM3"
clone_or_update "Simon7108528/EfficientSAM3" "${eff_repo}"
lfs_pull "${eff_repo}" "stage1_sam3p1/efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt,stage1_sam3p1/efficient_sam3p1_efficientvit_l_mobileclip_s0_ctx16.pt,stage1_all_converted/efficient_sam3_efficientvit-b0_mobileclip_s1.pth,stage1_all_converted/efficient_sam3_efficientvit-b2_mobileclip_s1.pth"
copy_file "${eff_repo}/stage1_sam3p1/efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt" "${OUT_DIR}/stage1_sam3p1/efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt"
copy_file "${eff_repo}/stage1_sam3p1/efficient_sam3p1_efficientvit_l_mobileclip_s0_ctx16.pt" "${OUT_DIR}/stage1_sam3p1/efficient_sam3p1_efficientvit_l_mobileclip_s0_ctx16.pt"
copy_file "${eff_repo}/stage1_all_converted/efficient_sam3_efficientvit-b0_mobileclip_s1.pth" "${OUT_DIR}/stage1_all_converted/efficient_sam3_efficientvit-b0_mobileclip_s1.pth"
copy_file "${eff_repo}/stage1_all_converted/efficient_sam3_efficientvit-b2_mobileclip_s1.pth" "${OUT_DIR}/stage1_all_converted/efficient_sam3_efficientvit-b2_mobileclip_s1.pth"

echo "Hugging Face git-lfs checkpoints are ready under ${OUT_DIR}."
