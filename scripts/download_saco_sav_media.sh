#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:?usage: scripts/download_saco_sav_media.sh MANIFEST_JSONL MEDIA_ROOT [val|test]}"
MEDIA_ROOT="${2:?usage: scripts/download_saco_sav_media.sh MANIFEST_JSONL MEDIA_ROOT [val|test]}"
SPLIT="${3:-val}"

if [[ "${SPLIT}" != "val" && "${SPLIT}" != "test" ]]; then
  echo "split must be 'val' or 'test', got: ${SPLIT}" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

asset_root="${SAM_BENCH_SCRATCH:-${repo_root}}"
ARCHIVE_DIR="${SAV_ARCHIVE_DIR:-${asset_root}/data/sa-v/_archives}"
PYTHON_BIN="${PYTHON:-python}"
SPLIT_NAME="sav_${SPLIT}"
ARCHIVE="${ARCHIVE_DIR}/${SPLIT_NAME}.tar"
LINKS_FILE="${ARCHIVE_DIR}/official_sa_v_links.tsv"

mkdir -p "${ARCHIVE_DIR}" "$(dirname "${MEDIA_ROOT}")"

resolve_archive_url() {
  if [[ -n "${SAV_ARCHIVE_URL:-}" ]]; then
    printf '%s\n' "${SAV_ARCHIVE_URL}"
    return
  fi
  "${PYTHON_BIN}" - "${SPLIT_NAME}.tar" "${LINKS_FILE}" "${SAV_LINKS_FILE:-}" <<'PY'
from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

target_name = sys.argv[1]
links_out = Path(sys.argv[2])
provided_links = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None


def parse_links(text: str) -> str:
    for line in text.splitlines():
        if not line.strip() or line.startswith("file_name\t"):
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0] == target_name:
            return parts[1]
    raise SystemExit(f"could not find {target_name} in official SA-V links")


if provided_links is not None:
    print(parse_links(provided_links.read_text(encoding="utf-8")))
    raise SystemExit(0)

if links_out.exists():
    print(parse_links(links_out.read_text(encoding="utf-8")))
    raise SystemExit(0)

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*",
    "Referer": "https://ai.meta.com/datasets/segment-anything-video-downloads/",
}
page_request = urllib.request.Request("https://ai.meta.com/datasets/segment-anything-video-downloads/", headers=headers)
with urllib.request.urlopen(page_request, timeout=30) as response:
    html = response.read().decode("utf-8", "replace")
lsd_match = re.search(r'\["LSD",\[\],\{"token":"([^"]+)"', html)
spin_r_match = re.search(r'"__spin_r":(\d+)', html)
spin_t_match = re.search(r'"__spin_t":(\d+)', html)
if not (lsd_match and spin_r_match and spin_t_match):
    raise SystemExit("could not resolve Meta download token fields")

lsd = lsd_match.group(1)
spin_r = spin_r_match.group(1)
spin_t = spin_t_match.group(1)
jazoest = "2" + "".join(str(ord(ch)) for ch in lsd)
payload = {
    "av": "0",
    "__user": "0",
    "__a": "1",
    "__req": "1",
    "__rev": spin_r,
    "__comet_req": "0",
    "lsd": lsd,
    "jazoest": jazoest,
    "__spin_r": spin_r,
    "__spin_b": "trunk",
    "__spin_t": spin_t,
    "fb_api_caller_class": "RelayModern",
    "fb_api_req_friendly_name": "AIUseDatasetEntFileNamesSelfServiceQuery",
    "variables": json.dumps({"input": {"bucket": "sam2_release", "getDownloadAllFile": True}}, separators=(",", ":")),
    "server_timestamps": "true",
    "doc_id": "9947021732020473",
}
encoded = urllib.parse.urlencode(payload).encode("utf-8")
graphql_request = urllib.request.Request(
    "https://ai.meta.com/api/graphql/",
    data=encoded,
    headers={**headers, "x-fb-lsd": lsd, "Content-Type": "application/x-www-form-urlencoded"},
    method="POST",
)
with urllib.request.urlopen(graphql_request, timeout=30) as response:
    graph_data = json.loads(response.read().decode("utf-8", "replace"))
download_all_url = graph_data["data"]["datasetFiles"][0]["url"]
links_request = urllib.request.Request(download_all_url, headers=headers)
with urllib.request.urlopen(links_request, timeout=60) as response:
    links_text = response.read().decode("utf-8", "replace")
links_out.parent.mkdir(parents=True, exist_ok=True)
links_out.write_text(links_text, encoding="utf-8")
print(parse_links(links_text))
PY
}

ARCHIVE_URL="$(resolve_archive_url)"

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "download ${SPLIT_NAME}.tar -> ${ARCHIVE}"
  if command -v wget >/dev/null 2>&1; then
    wget -c --progress=dot:giga "${ARCHIVE_URL}" -O "${ARCHIVE}"
  else
    curl -L --fail --retry 3 -C - "${ARCHIVE_URL}" -o "${ARCHIVE}"
  fi
else
  echo "exists: ${ARCHIVE}"
fi

WORK_DIR="${MEDIA_ROOT}.work"
if [[ -d "${WORK_DIR}" ]]; then
  chmod -R u+w "${WORK_DIR}" || true
fi
rm -rf "${WORK_DIR}" "${MEDIA_ROOT}.tmp"
mkdir -p "${WORK_DIR}/list" "${WORK_DIR}/extract" "${MEDIA_ROOT}.tmp"

SPLIT_FILE="${SPLIT_NAME}.txt"
SPLIT_MEMBER="$(tar -tf "${ARCHIVE}" | awk -v name="${SPLIT_FILE}" '$0 == name || $0 ~ "/" name "$" {print; exit}')"
if [[ -z "${SPLIT_MEMBER}" ]]; then
  echo "could not find ${SPLIT_FILE} in ${ARCHIVE}" >&2
  exit 1
fi

tar --no-same-permissions -xf "${ARCHIVE}" -C "${WORK_DIR}/list" "${SPLIT_MEMBER}"

"${PYTHON_BIN}" - "${MANIFEST}" "${WORK_DIR}/selected_ids.txt" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
out = Path(sys.argv[2])
video_ids: list[str] = []
seen: set[str] = set()
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    file_names = row.get("file_names") or []
    video_name = row.get("video_name") or ""
    if file_names:
        video_id = str(file_names[0]).split("/", 1)[0]
    elif video_name:
        video_id = str(video_name)
    else:
        video_id = str(row["video_id"])
    if video_id not in seen:
        video_ids.append(video_id)
        seen.add(video_id)
if not video_ids:
    raise SystemExit("manifest did not contain any SA-V video ids")
out.write_text("\n".join(video_ids) + "\n", encoding="utf-8")
print(f"SA-Co media videos: {len(video_ids)}")
PY

"${PYTHON_BIN}" - "${ARCHIVE}" "${SPLIT_MEMBER}" "${WORK_DIR}/selected_ids.txt" "${WORK_DIR}/members.txt" <<'PY'
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

archive = Path(sys.argv[1])
split_member = sys.argv[2]
selected = set(Path(sys.argv[3]).read_text(encoding="utf-8").splitlines())
out = Path(sys.argv[4])
prefix = split_member.removesuffix("/" + Path(split_member).name)
frame_prefix = f"{prefix}/JPEGImages_24fps/" if prefix else "JPEGImages_24fps/"
keep = [split_member]

listing = subprocess.check_output(["tar", "-tf", str(archive)], text=True)
for member in listing.splitlines():
    if member.endswith("/") or not member.startswith(frame_prefix):
        continue
    rest = member[len(frame_prefix):]
    video_id = rest.split("/", 1)[0]
    if video_id in selected:
        keep.append(member)

if len(keep) <= 1:
    raise SystemExit("no selected frame members found in archive")
out.write_text("\n".join(keep) + "\n", encoding="utf-8")
PY

tar --no-same-permissions -xf "${ARCHIVE}" -C "${WORK_DIR}/extract" -T "${WORK_DIR}/members.txt"

"${PYTHON_BIN}" - "${WORK_DIR}/extract" "${SPLIT_MEMBER}" "${WORK_DIR}/selected_ids.txt" "${MEDIA_ROOT}.tmp" <<'PY'
from __future__ import annotations

import shutil
import sys
from pathlib import Path

extract_root = Path(sys.argv[1])
split_member = sys.argv[2]
selected_file = Path(sys.argv[3])
out_root = Path(sys.argv[4])
prefix = split_member.removesuffix("/" + Path(split_member).name)
src_root = extract_root / prefix if prefix else extract_root
selected = [line.strip() for line in selected_file.read_text(encoding="utf-8").splitlines() if line.strip()]

out_root.mkdir(parents=True, exist_ok=True)
for video_id in selected:
    src = src_root / "JPEGImages_24fps" / video_id
    dst = out_root / video_id
    if not src.exists():
        raise SystemExit(f"missing extracted JPEGImages_24fps/{video_id}")
    shutil.copytree(src, dst)
PY

rm -rf "${MEDIA_ROOT}"
mv "${MEDIA_ROOT}.tmp" "${MEDIA_ROOT}"
chmod -R u+w "${WORK_DIR}" || true
rm -rf "${WORK_DIR}"

if [[ "${KEEP_SAV_ARCHIVE:-0}" != "1" ]]; then
  rm -f "${ARCHIVE}"
fi

echo "${MEDIA_ROOT}"
