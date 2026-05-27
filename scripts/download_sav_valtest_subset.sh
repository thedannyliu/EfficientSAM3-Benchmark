#!/usr/bin/env bash
set -euo pipefail

SPLIT="${1:-val}"
COUNT="${2:-3}"
OUT_ROOT="${3:-data/sa-v/sav_${SPLIT}_fixed${COUNT}}"
SEED="${SAV_SEED:-20260527}"
ARCHIVE_DIR="${SAV_ARCHIVE_DIR:-data/sa-v/_archives}"
PYTHON_BIN="${PYTHON:-python}"
SELECTION_POLICY="${SAV_SELECTION_POLICY:-largest_first_mask}"
MIN_AREA_RATIO="${SAV_MIN_AREA_RATIO:-0.0}"
MAX_ASPECT_RATIO="${SAV_MAX_ASPECT_RATIO:-0.0}"
if [[ -n "${SAV_CANDIDATE_COUNT:-}" ]]; then
  CANDIDATE_COUNT="${SAV_CANDIDATE_COUNT}"
elif [[ "${SELECTION_POLICY}" == "salient_first_mask" ]]; then
  CANDIDATE_COUNT=$((COUNT * 10))
else
  CANDIDATE_COUNT="${COUNT}"
fi

if [[ "${SPLIT}" != "val" && "${SPLIT}" != "test" ]]; then
  echo "split must be 'val' or 'test', got: ${SPLIT}" >&2
  exit 2
fi

if [[ -z "${OUT_ROOT}" || "${OUT_ROOT}" == "/" ]]; then
  echo "refusing unsafe OUT_ROOT: ${OUT_ROOT}" >&2
  exit 2
fi

mkdir -p "${ARCHIVE_DIR}" data/manifests

SPLIT_NAME="sav_${SPLIT}"
ARCHIVE="${ARCHIVE_DIR}/${SPLIT_NAME}.tar"
LINKS_FILE="${ARCHIVE_DIR}/official_sa_v_links.tsv"
MANIFEST_PREFIX="${SAV_MANIFEST_PREFIX:-${SPLIT_NAME}_fixed${COUNT}}"
MANIFEST="data/manifests/${MANIFEST_PREFIX}.jsonl"
TRACKED_RECORD="data/manifests/${MANIFEST_PREFIX}_selection.json"

write_selection_record() {
  "${PYTHON_BIN}" - "${OUT_ROOT}" "${MANIFEST}" "${TRACKED_RECORD}" "${SPLIT}" "${COUNT}" "${SEED}" "${SELECTION_POLICY}" "${MIN_AREA_RATIO}" "${MAX_ASPECT_RATIO}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
tracked_record = Path(sys.argv[3])
split = sys.argv[4]
count = int(sys.argv[5])
seed = int(sys.argv[6])
selection_policy = sys.argv[7]
min_area_ratio = float(sys.argv[8])
max_aspect_ratio = float(sys.argv[9])
official_record = out_root / "official_subset_manifest.json"
official = json.loads(official_record.read_text(encoding="utf-8")) if official_record.exists() else {}
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
payload = {
    "source": "Meta SA-V official val/test archive",
    "split": split,
    "count": count,
    "seed": seed,
    "manifest": str(manifest),
    "official_subset_manifest": str(official_record),
    "video_ids": official.get("video_ids", [row["video_id"] for row in rows]),
    "selection": f"random_video_seeded_{selection_policy}",
    "selection_filters": {
        "min_area_ratio": min_area_ratio,
        "max_aspect_ratio": max_aspect_ratio,
    },
    "prompt_protocol": {
        "point": "centroid of selected object's first available annotation mask",
        "text": "not used; official SA-V val/test has no semantic object labels",
    },
    "samples": [
        {
            "sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "object_id": row["object_id"],
            "prompt_frame_index": row["prompt_frame_index"],
            "point": row["point"],
            "initial_mask_area": row["initial_mask_area"],
            "initial_mask_area_ratio": row.get("initial_mask_area_ratio"),
            "initial_bbox_xyxy": row.get("initial_bbox_xyxy"),
            "initial_bbox_aspect_ratio": row.get("initial_bbox_aspect_ratio"),
            "width": row["width"],
            "height": row["height"],
        }
        for row in rows
    ],
}
tracked_record.parent.mkdir(parents=True, exist_ok=True)
tracked_record.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(tracked_record)
PY
}

if [[ -d "${OUT_ROOT}/JPEGImages_24fps" && -d "${OUT_ROOT}/Annotations_6fps" ]]; then
  "${PYTHON_BIN}" -m sam_backend.sav_manifest \
    --sav-root "${OUT_ROOT}" \
    --output "${MANIFEST}" \
    --count "${COUNT}" \
    --seed "${SEED}" \
    --selection-policy "${SELECTION_POLICY}" \
    --min-area-ratio "${MIN_AREA_RATIO}" \
    --max-aspect-ratio "${MAX_ASPECT_RATIO}"
  write_selection_record
  echo "${OUT_ROOT}"
  exit 0
fi

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

WORK_DIR="${OUT_ROOT}.work"
if [[ -d "${WORK_DIR}" ]]; then
  chmod -R u+w "${WORK_DIR}" || true
fi
rm -rf "${WORK_DIR}" "${OUT_ROOT}.tmp"
mkdir -p "${WORK_DIR}/list" "${WORK_DIR}/extract" "${OUT_ROOT}.tmp"

SPLIT_FILE="${SPLIT_NAME}.txt"
SPLIT_MEMBER="$(tar -tf "${ARCHIVE}" | awk -v name="${SPLIT_FILE}" '$0 == name || $0 ~ "/" name "$" {print; exit}')"
if [[ -z "${SPLIT_MEMBER}" ]]; then
  echo "could not find ${SPLIT_FILE} in ${ARCHIVE}" >&2
  exit 1
fi

tar --no-same-permissions -xf "${ARCHIVE}" -C "${WORK_DIR}/list" "${SPLIT_MEMBER}"

"${PYTHON_BIN}" - "${WORK_DIR}/list/${SPLIT_MEMBER}" "${CANDIDATE_COUNT}" "${SEED}" "${WORK_DIR}/selected_ids.txt" <<'PY'
from __future__ import annotations

import random
import sys
from pathlib import Path

split_file = Path(sys.argv[1])
count = int(sys.argv[2])
seed = int(sys.argv[3])
out = Path(sys.argv[4])
ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
rng = random.Random(seed)
rng.shuffle(ids)
selected = ids[:count]
if len(selected) < count:
    raise SystemExit(f"only found {len(selected)} videos, requested {count}")
out.write_text("\n".join(selected) + "\n", encoding="utf-8")
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
ann_prefix = f"{prefix}/Annotations_6fps/" if prefix else "Annotations_6fps/"
keep = [split_member]

listing = subprocess.check_output(["tar", "-tf", str(archive)], text=True)
for member in listing.splitlines():
    if member.endswith("/"):
        continue
    for base in (frame_prefix, ann_prefix):
        if member.startswith(base):
            rest = member[len(base):]
            video_id = rest.split("/", 1)[0]
            if video_id in selected:
                keep.append(member)
            break

if len(keep) <= 1:
    raise SystemExit("no selected frame/annotation members found in archive")
out.write_text("\n".join(keep) + "\n", encoding="utf-8")
PY

tar --no-same-permissions -xf "${ARCHIVE}" -C "${WORK_DIR}/extract" -T "${WORK_DIR}/members.txt"

"${PYTHON_BIN}" - "${WORK_DIR}/extract" "${SPLIT_MEMBER}" "${WORK_DIR}/selected_ids.txt" "${OUT_ROOT}.tmp" "${SPLIT_NAME}" "${SEED}" <<'PY'
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

extract_root = Path(sys.argv[1])
split_member = sys.argv[2]
selected_file = Path(sys.argv[3])
out_root = Path(sys.argv[4])
split_name = sys.argv[5]
seed = int(sys.argv[6])
prefix = split_member.removesuffix("/" + Path(split_member).name)
src_root = extract_root / prefix if prefix else extract_root
selected = [line.strip() for line in selected_file.read_text(encoding="utf-8").splitlines() if line.strip()]

for dirname in ("JPEGImages_24fps", "Annotations_6fps"):
    (out_root / dirname).mkdir(parents=True, exist_ok=True)
    for video_id in selected:
        src = src_root / dirname / video_id
        dst = out_root / dirname / video_id
        if not src.exists():
            raise SystemExit(f"missing extracted {dirname}/{video_id}")
        shutil.copytree(src, dst)

(out_root / f"{split_name}.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
(out_root / "official_subset_manifest.json").write_text(
    json.dumps(
        {
            "source": "Meta SA-V official val/test archive",
            "split": split_name.removeprefix("sav_"),
            "seed": seed,
            "count": len(selected),
            "video_ids": selected,
            "layout": "JPEGImages_24fps + Annotations_6fps",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

rm -rf "${OUT_ROOT}"
mv "${OUT_ROOT}.tmp" "${OUT_ROOT}"
chmod -R u+w "${WORK_DIR}" || true
rm -rf "${WORK_DIR}"

if [[ "${KEEP_SAV_ARCHIVE:-0}" != "1" ]]; then
  rm -f "${ARCHIVE}"
fi

"${PYTHON_BIN}" -m sam_backend.sav_manifest \
  --sav-root "${OUT_ROOT}" \
  --output "${MANIFEST}" \
  --count "${COUNT}" \
  --seed "${SEED}" \
  --selection-policy "${SELECTION_POLICY}" \
  --min-area-ratio "${MIN_AREA_RATIO}" \
  --max-aspect-ratio "${MAX_ASPECT_RATIO}"

if [[ "${SAV_PRUNE_TO_MANIFEST:-1}" == "1" ]]; then
  "${PYTHON_BIN}" - "${OUT_ROOT}" "${MANIFEST}" "${SPLIT_NAME}" "${SEED}" <<'PY'
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
split_name = sys.argv[3]
seed = int(sys.argv[4])
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
selected = [row["video_id"] for row in rows]
selected_set = set(selected)
for dirname in ("JPEGImages_24fps", "Annotations_6fps"):
    root = out_root / dirname
    for video_dir in root.iterdir():
        if video_dir.is_dir() and video_dir.name not in selected_set:
            shutil.rmtree(video_dir)
(out_root / f"{split_name}.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
(out_root / "official_subset_manifest.json").write_text(
    json.dumps(
        {
            "source": "Meta SA-V official val/test archive",
            "split": split_name.removeprefix("sav_"),
            "seed": seed,
            "count": len(selected),
            "video_ids": selected,
            "layout": "JPEGImages_24fps + Annotations_6fps",
            "note": "Pruned to videos selected by sam_backend.sav_manifest.",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
fi

write_selection_record

echo "${OUT_ROOT}"
