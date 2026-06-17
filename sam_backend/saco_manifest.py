from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    rows = build_saco_veval_manifest(
        annotation_path=args.annotation,
        media_root=args.media_root,
        output_path=args.output,
        count=args.count,
        seed=args.seed,
        include_negatives=args.include_negatives,
        require_media_exists=args.require_media_exists,
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


def build_saco_veval_manifest(
    annotation_path: Path,
    media_root: Path,
    output_path: Path,
    count: int,
    seed: int,
    include_negatives: bool = False,
    require_media_exists: bool = False,
) -> list[dict[str, Any]]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    videos = {video["id"]: video for video in data.get("videos", [])}
    annotations_by_pair: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for ann in data.get("annotations", []):
        key = (int(ann["video_id"]), int(ann["category_id"]))
        annotations_by_pair.setdefault(key, []).append(ann)

    pairs = list(data.get("video_np_pairs", []))
    rng = random.Random(seed)
    rng.shuffle(pairs)

    rows: list[dict[str, Any]] = []
    seen_videos: set[int] = set()
    for pair in pairs:
        video_id = int(pair["video_id"])
        category_id = int(pair["category_id"])
        if video_id in seen_videos:
            continue
        is_positive = int(pair.get("num_masklets", 0)) > 0
        if not is_positive and not include_negatives:
            continue
        video = videos.get(video_id)
        if video is None:
            continue
        file_names = list(video.get("file_names", []))
        if not file_names:
            continue
        if require_media_exists and not _media_exists(media_root, file_names):
            continue
        annotations = annotations_by_pair.get((video_id, category_id), [])
        rows.append(
            {
                "dataset": "saco-veval-sav",
                "source_id": f"{video.get('video_name', video_id)}_{category_id}",
                "video_id": video_id,
                "video_name": video.get("video_name", str(video_id)),
                "category_id": category_id,
                "noun_phrase": pair.get("noun_phrase", ""),
                "text_prompt": pair.get("noun_phrase", ""),
                "num_masklets": int(pair.get("num_masklets", 0)),
                "is_positive": is_positive,
                "media_root": str(media_root),
                "file_names": file_names,
                "height": int(video.get("height", 0)),
                "width": int(video.get("width", 0)),
                "length": int(video.get("length", len(file_names))),
                "annotations": annotations,
                "annotation_file": str(annotation_path),
                "prompt_frame_index": 0,
            }
        )
        seen_videos.add(video_id)
        if count > 0 and len(rows) >= count:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def _media_exists(media_root: Path, file_names: list[str]) -> bool:
    if not file_names:
        return False
    first = media_root / file_names[0]
    last = media_root / file_names[-1]
    return first.exists() and last.exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed SA-Co/VEval-SAV stream benchmark manifest.")
    parser.add_argument("--annotation", type=Path, required=True, help="SA-Co/VEval annotation JSON.")
    parser.add_argument("--media-root", type=Path, required=True, help="Root containing SA-Co/VEval SAV frames.")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--include-negatives", action="store_true")
    parser.add_argument("--require-media-exists", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
