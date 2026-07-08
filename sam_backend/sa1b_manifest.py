from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image

from .coco_manifest import ann_to_mask, foreground_point


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def main() -> None:
    args = parse_args()
    rows = build_sa1b_manifest(
        annotation_root=args.annotation_root,
        image_root=args.image_root or args.annotation_root,
        count=args.count,
        seed=args.seed,
        min_area=args.min_area,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(args.output)


def build_sa1b_manifest(
    annotation_root: Path,
    image_root: Path,
    count: int,
    seed: int,
    min_area: float,
) -> list[dict[str, Any]]:
    json_paths = sorted(path for path in annotation_root.rglob("*.json") if path.is_file())
    rng = random.Random(seed)
    rng.shuffle(json_paths)

    rows = []
    for json_path in json_paths:
        row = _build_row(json_path, annotation_root, image_root, len(rows), min_area)
        if row is not None:
            rows.append(row)
        if len(rows) == count:
            break
    if len(rows) < count:
        raise RuntimeError(f"only found {len(rows)} eligible SA-1B images, requested {count}")
    return rows


def _build_row(
    json_path: Path,
    annotation_root: Path,
    image_root: Path,
    index: int,
    min_area: float,
) -> dict[str, Any] | None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image_path = _resolve_image_path(data, json_path, annotation_root, image_root)
    if image_path is None:
        return None
    width, height = _image_size(data, image_path)

    annotations = data.get("annotations")
    if not isinstance(annotations, list):
        return None
    candidates = []
    for ann_index, ann in enumerate(annotations):
        if not isinstance(ann, dict):
            continue
        if ann.get("iscrowd", 0) or float(ann.get("area", 0.0)) < min_area:
            continue
        mask = ann_to_mask(ann, width, height)
        if mask is None or not mask.any():
            continue
        candidates.append((float(ann.get("area", float(mask.sum()))), ann_index, ann))
    if not candidates:
        return None

    _, ann_index, ann = max(candidates, key=lambda item: item[0])
    point_x, point_y = foreground_point(ann, width, height)
    image_id = _image_id(data, json_path)
    bbox = ann.get("bbox", [])
    return {
        "sample_id": f"sa1b_{index:04d}_{image_id}_{ann_index}",
        "dataset": "sa1b",
        "image_id": image_id,
        "annotation_id": ann.get("id", ann_index),
        "category_id": 1,
        "category_name": "object",
        "text_prompt": "object",
        "point": [point_x, point_y],
        "point_label": 1,
        "image_path": str(image_path),
        "file_name": image_path.name,
        "width": width,
        "height": height,
        "bbox_xywh": bbox,
        "area": ann.get("area", 0.0),
        "iscrowd": ann.get("iscrowd", 0),
        "segmentation": ann.get("segmentation"),
        "selection": "random_json_seeded_largest_mask",
    }


def _resolve_image_path(data: dict[str, Any], json_path: Path, annotation_root: Path, image_root: Path) -> Path | None:
    image_info = data.get("image") if isinstance(data.get("image"), dict) else {}
    file_name = image_info.get("file_name") or data.get("file_name")
    candidates = []
    if isinstance(file_name, str) and file_name:
        candidates.extend(
            [
                json_path.parent / file_name,
                image_root / file_name,
                image_root / Path(file_name).name,
            ]
        )
    candidates.extend(json_path.with_suffix(suffix) for suffix in IMAGE_SUFFIXES)
    rel_parent = json_path.parent.relative_to(annotation_root)
    candidates.extend(image_root / rel_parent / f"{json_path.stem}{suffix}" for suffix in IMAGE_SUFFIXES)
    candidates.extend(image_root / f"{json_path.stem}{suffix}" for suffix in IMAGE_SUFFIXES)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _image_size(data: dict[str, Any], image_path: Path) -> tuple[int, int]:
    image_info = data.get("image") if isinstance(data.get("image"), dict) else {}
    width = image_info.get("width") or data.get("width")
    height = image_info.get("height") or data.get("height")
    if width and height:
        return int(width), int(height)
    with Image.open(image_path) as image:
        return int(image.width), int(image.height)


def _image_id(data: dict[str, Any], json_path: Path) -> str:
    image_info = data.get("image") if isinstance(data.get("image"), dict) else {}
    value = image_info.get("image_id") or image_info.get("id") or data.get("image_id") or data.get("id")
    return str(value if value is not None else json_path.stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed SA-1B prompt/eval manifest from extracted SA-1B JSON files.")
    parser.add_argument("--annotation-root", type=Path, required=True, help="Root containing extracted SA-1B JSON annotations.")
    parser.add_argument("--image-root", type=Path, help="Root containing extracted SA-1B images. Defaults to --annotation-root.")
    parser.add_argument("--output", type=Path, default=Path("data/manifests/sa1b_fixed100.jsonl"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--min-area", type=float, default=1024.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
