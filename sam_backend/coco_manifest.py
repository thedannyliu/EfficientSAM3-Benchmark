from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def main() -> None:
    args = parse_args()
    rows = build_coco_manifest(
        annotations=args.annotations,
        image_dir=args.image_dir,
        count=args.count,
        seed=args.seed,
        min_area=args.min_area,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(args.output)


def build_coco_manifest(
    annotations: Path,
    image_dir: Path,
    count: int,
    seed: int,
    min_area: float,
) -> list[dict[str, Any]]:
    data = json.loads(annotations.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in data["images"]}
    categories = {item["id"]: item["name"] for item in data["categories"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) or float(ann.get("area", 0.0)) < min_area:
            continue
        if ann.get("category_id") not in categories:
            continue
        if not _supported_segmentation(ann.get("segmentation")):
            continue
        anns_by_image[ann["image_id"]].append(ann)

    image_ids = [image_id for image_id, anns in anns_by_image.items() if anns and image_id in images]
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    selected = image_ids[:count]
    if len(selected) < count:
        raise RuntimeError(f"only found {len(selected)} eligible COCO images, requested {count}")

    rows = []
    for index, image_id in enumerate(selected):
        image = images[image_id]
        ann = max(anns_by_image[image_id], key=lambda item: float(item.get("area", 0.0)))
        category_name = categories[ann["category_id"]]
        width = int(image["width"])
        height = int(image["height"])
        point_x, point_y = foreground_point(ann, width, height)
        rows.append(
            {
                "sample_id": f"coco_{index:02d}_{image_id}_{ann['id']}",
                "dataset": "coco",
                "image_id": image_id,
                "annotation_id": ann["id"],
                "category_id": ann["category_id"],
                "category_name": category_name,
                "text_prompt": category_name,
                "point": [point_x, point_y],
                "point_label": 1,
                "image_path": str(image_dir / image["file_name"]),
                "file_name": image["file_name"],
                "width": width,
                "height": height,
                "bbox_xywh": ann.get("bbox", []),
                "area": ann.get("area", 0.0),
                "iscrowd": ann.get("iscrowd", 0),
                "segmentation": ann.get("segmentation"),
                "selection": "random_image_seeded_largest_non_crowd_object",
            }
        )
    return rows


def foreground_point(ann: dict[str, Any], width: int, height: int) -> tuple[float, float]:
    mask = ann_to_mask(ann, width, height)
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        return float(xs.mean()), float(ys.mean())

    bbox = ann.get("bbox") or [0.0, 0.0, width, height]
    return float(bbox[0] + bbox[2] / 2.0), float(bbox[1] + bbox[3] / 2.0)


def ann_to_mask(ann: dict[str, Any], width: int, height: int) -> np.ndarray | None:
    segmentation = ann.get("segmentation")
    if isinstance(segmentation, list):
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in segmentation:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(points) >= 3:
                cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
        return mask.astype(bool)

    decoded = _decode_rle(segmentation, width, height)
    if decoded is not None:
        return decoded.astype(bool)
    return None


def _decode_rle(segmentation: Any, width: int, height: int) -> np.ndarray | None:
    if not isinstance(segmentation, dict):
        return None
    try:
        from pycocotools import mask as mask_utils  # type: ignore[import-not-found]
    except ImportError:
        return None

    rle = segmentation
    if isinstance(rle.get("counts"), list):
        rle = mask_utils.frPyObjects(rle, height, width)
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return np.asarray(decoded, dtype=bool)


def _supported_segmentation(segmentation: Any) -> bool:
    if isinstance(segmentation, list) and segmentation:
        return True
    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as _mask_utils  # noqa: F401
        except ImportError:
            return False
        return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed COCO image manifest for prompt/eval profiling.")
    parser.add_argument("--annotations", type=Path, required=True, help="COCO instances JSON, e.g. instances_val2017.json.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing COCO image files.")
    parser.add_argument("--output", type=Path, default=Path("data/manifests/coco_val2017_fixed10.jsonl"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--min-area", type=float, default=1024.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
