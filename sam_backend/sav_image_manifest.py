from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def main() -> None:
    args = parse_args()
    rows = build_sav_image_manifest(
        sav_root=args.sav_root,
        count=args.count,
        seed=args.seed,
        min_area=args.min_area,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(args.output)


def build_sav_image_manifest(sav_root: Path, count: int, seed: int, min_area: int) -> list[dict[str, Any]]:
    frames_root = sav_root / "JPEGImages_24fps"
    annotations_root = sav_root / "Annotations_6fps"
    if not frames_root.is_dir() or not annotations_root.is_dir():
        raise RuntimeError(f"expected SA-V layout under {sav_root}")

    candidates = []
    for video_dir in sorted(annotations_root.iterdir()):
        if not video_dir.is_dir():
            continue
        video_id = video_dir.name
        frames_dir = frames_root / video_id
        if not frames_dir.is_dir():
            continue
        for object_dir in sorted(video_dir.iterdir()):
            if not object_dir.is_dir():
                continue
            for mask_path in sorted(object_dir.glob("*.png")):
                frame_index = _frame_index(mask_path)
                frame_path = _resolve_frame(frames_dir, frame_index)
                if frame_path is None:
                    continue
                mask = _read_mask(mask_path)
                if mask is None:
                    continue
                area = int(mask.sum())
                if area < min_area:
                    continue
                candidates.append((video_id, object_dir.name, frame_index, frame_path, mask_path, mask, area))

    if not candidates:
        raise RuntimeError(f"found no eligible SA-V image samples under {sav_root}")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    if count > 0:
        candidates = candidates[:count]

    rows = []
    for index, (video_id, object_id, frame_index, frame_path, mask_path, mask, area) in enumerate(candidates):
        height, width = mask.shape[:2]
        ys, xs = np.nonzero(mask)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        rows.append(
            {
                "sample_id": f"sav_image_{index:06d}_{video_id}_{object_id}_{frame_index:05d}",
                "dataset": "sa-v-test-image",
                "image_id": f"sav_image_{index:06d}",
                "annotation_id": f"{video_id}_{object_id}_{frame_index:05d}",
                "category_id": 1,
                "category_name": "object",
                "text_prompt": "object",
                "video_id": video_id,
                "object_id": object_id,
                "frame_index": frame_index,
                "image_path": str(frame_path),
                "mask_path": str(mask_path),
                "file_name": frame_path.name,
                "width": int(width),
                "height": int(height),
                "point": [float(xs.mean()), float(ys.mean())],
                "point_label": 1,
                "bbox_xywh": [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                "area": area,
                "segmentation": _mask_polygons(mask),
                "selection": f"random_sa_v_test_image_seed_{seed}",
            }
        )
    return rows


def _frame_index(path: Path) -> int:
    return int(path.stem)


def _resolve_frame(frames_dir: Path, frame_index: int) -> Path | None:
    for name in (
        f"{frame_index:05d}.jpg",
        f"{frame_index:06d}.jpg",
        f"{frame_index:05d}.jpeg",
        f"{frame_index:06d}.jpeg",
        f"{frame_index}.jpg",
        f"{frame_index}.jpeg",
    ):
        path = frames_dir / name
        if path.exists():
            return path
    return None


def _read_mask(path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    values = mask > 0
    return values if bool(values.any()) else None


def _mask_polygons(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        points = contour.reshape(-1, 2)
        if points.shape[0] < 3:
            continue
        polygons.append([float(value) for value in points.reshape(-1)])
    return polygons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a COCO-style image manifest from SA-V frame masks.")
    parser.add_argument("--sav-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000, help="Number of image samples; 0 means all eligible samples.")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--min-area", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
