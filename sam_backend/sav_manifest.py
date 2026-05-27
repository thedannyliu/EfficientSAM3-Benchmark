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
    rows = build_sav_manifest(
        args.sav_root,
        args.count,
        args.seed,
        selection_policy=args.selection_policy,
        min_area_ratio=args.min_area_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(args.output)


def build_sav_manifest(
    sav_root: Path,
    count: int,
    seed: int,
    selection_policy: str = "largest_first_mask",
    min_area_ratio: float = 0.0,
    max_aspect_ratio: float = 0.0,
) -> list[dict[str, Any]]:
    frames_root = sav_root / "JPEGImages_24fps"
    annotations_root = sav_root / "Annotations_6fps"
    if not frames_root.is_dir() or not annotations_root.is_dir():
        raise RuntimeError(f"expected SA-V val/test layout under {sav_root}")

    video_ids = _discover_video_ids(sav_root, frames_root, annotations_root)
    rng = random.Random(seed)
    rng.shuffle(video_ids)

    rows = []
    for video_id in video_ids:
        row = _build_row(
            video_id,
            frames_root,
            annotations_root,
            selection_policy=selection_policy,
            min_area_ratio=min_area_ratio,
            max_aspect_ratio=max_aspect_ratio,
        )
        if row is not None:
            rows.append(row)
        if len(rows) == count:
            break
    if len(rows) < count:
        raise RuntimeError(f"only found {len(rows)} eligible SA-V videos, requested {count}")
    return rows


def _discover_video_ids(sav_root: Path, frames_root: Path, annotations_root: Path) -> list[str]:
    split_files = sorted(sav_root.glob("*.txt"))
    if split_files:
        ids = []
        for split_file in split_files:
            ids.extend(line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip())
    else:
        ids = [path.name for path in frames_root.iterdir() if path.is_dir()]
    return sorted({video_id for video_id in ids if (frames_root / video_id).is_dir() and (annotations_root / video_id).is_dir()})


def _build_row(
    video_id: str,
    frames_root: Path,
    annotations_root: Path,
    selection_policy: str = "largest_first_mask",
    min_area_ratio: float = 0.0,
    max_aspect_ratio: float = 0.0,
) -> dict[str, Any] | None:
    frames_dir = frames_root / video_id
    annotations_dir = annotations_root / video_id
    object_dirs = sorted(path for path in annotations_dir.iterdir() if path.is_dir())
    candidates = []
    for object_dir in object_dirs:
        mask_paths = sorted(object_dir.glob("*.png"))
        if not mask_paths:
            continue
        mask_path = mask_paths[0]
        mask = _read_mask(mask_path)
        if mask is None or not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        area = int(mask.sum())
        frame_area = int(mask.shape[0] * mask.shape[1])
        bbox_w = x1 - x0 + 1
        bbox_h = y1 - y0 + 1
        aspect_ratio = float(max(bbox_w / bbox_h, bbox_h / bbox_w)) if bbox_w and bbox_h else 0.0
        candidates.append(
            {
                "object_id": object_dir.name,
                "mask_path": mask_path,
                "frame_index": int(mask_path.stem),
                "area": area,
                "area_ratio": float(area / frame_area) if frame_area else 0.0,
                "point": [float(xs.mean()), float(ys.mean())],
                "height": int(mask.shape[0]),
                "width": int(mask.shape[1]),
                "bbox_xyxy": [x0, y0, x1, y1],
                "bbox_area": int(bbox_w * bbox_h),
                "bbox_area_ratio": float((bbox_w * bbox_h) / frame_area) if frame_area else 0.0,
                "aspect_ratio": aspect_ratio,
            }
        )
    if not candidates:
        return None

    if selection_policy == "largest_first_mask":
        target = max(candidates, key=lambda item: item["area"])
    elif selection_policy == "salient_first_mask":
        filtered = [
            item
            for item in candidates
            if item["area_ratio"] >= min_area_ratio
            and (max_aspect_ratio <= 0.0 or item["aspect_ratio"] <= max_aspect_ratio)
        ]
        if not filtered:
            return None
        target = max(filtered, key=lambda item: (item["area_ratio"], item["bbox_area_ratio"]))
    else:
        raise ValueError(f"unknown SA-V selection policy: {selection_policy}")

    return {
        "sample_id": f"sav_{video_id}_{target['object_id']}",
        "dataset": "sa-v",
        "video_id": video_id,
        "frames_dir": str(frames_dir),
        "annotations_dir": str(annotations_dir),
        "object_id": target["object_id"],
        "prompt_frame_index": target["frame_index"],
        "point": target["point"],
        "point_label": 1,
        "initial_mask_path": str(target["mask_path"]),
        "initial_mask_area": target["area"],
        "initial_mask_area_ratio": target["area_ratio"],
        "initial_bbox_xyxy": target["bbox_xyxy"],
        "initial_bbox_area": target["bbox_area"],
        "initial_bbox_area_ratio": target["bbox_area_ratio"],
        "initial_bbox_aspect_ratio": target["aspect_ratio"],
        "width": target["width"],
        "height": target["height"],
        "selection": f"random_video_seeded_{selection_policy}",
    }


def _read_mask(path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed SA-V val/test manifest for video profiling.")
    parser.add_argument("--sav-root", type=Path, required=True, help="SA-V val/test root with JPEGImages_24fps and Annotations_6fps.")
    parser.add_argument("--output", type=Path, default=Path("data/manifests/sav_fixed3.jsonl"))
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument(
        "--selection-policy",
        choices=["largest_first_mask", "salient_first_mask"],
        default="largest_first_mask",
    )
    parser.add_argument("--min-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-aspect-ratio", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
