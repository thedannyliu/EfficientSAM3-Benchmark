from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2

from .coco_manifest import ann_to_mask


def main() -> None:
    args = parse_args()
    output = write_mask_layout(args.manifest, args.output_root, copy_images=args.copy_images)
    print(output)


def write_mask_layout(manifest: Path, output_root: Path, copy_images: bool = False) -> Path:
    rows = _read_manifest(manifest)
    frames_root = output_root / "JPEGImages_24fps"
    anns_root = output_root / "Annotations_6fps"
    frames_root.mkdir(parents=True, exist_ok=True)
    anns_root.mkdir(parents=True, exist_ok=True)

    video_ids = []
    for index, row in enumerate(rows):
        video_id = _safe_name(str(row.get("sample_id") or f"sample_{index:04d}"))
        video_ids.append(video_id)
        image_path = Path(str(row["image_path"]))
        video_frames = frames_root / video_id
        object_dir = anns_root / video_id / "1"
        video_frames.mkdir(parents=True, exist_ok=True)
        object_dir.mkdir(parents=True, exist_ok=True)

        frame_path = video_frames / "00000.jpg"
        _place_image(image_path, frame_path, copy_images)

        mask = ann_to_mask(row, int(row["width"]), int(row["height"]))
        if mask is None:
            raise RuntimeError(f"failed to decode mask for {row.get('sample_id', image_path)}")
        ok = cv2.imwrite(str(object_dir / "00000.png"), mask.astype("uint8") * 255)
        if not ok:
            raise RuntimeError(f"failed to write mask for {row.get('sample_id', image_path)}")

    (output_root / "sav_train_benchmark.txt").write_text(
        "".join(f"{video_id}\n" for video_id in video_ids),
        encoding="utf-8",
    )
    return output_root


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _place_image(source: Path, destination: Path, copy_images: bool) -> None:
    if destination.exists() or destination.is_symlink():
        return
    if copy_images:
        shutil.copy2(source, destination)
        return
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert prompt/eval image manifest rows into a one-frame mask layout.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking them.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
