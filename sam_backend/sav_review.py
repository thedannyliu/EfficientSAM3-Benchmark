from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def main() -> None:
    args = parse_args()
    summary = write_sav_review(args.manifest, args.output_dir)
    print(json.dumps(summary, indent=2))


def write_sav_review(manifest: Path, output_dir: Path) -> dict[str, Any]:
    rows = _read_manifest(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    tiles = []
    for row in rows:
        frame_path = Path(row["frames_dir"]) / f"{int(row['prompt_frame_index']):05d}.jpg"
        mask_path = Path(row["initial_mask_path"])
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None:
            raise RuntimeError(f"failed to read SA-V review inputs for {row['sample_id']}")
        mask_bool = mask > 0
        overlay = overlay_initial_mask(frame, mask_bool, row)
        out_path = output_dir / f"{row['sample_id']}.png"
        if not cv2.imwrite(str(out_path), overlay):
            raise RuntimeError(f"failed to write SA-V review overlay: {out_path}")
        tiles.append(_resize_tile(overlay, width=480))
        samples.append(
            {
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "object_id": row["object_id"],
                "prompt_frame_index": row["prompt_frame_index"],
                "point": row["point"],
                "initial_mask_area": row["initial_mask_area"],
                "initial_mask_area_ratio": row.get("initial_mask_area_ratio", row["initial_mask_area"] / (row["width"] * row["height"])),
                "initial_bbox_xyxy": row.get("initial_bbox_xyxy"),
                "initial_bbox_aspect_ratio": row.get("initial_bbox_aspect_ratio"),
                "overlay": str(out_path),
            }
        )

    contact_sheet = ""
    if tiles:
        sheet = np.vstack(tiles)
        contact_sheet_path = output_dir / "contact_sheet.png"
        if not cv2.imwrite(str(contact_sheet_path), sheet):
            raise RuntimeError(f"failed to write SA-V review contact sheet: {contact_sheet_path}")
        contact_sheet = str(contact_sheet_path)

    summary = {
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "contact_sheet": contact_sheet,
        "samples": samples,
    }
    (output_dir / "review_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def overlay_initial_mask(frame_bgr: np.ndarray, mask: np.ndarray, row: dict[str, Any], alpha: float = 0.45) -> np.ndarray:
    overlay = frame_bgr.copy()
    if mask.shape != overlay.shape[:2]:
        mask = cv2.resize(mask.astype("uint8"), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay[mask] = (overlay[mask] * (1.0 - alpha) + np.asarray([40, 220, 60]) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (40, 40, 255), 2)
    point = row["point"]
    cv2.circle(overlay, (int(round(point[0])), int(round(point[1]))), 7, (255, 255, 255), -1)
    cv2.circle(overlay, (int(round(point[0])), int(round(point[1]))), 7, (0, 0, 0), 2)
    area_ratio = row.get("initial_mask_area_ratio", row["initial_mask_area"] / (row["width"] * row["height"]))
    bbox = row.get("initial_bbox_xyxy")
    if bbox:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 80, 30), 2)
    label = (
        f"{row['video_id']} obj={row['object_id']} "
        f"area={float(area_ratio) * 100:.2f}% point=({point[0]:.1f},{point[1]:.1f})"
    )
    cv2.putText(overlay, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def _resize_tile(frame: np.ndarray, width: int) -> np.ndarray:
    scale = width / frame.shape[1]
    height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write SA-V fixed-sample review overlays for selected GT objects.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
