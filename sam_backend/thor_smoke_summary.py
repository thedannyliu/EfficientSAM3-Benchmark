from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDNAMES = [
    "task",
    "dataset",
    "suite",
    "model_id",
    "backend",
    "prompt_mode",
    "status",
    "samples",
    "videos",
    "rows",
    "mIoU",
    "AP",
    "AP50",
    "AP75",
    "J&F",
    "J",
    "F",
    "mean_total_ms",
    "effective_fps",
    "mean_iou",
    "mean_effective_fps",
    "elapsed_sec",
    "sec_per_video",
    "source_csv",
]


def main() -> None:
    args = parse_args()
    output = write_smoke_summary(args.root, args.output)
    print(output)


def write_smoke_summary(root: Path, output: Path) -> Path:
    rows: list[dict[str, object]] = []
    rows.extend(_sam2d_rows(root / "sav_sam2d" / "sam2_stage1" / "benchmark_summary.csv", "sav", "sam2d_sam2_stage1"))
    rows.extend(_sam2d_rows(root / "sav_sam2d" / "edgetam" / "benchmark_summary.csv", "sav", "sam2d_edgetam"))
    rows.extend(_sam2d_rows(root / "sa1b_sam2d" / "sam2_stage1" / "benchmark_summary.csv", "sa1b", "sam2d_sam2_stage1"))
    rows.extend(_sam2d_rows(root / "sa1b_sam2d" / "edgetam" / "benchmark_summary.csv", "sa1b", "sam2d_edgetam"))
    rows.extend(_coco_suite_rows(root / "sa1b_sam_family" / "coco_suite_model_summary.csv", "sa1b", "sam_family"))
    rows.extend(_coco_suite_rows(root / "sav_image_sam_family" / "coco_suite_model_summary.csv", "sav", "sam_family_image_box"))
    rows.extend(_sav_video_rows(root / "sav_efficienttam" / "sav_video_suite_summary.csv", "sav", "efficienttam"))
    if not rows:
        raise RuntimeError(f"no smoke summary inputs found under {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})
    return output


def _sam2d_rows(path: Path, dataset: str, suite: str) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    for row in _read_csv(path):
        mode = row.get("mode", "")
        model_id = row.get("model", "")
        prompt = row.get("prompt", "")
        base = {
            "task": "video_tracking" if mode == "video_tracking" else "image_segmentation",
            "dataset": dataset,
            "suite": suite,
            "model_id": model_id,
            "prompt_mode": prompt,
            "status": row.get("status", ""),
            "samples": row.get("num_objects") or row.get("num_images", ""),
            "videos": row.get("videos", ""),
            "source_csv": str(path),
        }
        if mode == "video_tracking":
            base.update(
                {
                    "J&F": row.get("J&F", ""),
                    "J": row.get("J", ""),
                    "F": row.get("F", ""),
                    "elapsed_sec": row.get("elapsed_sec", ""),
                    "sec_per_video": row.get("sec_per_video", ""),
                }
            )
        else:
            base.update(
                {
                    "mIoU": row.get("mIoU", ""),
                    "AP": row.get("AP", ""),
                    "AP50": row.get("AP50", ""),
                    "AP75": row.get("AP75", ""),
                    "mean_total_ms": _seconds_to_ms(row.get("mean_total_object_seconds", "")),
                }
            )
        rows.append(base)
    return rows


def _coco_suite_rows(path: Path, dataset: str, suite: str) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    for row in _read_csv(path):
        rows.append(
            {
                "task": "image_segmentation",
                "dataset": dataset,
                "suite": suite,
                "model_id": row.get("model_id", ""),
                "backend": row.get("backend", ""),
                "prompt_mode": row.get("prompt_mode", ""),
                "status": "ok",
                "samples": row.get("samples", ""),
                "rows": row.get("rows", ""),
                "mIoU": row.get("miou_best", ""),
                "AP": row.get("AP", ""),
                "AP50": row.get("AP50", ""),
                "AP75": row.get("AP75", ""),
                "mean_total_ms": row.get("mean_total_ms", ""),
                "effective_fps": row.get("effective_fps", ""),
                "source_csv": str(path),
            }
        )
    return rows


def _sav_video_rows(path: Path, dataset: str, suite: str) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    for row in _read_csv(path):
        rows.append(
            {
                "task": "video_tracking",
                "dataset": dataset,
                "suite": suite,
                "model_id": row.get("model_id", ""),
                "backend": row.get("backend", ""),
                "status": "ok",
                "videos": row.get("videos", ""),
                "mean_iou": row.get("mean_iou", ""),
                "mean_effective_fps": row.get("mean_effective_fps", ""),
                "source_csv": str(path),
            }
        )
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _seconds_to_ms(value: object) -> str:
    if value in ("", None):
        return ""
    try:
        return str(float(value) * 1000.0)
    except (TypeError, ValueError):
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Thor formal smoke benchmark summaries into one CSV.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
