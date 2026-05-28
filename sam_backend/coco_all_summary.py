from __future__ import annotations

import argparse
import csv
from pathlib import Path


MODEL_SUMMARY_FIELDS = [
    "suite",
    "model_id",
    "family",
    "backend",
    "prompt_mode",
    "samples",
    "rows",
    "effective_fps",
    "mean_total_ms",
    "miou_best",
    "miou_merged",
    "mean_best_box_iou",
    "mean_target_detection_count",
    "mean_cuda_peak_allocated_mb",
    "mean_cuda_peak_reserved_mb",
    "params_total_m",
    "params_backbone_m",
    "params_image_encoder_m",
    "params_text_encoder_m",
    "params_transformer_m",
    "params_geometry_encoder_m",
    "params_segmentation_head_m",
    "params_prompt_encoder_m",
    "params_mask_decoder_m",
    "params_detector_m",
    "params_memory_encoder_m",
    "params_memory_attention_m",
    "params_yolo_backbone_m",
    "params_yolo_neck_m",
    "params_yolo_head_m",
    "weight_total_mb",
    "weight_backbone_mb",
    "weight_image_encoder_mb",
    "weight_text_encoder_mb",
    "weight_transformer_mb",
    "weight_geometry_encoder_mb",
    "weight_segmentation_head_mb",
    "weight_prompt_encoder_mb",
    "weight_mask_decoder_mb",
    "weight_detector_mb",
    "weight_memory_encoder_mb",
    "weight_memory_attention_mb",
    "weight_yolo_backbone_mb",
    "weight_yolo_neck_mb",
    "weight_yolo_head_mb",
    "checkpoint_file_mb",
    "source_csv",
]


def main() -> None:
    args = parse_args()
    output = write_coco_all_summary(args.sam_dir, args.yolo_dir, args.output)
    print(output)


def write_coco_all_summary(sam_dir: Path, yolo_dir: Path, output: Path) -> Path:
    rows = collect_rows(sam_dir, yolo_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MODEL_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def collect_rows(sam_dir: Path, yolo_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sam_summary = sam_dir / "coco_suite_component_summary.csv"
    if sam_summary.exists():
        rows.extend(_normalize_sam_row(row, sam_summary) for row in _read_csv(sam_summary))
    yolo_summary = yolo_dir / "yolo_coco_model_summary.csv"
    if yolo_summary.exists():
        rows.extend(_normalize_yolo_row(row, yolo_summary) for row in _read_csv(yolo_summary))
    return rows


def _normalize_sam_row(row: dict[str, str], source_csv: Path) -> dict[str, object]:
    out = _select_common(row)
    out.update(
        {
            "suite": "sam_coco",
            "family": row.get("backend", ""),
            "backend": row.get("backend", ""),
            "prompt_mode": row.get("prompt_mode", ""),
            "source_csv": str(source_csv),
        }
    )
    return out


def _normalize_yolo_row(row: dict[str, str], source_csv: Path) -> dict[str, object]:
    out = _select_common(row)
    out.update(
        {
            "suite": "yolo_coco",
            "family": row.get("family", ""),
            "backend": row.get("family", ""),
            "prompt_mode": "text" if row.get("family") == "yoloe-seg" else "closed_set_class",
            "source_csv": str(source_csv),
        }
    )
    return out


def _select_common(row: dict[str, str]) -> dict[str, object]:
    return {field: row.get(field, "") for field in MODEL_SUMMARY_FIELDS}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge SAM-family and YOLO COCO summaries into one model table.")
    parser.add_argument("--sam-dir", type=Path, required=True)
    parser.add_argument("--yolo-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
