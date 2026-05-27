from __future__ import annotations

import argparse
import csv
from pathlib import Path


COMPONENT_FIELDS = [
    "image_encoder_ms",
    "prompt_encoder_ms",
    "mask_decoder_ms",
    "transformer_ms",
    "geometry_encoder_ms",
    "segmentation_head_ms",
    "grounding_ms",
    "detector_ms",
    "memory_attention_ms",
    "memory_encoder_ms",
]

PARAM_WEIGHT_FIELDS = [
    "params_total",
    "params_backbone",
    "params_image_encoder",
    "params_text_encoder",
    "params_transformer",
    "params_geometry_encoder",
    "params_segmentation_head",
    "params_prompt_encoder",
    "params_mask_decoder",
    "params_detector",
    "params_memory_attention",
    "params_memory_encoder",
    "weight_total_bytes",
    "weight_backbone_bytes",
    "weight_image_encoder_bytes",
    "weight_text_encoder_bytes",
    "weight_transformer_bytes",
    "weight_geometry_encoder_bytes",
    "weight_segmentation_head_bytes",
    "weight_prompt_encoder_bytes",
    "weight_mask_decoder_bytes",
    "weight_detector_bytes",
    "weight_memory_attention_bytes",
    "weight_memory_encoder_bytes",
]


def main() -> None:
    args = parse_args()
    path = write_sav_video_report(args.root, args.output)
    print(path)


def write_sav_video_report(root: Path, output: Path | None = None) -> Path:
    rows = []
    for summary_csv in sorted(root.glob("*/frames_summary.csv")):
        with summary_csv.open(newline="", encoding="utf-8") as f:
            video_rows = list(csv.DictReader(f))
        if not video_rows:
            continue
        first = video_rows[0]
        total_frames = sum(int(float(row.get("frames_tracked") or 0)) for row in video_rows)
        total_gt_frames = sum(int(float(row.get("gt_frames_evaluated") or 0)) for row in video_rows)
        row = {
            "model_id": first.get("model_id", summary_csv.parent.name),
            "backend": first.get("backend", ""),
            "videos": len(video_rows),
            "frames_tracked": total_frames,
            "gt_frames_evaluated": total_gt_frames,
            "mean_iou": _mean(video_rows, "mean_iou"),
            "mean_effective_fps": _mean(video_rows, "effective_fps"),
            "mean_session_init_ms": _mean(video_rows, "session_init_ms"),
            "mean_add_prompt_ms": _mean(video_rows, "add_prompt_ms"),
            "mean_propagate_total_ms": _mean(video_rows, "propagate_total_ms"),
            "mean_propagate_step_ms": _mean(video_rows, "mean_propagate_step_ms"),
            "mean_cuda_peak_allocated_mb": _mean(video_rows, "cuda_peak_allocated_mb"),
            "mean_cuda_peak_reserved_mb": _mean(video_rows, "cuda_peak_reserved_mb"),
            "overlay_videos": sum(1 for row in video_rows if row.get("overlay_video")),
        }
        for field_name in COMPONENT_FIELDS:
            row[f"mean_{field_name}"] = _mean(video_rows, field_name)
        for field_name in PARAM_WEIGHT_FIELDS:
            row[field_name] = first.get(field_name, "")
        row.update(_readable_param_weight_fields(row))
        rows.append(row)

    if not rows:
        raise RuntimeError(f"no SA-V video summaries found under {root}")

    output = output or root / "sav_video_suite_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output


def _mean(rows: list[dict[str, str]], key: str) -> float | str:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", None):
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    return sum(values) / len(values) if values else ""


def _readable_param_weight_fields(row: dict[str, object]) -> dict[str, float | str]:
    fields: dict[str, float | str] = {}
    for key, value in list(row.items()):
        if key.startswith("params_"):
            fields[f"{key}_m"] = _numeric(value) / 1_000_000.0 if _numeric(value) is not None else ""
        elif key.startswith("weight_") and key.endswith("_bytes"):
            fields[f"{key[:-6]}_mb"] = _numeric(value) / (1024.0 * 1024.0) if _numeric(value) is not None else ""
    return fields


def _numeric(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize per-model SA-V video profiler outputs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
