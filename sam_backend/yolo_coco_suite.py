from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


COMPONENT_FIELDS = [
    "set_classes_ms",
    "predict_ms",
    "postprocess_ms",
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
    "params_memory_encoder",
    "params_memory_attention",
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
    "weight_memory_encoder_bytes",
    "weight_memory_attention_bytes",
    "params_yolo_backbone",
    "params_yolo_neck",
    "params_yolo_head",
    "weight_yolo_backbone_bytes",
    "weight_yolo_neck_bytes",
    "weight_yolo_head_bytes",
    "checkpoint_file_bytes",
]


@dataclass(frozen=True)
class YoloCocoRun:
    model_id: str
    family: str
    weights: str
    presets: tuple[str, ...]
    extra_args: tuple[str, ...] = field(default_factory=tuple)


RUNS = [
    YoloCocoRun("yoloe_11s_seg", "yoloe-seg", "yoloe-11s-seg.pt", ("small", "all")),
    YoloCocoRun("yoloe_11m_seg", "yoloe-seg", "yoloe-11m-seg.pt", ("all",)),
    YoloCocoRun("yoloe_11l_seg", "yoloe-seg", "yoloe-11l-seg.pt", ("all",)),
    YoloCocoRun("yoloe_v8s_seg", "yoloe-seg", "yoloe-v8s-seg.pt", ("small", "all")),
    YoloCocoRun("yoloe_v8m_seg", "yoloe-seg", "yoloe-v8m-seg.pt", ("all",)),
    YoloCocoRun("yoloe_v8l_seg", "yoloe-seg", "yoloe-v8l-seg.pt", ("all",)),
    YoloCocoRun("yoloe_26n_seg", "yoloe-seg", "yoloe-26n-seg.pt", ("quick", "small", "all")),
    YoloCocoRun("yoloe_26s_seg", "yoloe-seg", "yoloe-26s-seg.pt", ("small", "all")),
    YoloCocoRun("yoloe_26m_seg", "yoloe-seg", "checkpoints/yoloe/yoloe-26m-seg.pt", ("all",)),
    YoloCocoRun("yoloe_26l_seg", "yoloe-seg", "yoloe-26l-seg.pt", ("all",)),
    YoloCocoRun("yoloe_26x_seg", "yoloe-seg", "yoloe-26x-seg.pt", ("all",)),
    YoloCocoRun("yolo11n_seg", "yolo-seg", "yolo11n-seg.pt", ("quick", "small", "all")),
    YoloCocoRun("yolo11s_seg", "yolo-seg", "yolo11s-seg.pt", ("small", "all")),
    YoloCocoRun("yolo11m_seg", "yolo-seg", "yolo11m-seg.pt", ("small", "all")),
    YoloCocoRun("yolo11l_seg", "yolo-seg", "yolo11l-seg.pt", ("all",)),
    YoloCocoRun("yolo11x_seg", "yolo-seg", "yolo11x-seg.pt", ("all",)),
]


def main() -> None:
    args = parse_args()
    results = run_suite(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "yolo_coco_suite_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_id", "family", "weights", "status", "summary_json", "csv", "message"])
        writer.writeheader()
        writer.writerows(results)
    print(summary_path)
    component_summary_path = write_component_summary(args.output_dir)
    if component_summary_path:
        print(component_summary_path)


def run_suite(args: argparse.Namespace) -> list[dict[str, str]]:
    selected = {name for name in args.models} if args.models else None
    runs = [run for run in RUNS if args.preset in run.presets and (selected is None or run.model_id in selected)]
    if selected:
        known = {run.model_id for run in RUNS}
        missing = selected - known
        if missing:
            raise ValueError(f"unknown model ids: {', '.join(sorted(missing))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overlay_dir:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for run in runs:
        run_dir = args.output_dir / run.model_id
        csv_path = run_dir / "profile.csv"
        summary_path = run_dir / "summary.json"
        cmd = _build_cmd(args, run, csv_path, summary_path)
        if args.dry_run:
            results.append(_result(run, "dry-run", summary_path, csv_path, " ".join(cmd)))
            continue
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            results.append(_result(run, "failed", summary_path, csv_path, f"exit {exc.returncode}: {' '.join(cmd)}"))
            continue
        results.append(_result(run, "ok", summary_path, csv_path, ""))
    return results


def _build_cmd(args: argparse.Namespace, run: YoloCocoRun, csv_path: Path, summary_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "sam_backend.profile_yolo_coco",
        "--model-id",
        run.model_id,
        "--family",
        run.family,
        "--weights",
        run.weights,
        "--device",
        args.device,
        "--manifest",
        str(args.manifest),
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--max-det",
        str(args.max_det),
    ]
    if args.limit and args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])
    cmd.extend(
        [
            "--eval-mode",
            args.eval_mode,
            "--csv-output",
            str(csv_path),
            "--summary-output",
            str(summary_path),
        ]
    )
    if args.agnostic_nms is not None:
        cmd.append("--agnostic-nms" if args.agnostic_nms else "--no-agnostic-nms")
    if args.overlay_dir:
        cmd.extend(["--overlay-dir", str(args.overlay_dir / run.model_id)])
    cmd.extend(run.extra_args)
    return cmd


def weight_names_for_preset(preset: str) -> list[str]:
    return [run.weights for run in RUNS if preset in run.presets]


def _result(run: YoloCocoRun, status: str, summary_path: Path, csv_path: Path, message: str) -> dict[str, str]:
    return {
        "model_id": run.model_id,
        "family": run.family,
        "weights": run.weights,
        "status": status,
        "summary_json": str(summary_path),
        "csv": str(csv_path),
        "message": message,
    }


def write_component_summary(output_dir: Path) -> Path | None:
    rows = []
    for profile_csv in sorted(output_dir.glob("*/profile.csv")):
        with profile_csv.open(newline="", encoding="utf-8") as f:
            profile_rows = list(csv.DictReader(f))
        if not profile_rows:
            continue
        first = profile_rows[0]
        total_ms = _mean(profile_rows, "total_ms")
        row = {
            "model_id": first.get("model_id", profile_csv.parent.name),
            "family": first.get("family", ""),
            "weights": first.get("weights", ""),
            "rows": len(profile_rows),
            "samples": len({row.get("sample_id", "") for row in profile_rows}),
            "mean_total_ms": total_ms,
            "effective_fps": 1000.0 / total_ms if isinstance(total_ms, float) and total_ms > 0 else "",
            "miou_best": _mean(profile_rows, "best_iou"),
            "miou_merged": _mean(profile_rows, "merged_iou"),
            "mean_best_box_iou": _mean(profile_rows, "best_box_iou"),
            "mean_target_detection_count": _mean(profile_rows, "target_detection_count"),
            "mean_cuda_peak_allocated_mb": _mean(profile_rows, "cuda_peak_allocated_mb"),
            "mean_cuda_peak_reserved_mb": _mean(profile_rows, "cuda_peak_reserved_mb"),
            "component_note": first.get("component_note", ""),
            "yolo_backbone_layers": first.get("yolo_backbone_layers", ""),
            "yolo_neck_layers": first.get("yolo_neck_layers", ""),
            "yolo_head_layers": first.get("yolo_head_layers", ""),
        }
        for field_name in COMPONENT_FIELDS:
            row[f"mean_{field_name}"] = _mean(profile_rows, field_name)
        for field_name in PARAM_WEIGHT_FIELDS:
            row[field_name] = first.get(field_name, "")
        row.update(_readable_param_weight_fields(row))
        rows.append(row)

    if not rows:
        return None

    path = output_dir / "yolo_coco_component_summary.csv"
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


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
        elif key == "checkpoint_file_bytes":
            fields["checkpoint_file_mb"] = _numeric(value) / (1024.0 * 1024.0) if _numeric(value) is not None else ""
    return fields


def _numeric(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the YOLO COCO profiling suite.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1, help="Profile only the first N manifest rows; 0 means all rows.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preset", choices=["quick", "small", "all"], default="quick")
    parser.add_argument("--models", nargs="*", help="Optional subset of model ids to run.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--agnostic-nms", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--eval-mode", choices=["gt", "overlay", "both", "profile"], default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("results/yolo_coco_suite"))
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Write the command matrix without running models.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
