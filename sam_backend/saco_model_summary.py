from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


FIELDS = [
    "layer",
    "run_id",
    "model_id",
    "backend",
    "mode",
    "status",
    "sources",
    "frames",
    "mean_iou",
    "mask_ap_50_95",
    "mask_f1_50",
    "mask_f1_75",
    "presence_accuracy",
    "mean_model_latency_ms",
    "p95_model_latency_ms",
    "mean_callback_total_ms",
    "mean_end_to_end_ms",
    "p95_end_to_end_ms",
    "end_to_end_fps",
    "mean_tracking_fps",
    "mean_mask_count",
    "cuda_peak_allocated_mb",
    "cuda_peak_reserved_mb",
    "params_total",
    "params_total_m",
    "weight_total_bytes",
    "weight_total_mb",
    "overlay_count",
    "summary_csv",
    "source_csv",
    "message",
]


def main() -> None:
    args = parse_args()
    rows = collect_model_summary(args)
    if not rows:
        raise RuntimeError("no SA-Co offline or ROS summaries were found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


def collect_model_summary(args: argparse.Namespace) -> list[dict[str, Any]]:
    offline_root = args.offline_root or _latest_dir(args.offline_base)
    ros_root = args.ros_root or _latest_dir(args.ros_base)
    rows: list[dict[str, Any]] = []
    if offline_root and offline_root.exists():
        rows.extend(collect_offline_rows(offline_root))
    if not getattr(args, "offline_only", False) and ros_root and ros_root.exists():
        rows.extend(collect_ros_rows(ros_root))
    if args.models:
        selected = set(args.models)
        rows = [row for row in rows if row.get("model_id") in selected]
    return rows


def collect_offline_rows(root: Path) -> list[dict[str, Any]]:
    suite_status = _offline_status_by_model(root / "saco_stream_suite_summary.csv")
    rows = []
    for path in sorted(root.glob("*/frames_summary.csv")):
        summary_rows = _read_csv(path)
        if not summary_rows:
            continue
        item = summary_rows[0]
        model_id = item.get("model_id") or path.parent.name
        status = suite_status.get(model_id, {})
        rows.append(
            {
                "layer": "offline",
                "run_id": root.name,
                "model_id": model_id,
                "backend": item.get("backend", ""),
                "mode": item.get("stream_mode", ""),
                "status": status.get("status", "ok"),
                "sources": item.get("sources", ""),
                "frames": item.get("frames", ""),
                "mean_iou": item.get("mean_iou", ""),
                "mask_ap_50_95": item.get("mask_ap_50_95", ""),
                "mask_f1_50": item.get("mask_f1_50", ""),
                "mask_f1_75": item.get("mask_f1_75", ""),
                "presence_accuracy": item.get("presence_accuracy", ""),
                "mean_model_latency_ms": item.get("mean_latency_ms", ""),
                "p95_model_latency_ms": item.get("p95_latency_ms", ""),
                "mean_callback_total_ms": item.get("mean_callback_total_ms", ""),
                "mean_end_to_end_ms": item.get("mean_end_to_end_ms", ""),
                "p95_end_to_end_ms": item.get("p95_end_to_end_ms", ""),
                "end_to_end_fps": item.get("effective_fps", ""),
                "cuda_peak_allocated_mb": item.get("cuda_peak_allocated_mb", ""),
                "cuda_peak_reserved_mb": item.get("cuda_peak_reserved_mb", ""),
                "params_total": item.get("params_total", ""),
                "params_total_m": _millions(item.get("params_total", "")),
                "weight_total_bytes": item.get("weight_total_bytes", ""),
                "weight_total_mb": _mib(item.get("weight_total_bytes", "")),
                "overlay_count": item.get("overlay_video_count", ""),
                "summary_csv": str(path),
                "source_csv": str(path.with_name("frames.csv")),
                "message": status.get("message", ""),
            }
        )
    for model_id, status in suite_status.items():
        if status.get("status") in {"ok", ""}:
            continue
        if any(row["model_id"] == model_id for row in rows):
            continue
        rows.append(
            {
                "layer": "offline",
                "run_id": root.name,
                "model_id": model_id,
                "backend": status.get("backend", ""),
                "mode": status.get("stream_mode", ""),
                "status": status.get("status", ""),
                "summary_csv": status.get("summary_json", ""),
                "source_csv": status.get("csv", ""),
                "message": status.get("message", ""),
            }
        )
    return rows


def collect_ros_rows(root: Path) -> list[dict[str, Any]]:
    ros_status = _ros_status_by_model(root / "ros_saco_stream_summary.csv")
    rows = []
    for path in sorted(root.glob("*/summary.csv")):
        summary_rows = _read_csv(path)
        if not summary_rows:
            continue
        item = summary_rows[0]
        model_id = path.parent.name
        status = ros_status.get(model_id, {})
        rows.append(
            {
                "layer": "ros_video_stream",
                "run_id": root.name,
                "model_id": model_id,
                "backend": status.get("backend", ""),
                "mode": "ros_video_stream",
                "status": status.get("status", "ok"),
                "frames": item.get("frames", ""),
                "mean_model_latency_ms": item.get("mean_latency_ms", ""),
                "p95_model_latency_ms": item.get("p95_latency_ms", ""),
                "mean_callback_total_ms": item.get("mean_callback_total_ms", ""),
                "mean_end_to_end_ms": item.get("mean_end_to_end_ms", ""),
                "p95_end_to_end_ms": item.get("p95_end_to_end_ms", ""),
                "end_to_end_fps": item.get("mean_end_to_end_fps", ""),
                "mean_tracking_fps": item.get("mean_tracking_fps", ""),
                "mean_mask_count": item.get("mean_mask_count", ""),
                "params_total": item.get("params_total", ""),
                "params_total_m": _millions(item.get("params_total", "")),
                "weight_total_bytes": item.get("weight_total_bytes", ""),
                "weight_total_mb": _mib(item.get("weight_total_bytes", "")),
                "overlay_count": 1 if status.get("overlay_video") else "",
                "summary_csv": str(path),
                "source_csv": str(path.with_name("results.csv")),
                "message": status.get("message", ""),
            }
        )
    for model_id, status in ros_status.items():
        if status.get("status") in {"ok", ""}:
            continue
        if any(row["model_id"] == model_id for row in rows):
            continue
        rows.append(
            {
                "layer": "ros_video_stream",
                "run_id": root.name,
                "model_id": model_id,
                "mode": "ros_video_stream",
                "status": status.get("status", ""),
                "source_csv": status.get("result_csv", ""),
                "summary_csv": status.get("summary_csv", ""),
                "message": status.get("message", ""),
            }
        )
    return rows


def _offline_status_by_model(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {row.get("model_id", ""): row for row in _read_csv(path) if row.get("model_id")}


def _ros_status_by_model(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {row.get("model_id", ""): row for row in _read_csv(path) if row.get("model_id")}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _latest_dir(base: Path) -> Path | None:
    if not base.exists():
        return None
    dirs = [path for path in base.iterdir() if path.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda path: path.stat().st_mtime)


def _number(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _millions(value: object) -> float | str:
    number = _number(value)
    return number / 1_000_000.0 if number is not None else ""


def _mib(value: object) -> float | str:
    number = _number(value)
    return number / (1024.0 * 1024.0) if number is not None else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a model-wise SA-Co summary CSV from offline and ROS Thor runs.")
    parser.add_argument("--offline-root", type=Path, help="Specific results/thor/saco_video_image_per_frame/<run_id> directory.")
    parser.add_argument("--ros-root", type=Path, help="Specific results/thor/ros_saco_stream/<run_id> directory.")
    parser.add_argument("--offline-base", type=Path, default=Path("results/thor/saco_video_image_per_frame"))
    parser.add_argument("--ros-base", type=Path, default=Path("results/thor/ros_saco_stream"))
    parser.add_argument("--offline-only", action="store_true", help="Only collect offline SA-Co summaries.")
    parser.add_argument("--models", nargs="*", help="Optional model IDs to keep in the summary.")
    parser.add_argument("--output", type=Path, default=Path("results/thor/saco_model_wise_summary.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
