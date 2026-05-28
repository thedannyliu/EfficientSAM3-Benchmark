from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BYTES_PER_MIB = 1024.0 * 1024.0


@dataclass(frozen=True)
class ModelAsset:
    model_id: str
    component: str
    paths: tuple[str, ...]


MODEL_ASSETS = [
    ModelAsset("sam3", "sam3_checkpoint", ("checkpoints/sam3/sam3.pt",)),
    ModelAsset("sam3", "sam3_config", ("checkpoints/sam3/config.json",)),
    ModelAsset(
        "es3p1_weak_image_weak_text",
        "efficient_sam3_checkpoint",
        ("checkpoints/stage1_sam3p1/efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt",),
    ),
    ModelAsset(
        "es3p1_strong_image_weak_text",
        "efficient_sam3_checkpoint",
        ("checkpoints/stage1_sam3p1/efficient_sam3p1_efficientvit_l_mobileclip_s0_ctx16.pt",),
    ),
    ModelAsset(
        "es3_weak_image_strong_available_text",
        "efficient_sam3_checkpoint",
        ("checkpoints/stage1_all_converted/efficient_sam3_efficientvit-b0_mobileclip_s1.pth",),
    ),
    ModelAsset(
        "es3_strong_image_strong_available_text",
        "efficient_sam3_checkpoint",
        ("checkpoints/stage1_all_converted/efficient_sam3_efficientvit-b2_mobileclip_s1.pth",),
    ),
    ModelAsset("sam2p1_hiera_tiny", "sam2_checkpoint", ("checkpoints/sam2/sam2.1_hiera_tiny.pt",)),
    ModelAsset(
        "efficient_sam2p1_hiera_tiny",
        "efficient_sam2_checkpoint",
        ("checkpoints/efficient-sam2/sam2.1_hiera_tiny.pt",),
    ),
    ModelAsset("efficienttam_ti", "efficienttam_checkpoint", ("checkpoints/efficienttam/efficienttam_ti.pt",)),
    ModelAsset("efficienttam_s", "efficienttam_checkpoint", ("checkpoints/efficienttam/efficienttam_s.pt",)),
    ModelAsset("mobilesam_vit_t", "mobilesam_checkpoint", ("checkpoints/mobilesam/mobile_sam.pt",)),
    ModelAsset("yoloe_26m_seg_edgetam", "yoloe_weights", ("checkpoints/yoloe/yoloe-26m-seg.pt",)),
    ModelAsset("yoloe_26m_seg_edgetam", "edgetam_checkpoint", ("checkpoints/edgetam/edgetam.pt",)),
    ModelAsset("yoloe_26m_seg_edgetam", "mobileclip_asset", ("checkpoints/yoloe/mobileclip2_b.ts", "mobileclip2_b.ts")),
]


LEADING_FIELDS = [
    "task",
    "run_id",
    "model_id",
    "backend",
    "prompt_mode",
    "source_id",
    "status",
    "rows",
    "samples",
    "videos",
    "sources",
    "frames_tracked",
    "gt_frames_evaluated",
    "effective_fps",
    "effective_tracking_fps",
    "mean_effective_fps",
    "mean_effective_tracking_fps",
    "mean_total_ms",
    "first_mask_latency_ms",
    "mean_first_mask_latency_ms",
    "mean_iou",
    "miou_best",
    "miou_merged",
    "reground_count",
    "overlay_videos",
    "overlay_video",
    "summary_csv",
    "source_csv",
    "storage_total_bytes",
    "storage_total_mb",
    "storage_components",
    "storage_missing_components",
]


def main() -> None:
    args = parse_args()
    outputs = write_thor_offline_reports(args.root, args.output_dir, args.project_root)
    for path in outputs:
        print(path)


def write_thor_offline_reports(root: Path, output_dir: Path, project_root: Path = Path(".")) -> list[Path]:
    storage_rows = build_storage_rows(project_root)
    storage_by_model = _storage_by_model(storage_rows)

    coco_rows = [_attach_storage(row, storage_by_model) for row in collect_coco_rows(root)]
    sav_rows = [_attach_storage(row, storage_by_model) for row in collect_sav_rows(root)]
    yoloe_rows = [_attach_storage(row, storage_by_model) for row in collect_yoloe_rows(root)]
    all_rows = coco_rows + sav_rows + yoloe_rows

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        _write_csv(output_dir / "thor_offline_coco_summary.csv", coco_rows),
        _write_csv(output_dir / "thor_offline_sav_summary.csv", sav_rows),
        _write_csv(output_dir / "thor_offline_yoloe_edgetam_summary.csv", yoloe_rows),
        _write_csv(output_dir / "thor_offline_all_summary.csv", all_rows),
        _write_csv(output_dir / "thor_offline_model_storage_components.csv", storage_rows),
    ]
    return [path for path in outputs if path is not None]


def collect_coco_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((root / "coco").glob("*/coco_suite_component_summary.csv")):
        for row in _read_csv(path):
            row = dict(row)
            row.update(
                {
                    "task": "coco",
                    "run_id": path.parent.name,
                    "summary_csv": str(path),
                    "source_csv": str(path.parent / row.get("model_id", "") / "profile.csv"),
                }
            )
            rows.append(row)
    return rows


def collect_sav_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((root / "sav").glob("*/*/frames_summary.csv")):
        video_rows = _read_csv(path)
        if not video_rows:
            continue
        first = video_rows[0]
        row: dict[str, object] = {
            "task": "sav",
            "run_id": path.parents[1].name,
            "model_id": first.get("model_id", path.parent.name),
            "backend": first.get("backend", ""),
            "videos": len(video_rows),
            "frames_tracked": _sum(video_rows, "frames_tracked"),
            "gt_frames_evaluated": _sum(video_rows, "gt_frames_evaluated"),
            "mean_iou": _mean(video_rows, "mean_iou"),
            "mean_effective_fps": _mean(video_rows, "effective_fps"),
            "mean_session_init_ms": _mean(video_rows, "session_init_ms"),
            "mean_add_prompt_ms": _mean(video_rows, "add_prompt_ms"),
            "mean_propagate_total_ms": _mean(video_rows, "propagate_total_ms"),
            "mean_propagate_step_ms": _mean(video_rows, "mean_propagate_step_ms"),
            "mean_cuda_peak_allocated_mb": _mean(video_rows, "cuda_peak_allocated_mb"),
            "mean_cuda_peak_reserved_mb": _mean(video_rows, "cuda_peak_reserved_mb"),
            "overlay_videos": sum(1 for item in video_rows if item.get("overlay_video")),
            "summary_csv": str(path),
            "source_csv": str(path.with_name("frames.csv")),
        }
        _copy_first_matching_prefix(row, first, ("params_", "weight_"))
        _add_readable_param_weight_fields(row)
        rows.append(row)
    return rows


def collect_yoloe_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_dir in ("yoloe_edgetam", "yoloe_edgetam_multi", "yoloe_edgetam_sav"):
        for path in sorted((root / task_dir).glob("*/frames_summary.csv")):
            source_rows = _read_csv(path)
            if not source_rows:
                continue
            first = source_rows[0]
            row: dict[str, object] = {
                "task": task_dir,
                "run_id": path.parent.name,
                "model_id": "yoloe_26m_seg_edgetam",
                "backend": "yoloe+edgetam",
                "sources": len(source_rows),
                "status": _join_unique(item.get("status", "") for item in source_rows),
                "frames_tracked": _sum(source_rows, "frames_tracked"),
                "mean_effective_tracking_fps": _mean(source_rows, "effective_tracking_fps"),
                "mean_first_mask_latency_ms": _mean(source_rows, "first_mask_latency_ms"),
                "mean_edgetam_step_ms": _mean(source_rows, "mean_edgetam_step_ms"),
                "mean_yoloe_validation_ms": _mean(source_rows, "mean_yoloe_validation_ms"),
                "reground_count": _sum(source_rows, "reground_count"),
                "overlay_videos": sum(1 for item in source_rows if item.get("overlay_video")),
                "summary_csv": str(path),
                "source_csv": str(path.with_name("frames.csv")),
            }
            _copy_first_matching_prefix(row, first, ("yoloe_params_", "yoloe_weight_", "edgetam_params_", "edgetam_weight_"))
            _add_yoloe_combined_param_weight_fields(row)
            _add_readable_param_weight_fields(row)
            rows.append(row)
    return rows


def build_storage_rows(project_root: Path) -> list[dict[str, object]]:
    rows = []
    for asset in MODEL_ASSETS:
        selected_path = _first_existing_path(project_root, asset.paths)
        size = selected_path.stat().st_size if selected_path is not None else None
        rows.append(
            {
                "model_id": asset.model_id,
                "component": asset.component,
                "exists": bool(selected_path),
                "path": str(selected_path) if selected_path else "",
                "candidate_paths": ";".join(asset.paths),
                "size_bytes": size if size is not None else "",
                "size_mb": size / BYTES_PER_MIB if size is not None else "",
            }
        )
    return rows


def _attach_storage(row: dict[str, object], storage_by_model: dict[str, dict[str, object]]) -> dict[str, object]:
    out = dict(row)
    storage = storage_by_model.get(str(out.get("model_id", "")), {})
    out.update(storage)
    return out


def _storage_by_model(storage_rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in storage_rows:
        grouped.setdefault(str(row["model_id"]), []).append(row)

    output = {}
    for model_id, rows in grouped.items():
        existing = [row for row in rows if row["exists"]]
        missing = [str(row["component"]) for row in rows if not row["exists"]]
        total = sum(int(row["size_bytes"]) for row in existing)
        summary: dict[str, object] = {
            "storage_total_bytes": total,
            "storage_total_mb": total / BYTES_PER_MIB,
            "storage_components": ";".join(f"{row['component']}={row['size_mb']}MB" for row in existing),
            "storage_missing_components": ";".join(missing),
        }
        for row in rows:
            component = str(row["component"])
            summary[f"storage_{component}_bytes"] = row["size_bytes"]
            summary[f"storage_{component}_mb"] = row["size_mb"]
        output[model_id] = summary
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path | None:
    if not rows:
        return None
    fieldnames = _fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _fieldnames(rows: list[dict[str, object]]) -> list[str]:
    keys = set().union(*(row.keys() for row in rows))
    leading = [field for field in LEADING_FIELDS if field in keys]
    return leading + sorted(keys - set(leading))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _copy_first_matching_prefix(target: dict[str, object], source: dict[str, str], prefixes: tuple[str, ...]) -> None:
    for key, value in source.items():
        if key.startswith(prefixes):
            target[key] = value


def _add_yoloe_combined_param_weight_fields(row: dict[str, object]) -> None:
    yoloe_params = _numeric(row.get("yoloe_params_total"))
    edgetam_params = _numeric(row.get("edgetam_params_total"))
    yoloe_weight = _numeric(row.get("yoloe_weight_total_bytes"))
    edgetam_weight = _numeric(row.get("edgetam_weight_total_bytes"))
    if yoloe_params is not None or edgetam_params is not None:
        row["params_total"] = int((yoloe_params or 0.0) + (edgetam_params or 0.0))
    if yoloe_weight is not None or edgetam_weight is not None:
        row["weight_total_bytes"] = int((yoloe_weight or 0.0) + (edgetam_weight or 0.0))


def _add_readable_param_weight_fields(row: dict[str, object]) -> None:
    for key, value in list(row.items()):
        number = _numeric(value)
        if number is None:
            continue
        if key.startswith("params_"):
            row[f"{key}_m"] = number / 1_000_000.0
        elif key.startswith("weight_") and key.endswith("_bytes"):
            row[f"{key[:-6]}_mb"] = number / BYTES_PER_MIB
        elif key.startswith(("yoloe_params_", "edgetam_params_")):
            row[f"{key}_m"] = number / 1_000_000.0
        elif key.startswith(("yoloe_weight_", "edgetam_weight_")) and key.endswith("_bytes"):
            row[f"{key[:-6]}_mb"] = number / BYTES_PER_MIB


def _first_existing_path(project_root: Path, candidates: tuple[str, ...]) -> Path | None:
    for candidate in candidates:
        path = project_root / candidate
        if path.exists():
            return path
    return None


def _sum(rows: list[dict[str, str]], key: str) -> int:
    return int(sum(_numeric(row.get(key)) or 0.0 for row in rows))


def _mean(rows: list[dict[str, str]], key: str) -> float | str:
    values = [_numeric(row.get(key)) for row in rows]
    numeric = [value for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else ""


def _numeric(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _join_unique(values: Iterable[str]) -> str:
    return ";".join(sorted({value for value in values if value}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize completed Thor offline benchmark outputs by task and model.")
    parser.add_argument("--root", type=Path, default=Path("results/thor/offline"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/thor/offline/reports"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser.parse_args()


if __name__ == "__main__":
    main()
