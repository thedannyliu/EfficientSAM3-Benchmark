from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .backends import BackendConfig, Prompt, create_backend
from .overlay import overlay_prediction, to_numpy
from .profiling import component_timer, cuda_memory_mb, parameter_counts


COMPONENT_FIELDS = [
    "image_encoder_ms",
    "text_encoder_ms",
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


FIELDNAMES = [
    "model_id",
    "backend",
    "video_id",
    "object_id",
    "frame_index",
    "prompt_mode",
    "prompt",
    "point_x",
    "point_y",
    "image",
    "mask",
    "width",
    "height",
    "mask_count",
    "box_count",
    "score_max",
    "iou",
    "total_ms",
    *COMPONENT_FIELDS,
    "other_ms",
    "cuda_allocated_mb",
    "cuda_reserved_mb",
    "cuda_peak_allocated_mb",
    "cuda_peak_reserved_mb",
    "overlay",
]


SUMMARY_FIELDS = [
    "model_id",
    "backend",
    "video_id",
    "object_id",
    "prompt_mode",
    "frames_evaluated",
    "mean_iou",
    "mean_total_ms",
    "effective_fps",
    *[f"mean_{field}" for field in COMPONENT_FIELDS],
    "mean_cuda_peak_allocated_mb",
    "mean_cuda_peak_reserved_mb",
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
    summary = profile_sav_frames(args)
    print(json.dumps(summary, indent=2))
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def profile_sav_frames(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_manifest(args.manifest)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("manifest did not provide any SA-V rows")

    prompt_modes = _prompt_modes(args.prompt_mode)
    if args.backend in {"sam2", "efficient-sam2", "efficienttam", "mobilesam"} and "text" in prompt_modes:
        raise ValueError(f"{args.backend} supports point prompts in this frame benchmark, not text prompts")

    backend = create_backend(
        BackendConfig(
            backend=args.backend,
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            backbone_type=args.backbone_type,
            model_name=args.model_name,
            text_encoder_type=args.text_encoder_type,
            text_encoder_context_length=args.text_encoder_context_length,
            text_encoder_pos_embed_table_size=args.text_encoder_pos_embed_table_size,
            interpolate_pos_embed=args.interpolate_pos_embed,
            enable_inst_interactivity="point" in prompt_modes,
            model_config=args.model_config,
            external_repo=args.external_repo,
            mobile_sam_model_type=args.mobile_sam_model_type,
        )
    )
    torch_module = getattr(backend, "torch", None)
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()
    params = parameter_counts(getattr(backend, "model", None))

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    if args.overlay_dir:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, Any]] = []
    with args.csv_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for item in rows:
            for frame_index, mask_path in _iter_annotation_frames(item, args.max_frames, args.frame_stride):
                frame_path = Path(item["frames_dir"]) / f"{frame_index:05d}.jpg"
                frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    raise RuntimeError(f"failed to read SA-V frame: {frame_path}")
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                gt_mask = _read_mask(mask_path)
                if gt_mask is None:
                    raise RuntimeError(f"failed to read SA-V mask: {mask_path}")

                for prompt_mode in prompt_modes:
                    prompt, point = _build_prompt(item, prompt_mode, gt_mask)
                    profile = {}
                    start = perf_counter()
                    if torch_module is not None:
                        with component_timer(getattr(backend, "model", None), torch_module) as profile:
                            prediction = backend.predict(frame_rgb, prompt)
                    else:
                        prediction = backend.predict(frame_rgb, prompt)
                    total_ms = (perf_counter() - start) * 1000.0
                    component_total = sum(profile.values())
                    memory = cuda_memory_mb(torch_module) if torch_module is not None else cuda_memory_mb(None)
                    iou = _prediction_iou(prediction.masks, gt_mask)
                    scores = to_numpy(prediction.scores)
                    overlay_path = (
                        _write_overlay(args.overlay_dir, item, frame_index, prompt_mode, frame_rgb, prediction)
                        if args.overlay_dir
                        else ""
                    )
                    row = {
                        "model_id": args.model_id,
                        "backend": args.backend,
                        "video_id": item["video_id"],
                        "object_id": item["object_id"],
                        "frame_index": frame_index,
                        "prompt_mode": prompt_mode,
                        "prompt": prompt.text or "",
                        "point_x": point[0] if point else "",
                        "point_y": point[1] if point else "",
                        "image": str(frame_path),
                        "mask": str(mask_path),
                        "width": frame_rgb.shape[1],
                        "height": frame_rgb.shape[0],
                        "mask_count": _safe_len(prediction.masks),
                        "box_count": _safe_len(prediction.boxes),
                        "score_max": float(scores.max()) if scores.size else "",
                        "iou": iou,
                        "total_ms": total_ms,
                        **{field: profile.get(field, 0.0) for field in COMPONENT_FIELDS},
                        "other_ms": max(0.0, total_ms - component_total),
                        "overlay": str(overlay_path),
                        **memory,
                    }
                    writer.writerow(row)
                    output_rows.append(row)

    summary_rows = _summary_rows(args, output_rows, params)
    summary_csv = args.csv_output.with_name(args.csv_output.stem + "_summary.csv")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    return {
        "model_id": args.model_id,
        "backend": args.backend,
        "manifest": str(args.manifest),
        "csv": str(args.csv_output),
        "summary_csv": str(summary_csv),
        "videos": len({row["video_id"] for row in output_rows}),
        "rows": len(output_rows),
        "prompt_modes": sorted({row["prompt_mode"] for row in output_rows}),
        "mean_iou": _mean([row["iou"] for row in output_rows]),
        "mean_total_ms": _mean([row["total_ms"] for row in output_rows]),
        "effective_fps": _fps(_mean([row["total_ms"] for row in output_rows])),
        "params_and_weights": params,
    }


def _iter_annotation_frames(item: dict[str, Any], max_frames: int, frame_stride: int) -> list[tuple[int, Path]]:
    mask_dir = Path(item["annotations_dir"]) / str(item["object_id"])
    paths = sorted(mask_dir.glob("*.png"))
    if frame_stride > 1:
        paths = paths[::frame_stride]
    if max_frames > 0:
        paths = paths[:max_frames]
    return [(int(path.stem), path) for path in paths]


def _build_prompt(item: dict[str, Any], prompt_mode: str, gt_mask: np.ndarray) -> tuple[Prompt, tuple[float, float] | None]:
    if prompt_mode == "text":
        text_prompt = str(item.get("text_prompt", "")).strip()
        if not text_prompt:
            raise ValueError(
                f"text prompt missing for sample {item.get('sample_id', item.get('video_id', ''))}; "
                "merge manual prompts with sam-sav-text-prompts apply first"
            )
        return Prompt(text=text_prompt), None
    if prompt_mode == "point":
        point = _mask_centroid(gt_mask)
        return Prompt(points=[point], labels=[int(item.get("point_label", 1))]), point
    raise ValueError(f"unknown SA-V frame prompt mode: {prompt_mode}")


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("cannot build point prompt from an empty mask")
    return float(xs.mean()), float(ys.mean())


def _prediction_iou(masks: Any, gt_mask: np.ndarray) -> float:
    values = to_numpy(masks)
    if values.size == 0:
        return 0.0
    if values.ndim == 4:
        values = values[:, 0]
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3:
        return 0.0
    ious = []
    for mask in values:
        pred = mask.astype(bool)
        if pred.shape != gt_mask.shape:
            pred = cv2.resize(pred.astype("uint8"), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        ious.append(_mask_iou(pred, gt_mask))
    return max(ious) if ious else 0.0


def _mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(intersection / union) if union else 0.0


def _read_mask(path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask > 0


def _write_overlay(
    overlay_dir: Path | None,
    item: dict[str, Any],
    frame_index: int,
    prompt_mode: str,
    frame_rgb: np.ndarray,
    prediction: Any,
) -> Path:
    assert overlay_dir is not None
    out_dir = overlay_dir / item["video_id"] / str(item["object_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{frame_index:05d}-{prompt_mode}.png"
    overlay_rgb = overlay_prediction(frame_rgb, prediction.masks, prediction.boxes, prediction.scores)
    ok = cv2.imwrite(str(path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"failed to write overlay image: {path}")
    return path


def _summary_rows(args: argparse.Namespace, rows: list[dict[str, Any]], params: dict[str, int]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row["video_id"]), str(row["object_id"]), str(row["prompt_mode"])), []).append(row)
    out = []
    for (video_id, object_id, prompt_mode), group in sorted(by_key.items()):
        mean_total = _mean([row["total_ms"] for row in group])
        out.append(
            {
                "model_id": args.model_id,
                "backend": args.backend,
                "video_id": video_id,
                "object_id": object_id,
                "prompt_mode": prompt_mode,
                "frames_evaluated": len(group),
                "mean_iou": _mean([row["iou"] for row in group]),
                "mean_total_ms": mean_total,
                "effective_fps": _fps(mean_total),
                **{f"mean_{field}": _mean([row[field] for row in group]) for field in COMPONENT_FIELDS},
                "mean_cuda_peak_allocated_mb": _mean([row["cuda_peak_allocated_mb"] for row in group]),
                "mean_cuda_peak_reserved_mb": _mean([row["cuda_peak_reserved_mb"] for row in group]),
                **params,
            }
        )
    return out


def _prompt_modes(value: str) -> list[str]:
    if value == "both":
        return ["text", "point"]
    return [value]


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _mean(values: list[Any]) -> float | str:
    numeric = [float(value) for value in values if value not in ("", None)]
    return mean(numeric) if numeric else ""


def _fps(mean_total_ms: float | str) -> float | str:
    return 1000.0 / mean_total_ms if isinstance(mean_total_ms, float) and mean_total_ms > 0 else ""


def _safe_len(value: object) -> int:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile SA-V annotated frames as independent image segmentation samples.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Profile only the first N videos; 0 means all manifest rows.")
    parser.add_argument("--max-frames", type=int, default=30, help="Maximum annotated frames per video; 0 means all annotated frames.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every Nth annotated mask frame.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--backend",
        choices=["null", "sam3", "efficientsam3", "sam2", "efficient-sam2", "efficienttam", "mobilesam"],
        required=True,
    )
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-config")
    parser.add_argument("--external-repo")
    parser.add_argument("--backbone-type", default="efficientvit")
    parser.add_argument("--model-name", default="b0")
    parser.add_argument("--text-encoder-type")
    parser.add_argument("--text-encoder-context-length", type=int, default=77)
    parser.add_argument("--text-encoder-pos-embed-table-size", type=int)
    parser.add_argument("--interpolate-pos-embed", action="store_true")
    parser.add_argument("--mobile-sam-model-type", default="vit_t")
    parser.add_argument("--prompt-mode", choices=["text", "point", "both"], default="point")
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--overlay-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
