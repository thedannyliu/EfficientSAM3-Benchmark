from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import cv2

from .backends import BackendConfig, Prompt, create_backend
from .coco_manifest import ann_to_mask
from .overlay import overlay_prediction, to_numpy
from .profiling import component_timer, cuda_memory_mb, parameter_counts


FIELDNAMES = [
    "model_id",
    "backend",
    "sample_id",
    "prompt_mode",
    "prompt",
    "point_x",
    "point_y",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "image",
    "image_id",
    "annotation_id",
    "category_name",
    "width",
    "height",
    "mask_count",
    "box_count",
    "score_max",
    "best_iou",
    "merged_iou",
    "total_ms",
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
    "other_ms",
    "cuda_allocated_mb",
    "cuda_reserved_mb",
    "cuda_peak_allocated_mb",
    "cuda_peak_reserved_mb",
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
    "overlay",
]


def main() -> None:
    args = parse_args()
    summary = profile_coco(args)
    print(json.dumps(summary, indent=2))
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def profile_coco(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_manifest(args.manifest)
    limit = getattr(args, "limit", 0)
    if limit and limit > 0:
        rows = rows[:limit]
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
            enable_inst_interactivity=args.prompt_mode in {"point", "box", "both", "all"},
            model_config=args.model_config,
            external_repo=args.external_repo,
            mobile_sam_model_type=getattr(args, "mobile_sam_model_type", "vit_t"),
        )
    )
    torch_module = getattr(backend, "torch", None)
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()
    params = parameter_counts(getattr(backend, "model", None))

    interactive_only_backends = {"sam2", "efficient-sam2", "efficienttam", "mobilesam"}
    if args.backend in interactive_only_backends and args.prompt_mode == "text":
        raise ValueError(f"{args.backend} supports point prompts in this benchmark, not text prompts")
    prompt_modes = _prompt_modes(args.prompt_mode)
    if args.backend in interactive_only_backends and args.prompt_mode == "both":
        prompt_modes = ["point"]
    if args.backend in interactive_only_backends and args.prompt_mode == "all":
        prompt_modes = ["point", "box"]
    eval_mode = getattr(args, "eval_mode", "both")
    use_gt = eval_mode in {"gt", "both"}
    use_overlay = eval_mode in {"overlay", "both"} and args.overlay_dir is not None
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    if use_overlay:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)

    output_rows = []
    with args.csv_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for item in rows:
            frame_bgr = cv2.imread(str(item["image_path"]), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise RuntimeError(f"failed to read image: {item['image_path']}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            gt_mask = ann_to_mask(item, int(item["width"]), int(item["height"])) if use_gt else None
            if use_gt and gt_mask is None:
                raise RuntimeError(f"failed to decode ground-truth mask for sample {item['sample_id']}")

            for prompt_mode in prompt_modes:
                prompt = _build_prompt(item, prompt_mode)
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
                scores = to_numpy(prediction.scores)
                best_iou, merged_iou = _prediction_iou(prediction.masks, gt_mask) if use_gt else ("", "")
                overlay_path = _write_overlay(args.overlay_dir, item, prompt_mode, frame_rgb, prediction) if use_overlay else ""
                point = item["point"]
                box = _bbox_xyxy(item)
                row = {
                    "model_id": args.model_id,
                    "backend": args.backend,
                    "sample_id": item["sample_id"],
                    "prompt_mode": prompt_mode,
                    "prompt": prompt.text or "",
                    "point_x": point[0],
                    "point_y": point[1],
                    "box_x1": box[0] if box else "",
                    "box_y1": box[1] if box else "",
                    "box_x2": box[2] if box else "",
                    "box_y2": box[3] if box else "",
                    "image": item["image_path"],
                    "image_id": item["image_id"],
                    "annotation_id": item["annotation_id"],
                    "category_name": item["category_name"],
                    "width": frame_rgb.shape[1],
                    "height": frame_rgb.shape[0],
                    "mask_count": _safe_len(prediction.masks),
                    "box_count": _safe_len(prediction.boxes),
                    "score_max": float(scores.max()) if scores.size else "",
                    "best_iou": best_iou,
                    "merged_iou": merged_iou,
                    "total_ms": total_ms,
                    "image_encoder_ms": profile.get("image_encoder_ms", 0.0),
                    "text_encoder_ms": profile.get("text_encoder_ms", 0.0),
                    "prompt_encoder_ms": profile.get("prompt_encoder_ms", 0.0),
                    "mask_decoder_ms": profile.get("mask_decoder_ms", 0.0),
                    "transformer_ms": profile.get("transformer_ms", 0.0),
                    "geometry_encoder_ms": profile.get("geometry_encoder_ms", 0.0),
                    "segmentation_head_ms": profile.get("segmentation_head_ms", 0.0),
                    "grounding_ms": profile.get("grounding_ms", 0.0),
                    "detector_ms": profile.get("detector_ms", 0.0),
                    "memory_attention_ms": profile.get("memory_attention_ms", 0.0),
                    "memory_encoder_ms": profile.get("memory_encoder_ms", 0.0),
                    "other_ms": max(0.0, total_ms - component_total),
                    "overlay": str(overlay_path),
                    **memory,
                    **params,
                }
                writer.writerow(row)
                output_rows.append(row)

    return _summarize(args, output_rows)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_prompt(item: dict[str, Any], prompt_mode: str) -> Prompt:
    if prompt_mode == "text":
        return Prompt(text=item["text_prompt"])
    if prompt_mode == "point":
        point = item["point"]
        return Prompt(points=[(float(point[0]), float(point[1]))], labels=[int(item.get("point_label", 1))])
    if prompt_mode == "box":
        box = _bbox_xyxy(item)
        if box is None:
            raise ValueError(f"missing bbox_xywh for sample {item.get('sample_id', '')}")
        return Prompt(boxes=[box])
    raise ValueError(f"unknown prompt mode: {prompt_mode}")


def _prompt_modes(prompt_mode: str) -> list[str]:
    if prompt_mode == "both":
        return ["text", "point"]
    if prompt_mode == "all":
        return ["text", "point", "box"]
    return [prompt_mode]


def _bbox_xyxy(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = item.get("bbox_xywh")
    if not bbox or len(bbox) != 4:
        return None
    x, y, w, h = [float(value) for value in bbox]
    return x, y, x + w, y + h


def _prediction_iou(masks: Any, gt_mask: Any) -> tuple[float, float]:
    values = to_numpy(masks)
    if values.size == 0:
        return 0.0, 0.0
    if values.ndim == 4:
        values = values[:, 0]
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3:
        return 0.0, 0.0

    pred_masks = []
    for mask in values:
        pred = mask.astype(bool)
        if pred.shape != gt_mask.shape:
            pred = cv2.resize(pred.astype("uint8"), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        pred_masks.append(pred)

    ious = [_mask_iou(pred, gt_mask) for pred in pred_masks]
    merged = pred_masks[0].copy()
    for pred in pred_masks[1:]:
        merged |= pred
    return max(ious) if ious else 0.0, _mask_iou(merged, gt_mask)


def _mask_iou(pred: Any, gt: Any) -> float:
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(intersection / union) if union else 0.0


def _write_overlay(
    overlay_dir: Path | None,
    item: dict[str, Any],
    prompt_mode: str,
    frame_rgb: Any,
    prediction: Any,
) -> Path:
    assert overlay_dir is not None
    path = overlay_dir / f"{item['sample_id']}-{prompt_mode}.png"
    overlay_rgb = overlay_prediction(frame_rgb, prediction.masks, prediction.boxes, prediction.scores)
    ok = cv2.imwrite(str(path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"failed to write overlay image: {path}")
    return path


def _summarize(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row["prompt_mode"]), []).append(row)
    return {
        "model_id": args.model_id,
        "backend": args.backend,
        "manifest": str(args.manifest),
        "eval_mode": getattr(args, "eval_mode", "both"),
        "csv": str(args.csv_output),
        "samples": len({row["sample_id"] for row in rows}),
        "rows": len(rows),
        "prompt_modes": {
            mode: {
                "rows": len(mode_rows),
                "mean_total_ms": mean(float(row["total_ms"]) for row in mode_rows),
                "miou_best": _mean_numeric([row["best_iou"] for row in mode_rows]),
                "miou_merged": _mean_numeric([row["merged_iou"] for row in mode_rows]),
            }
            for mode, mode_rows in sorted(by_mode.items())
        },
    }


def _safe_len(value: object) -> int:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _mean_numeric(values: list[Any]) -> float | str:
    numeric = [float(value) for value in values if value not in ("", None)]
    return mean(numeric) if numeric else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a backend on a fixed COCO manifest with IoU metrics.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Profile only the first N manifest rows; 0 means all rows.")
    parser.add_argument("--model-id", default="sam3-coco")
    parser.add_argument(
        "--backend",
        choices=["null", "sam3", "efficientsam3", "sam2", "efficient-sam2", "efficienttam", "mobilesam"],
        default="sam3",
    )
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-config", help="Native config path for SAM2/EfficientTAM-style backends.")
    parser.add_argument("--external-repo", help="Optional repo root to prepend to PYTHONPATH for external backends.")
    parser.add_argument("--backbone-type", default="efficientvit")
    parser.add_argument("--model-name", default="b0")
    parser.add_argument("--text-encoder-type")
    parser.add_argument("--text-encoder-context-length", type=int, default=77)
    parser.add_argument("--text-encoder-pos-embed-table-size", type=int)
    parser.add_argument("--interpolate-pos-embed", action="store_true")
    parser.add_argument("--mobile-sam-model-type", default="vit_t")
    parser.add_argument("--prompt-mode", choices=["text", "point", "box", "both", "all"], default="both")
    parser.add_argument(
        "--eval-mode",
        choices=["gt", "overlay", "both", "profile"],
        default="both",
        help="Choose GT metrics, visual overlays, both, or profiling only.",
    )
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--overlay-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
