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

from .coco_manifest import ann_to_mask
from .overlay import overlay_prediction, to_numpy
from .profiling import cuda_memory_mb, parameter_counts


FIELDNAMES = [
    "model_id",
    "family",
    "weights",
    "sample_id",
    "prompt",
    "image",
    "image_id",
    "annotation_id",
    "category_name",
    "width",
    "height",
    "all_detection_count",
    "target_detection_count",
    "mask_count",
    "box_count",
    "score_max",
    "best_iou",
    "merged_iou",
    "best_box_iou",
    "total_ms",
    "set_classes_ms",
    "predict_ms",
    "postprocess_ms",
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
    "params_yolo_backbone",
    "params_yolo_neck",
    "params_yolo_head",
    "weight_yolo_backbone_bytes",
    "weight_yolo_neck_bytes",
    "weight_yolo_head_bytes",
    "yolo_backbone_layers",
    "yolo_neck_layers",
    "yolo_head_layers",
    "checkpoint_file_bytes",
    "component_note",
    "overlay",
]


def main() -> None:
    args = parse_args()
    summary = profile_yolo_coco(args)
    print(json.dumps(summary, indent=2))
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def profile_yolo_coco(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_manifest(args.manifest)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    model, torch_module = _build_model(args)
    model_core = getattr(model, "model", model)
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()
    params = _yolo_parameter_counts(model_core)
    checkpoint_file_bytes = _checkpoint_file_bytes(model, args.weights)

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    use_gt = args.eval_mode in {"gt", "both"}
    use_overlay = args.eval_mode in {"overlay", "both"} and args.overlay_dir is not None
    if use_overlay:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, Any]] = []
    current_prompt: str | None = None
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

            prompt = str(item.get("text_prompt") or item["category_name"])
            set_classes_ms = 0.0
            if args.family == "yoloe-seg" and prompt != current_prompt:
                start = perf_counter()
                _set_open_vocab_classes(model, [prompt])
                _sync(torch_module)
                set_classes_ms = (perf_counter() - start) * 1000.0
                current_prompt = prompt

            predict_kwargs = _predict_kwargs(args)
            start = perf_counter()
            results = model.predict(frame_rgb, **predict_kwargs)
            _sync(torch_module)
            predict_ms = (perf_counter() - start) * 1000.0

            start = perf_counter()
            result = results[0] if results else None
            all_detections = _extract_detections(result, frame_rgb.shape[:2])
            detections = all_detections
            if args.family == "yolo-seg":
                detections = _filter_detections_by_class(all_detections, prompt)
            detections = detections[: args.max_det_for_iou] if args.max_det_for_iou > 0 else detections
            best_iou, merged_iou = _mask_ious([det["mask"] for det in detections], gt_mask) if use_gt else ("", "")
            best_box_iou = _best_box_iou([det["box"] for det in detections], item) if use_gt else ""
            overlay_path = _write_overlay(args.overlay_dir, item, frame_rgb, detections) if use_overlay else ""
            postprocess_ms = (perf_counter() - start) * 1000.0
            total_ms = set_classes_ms + predict_ms + postprocess_ms
            memory = cuda_memory_mb(torch_module)
            scores = [det["score"] for det in detections if det["score"] != ""]
            row = {
                "model_id": args.model_id,
                "family": args.family,
                "weights": args.weights,
                "sample_id": item["sample_id"],
                "prompt": prompt,
                "image": item["image_path"],
                "image_id": item["image_id"],
                "annotation_id": item["annotation_id"],
                "category_name": item["category_name"],
                "width": frame_rgb.shape[1],
                "height": frame_rgb.shape[0],
                "all_detection_count": len(all_detections),
                "target_detection_count": len(detections),
                "mask_count": sum(1 for det in detections if det["mask"] is not None),
                "box_count": sum(1 for det in detections if det["box"] is not None),
                "score_max": max(scores) if scores else "",
                "best_iou": best_iou,
                "merged_iou": merged_iou,
                "best_box_iou": best_box_iou,
                "total_ms": total_ms,
                "set_classes_ms": set_classes_ms,
                "predict_ms": predict_ms,
                "postprocess_ms": postprocess_ms,
                "checkpoint_file_bytes": checkpoint_file_bytes,
                "overlay": str(overlay_path),
                **memory,
                **params,
            }
            writer.writerow(row)
            output_rows.append(row)

    return _summarize(args, output_rows)


def _build_model(args: argparse.Namespace) -> tuple[Any, Any]:
    try:
        import torch
        import ultralytics
    except ImportError as exc:
        raise RuntimeError("Ultralytics and torch are required for YOLO COCO profiling") from exc

    if args.family == "yoloe-seg":
        model_cls = getattr(ultralytics, "YOLOE")
    elif args.family == "yolo-seg":
        model_cls = getattr(ultralytics, "YOLO")
    else:
        raise ValueError(f"unsupported YOLO family: {args.family}")

    model = model_cls(args.weights)
    if hasattr(model, "to"):
        model.to(args.device)
    return model, torch


def _predict_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "verbose": False,
        "device": args.device,
    }
    if args.max_det > 0:
        kwargs["max_det"] = args.max_det
    if args.agnostic_nms is not None:
        kwargs["agnostic_nms"] = args.agnostic_nms
    return kwargs


def _set_open_vocab_classes(model: Any, class_names: list[str]) -> None:
    if not hasattr(model, "set_classes"):
        raise RuntimeError("selected YOLOE model does not expose set_classes")
    model.set_classes(class_names)


def _extract_detections(result: Any, frame_hw: tuple[int, int]) -> list[dict[str, Any]]:
    if result is None:
        return []
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None:
        return []
    boxes = to_numpy(getattr(boxes_obj, "xyxy", None))
    scores = to_numpy(getattr(boxes_obj, "conf", None))
    classes = to_numpy(getattr(boxes_obj, "cls", None))
    masks_obj = getattr(result, "masks", None)
    masks = to_numpy(getattr(masks_obj, "data", None)) if masks_obj is not None else np.asarray([])
    names = getattr(result, "names", None)
    detections = []
    for idx in range(len(boxes)):
        mask = _resize_mask(masks[idx], frame_hw) if masks.size and idx < len(masks) else None
        class_id = int(classes[idx]) if classes.size and idx < len(classes) else None
        score = float(scores[idx]) if scores.size and idx < len(scores) else ""
        detections.append(
            {
                "box": boxes[idx].astype(np.float32),
                "mask": mask,
                "score": score,
                "class_id": class_id,
                "class_name": _class_name(names, class_id),
            }
        )
    return detections


def _filter_detections_by_class(detections: list[dict[str, Any]], target_name: str) -> list[dict[str, Any]]:
    target = _normalize_class_name(target_name)
    return [det for det in detections if _normalize_class_name(str(det.get("class_name", ""))) == target]


def _class_name(names: Any, class_id: int | None) -> str:
    if class_id is None or names is None:
        return ""
    if isinstance(names, dict):
        return str(names.get(class_id, ""))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return ""


def _normalize_class_name(value: str) -> str:
    return value.strip().lower().replace("_", " ")


def _resize_mask(mask: Any, frame_hw: tuple[int, int]) -> np.ndarray:
    pred = to_numpy(mask) > 0.5
    if pred.shape != frame_hw:
        pred = cv2.resize(pred.astype("uint8"), (frame_hw[1], frame_hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return pred.astype(bool)


def _mask_ious(masks: list[np.ndarray | None], gt_mask: Any) -> tuple[float, float]:
    pred_masks = [mask.astype(bool) for mask in masks if mask is not None]
    if not pred_masks:
        return 0.0, 0.0
    resized = []
    for mask in pred_masks:
        if mask.shape != gt_mask.shape:
            mask = cv2.resize(mask.astype("uint8"), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        resized.append(mask)
    ious = [_mask_iou(mask, gt_mask) for mask in resized]
    merged = resized[0].copy()
    for mask in resized[1:]:
        merged |= mask
    return max(ious), _mask_iou(merged, gt_mask)


def _mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(intersection / union) if union else 0.0


def _best_box_iou(boxes: list[Any], item: dict[str, Any]) -> float:
    gt = _bbox_xyxy(item)
    if gt is None:
        return 0.0
    values = [_box_iou(tuple(float(v) for v in box[:4]), gt) for box in boxes if box is not None and len(box) >= 4]
    return max(values) if values else 0.0


def _box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - inter
    return inter / union if union else 0.0


def _bbox_xyxy(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = item.get("bbox_xywh")
    if not bbox or len(bbox) != 4:
        return None
    x, y, w, h = [float(value) for value in bbox]
    return x, y, x + w, y + h


def _write_overlay(
    overlay_dir: Path | None,
    item: dict[str, Any],
    frame_rgb: np.ndarray,
    detections: list[dict[str, Any]],
) -> Path:
    assert overlay_dir is not None
    path = overlay_dir / f"{item['sample_id']}.png"
    masks = [det["mask"] for det in detections if det["mask"] is not None]
    boxes = [det["box"] for det in detections if det["box"] is not None]
    scores = [det["score"] for det in detections if det["score"] != ""]
    overlay_rgb = overlay_prediction(frame_rgb, masks, boxes, scores)
    ok = cv2.imwrite(str(path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"failed to write overlay image: {path}")
    return path


def _yolo_parameter_counts(model: Any) -> dict[str, Any]:
    counts = parameter_counts(model)
    layers = list(getattr(model, "model", []) or [])
    if not layers:
        return counts

    split = _split_yolo_layers(layers)
    backbone_params, backbone_bytes = _module_list_stats(split["backbone"])
    neck_params, neck_bytes = _module_list_stats(split["neck"])
    head_params, head_bytes = _module_list_stats(split["head"])
    counts["params_segmentation_head"] = head_params
    counts["weight_segmentation_head_bytes"] = head_bytes
    counts["params_backbone"] = backbone_params
    counts["weight_backbone_bytes"] = backbone_bytes
    counts["params_detector"] = counts["params_total"]
    counts["weight_detector_bytes"] = counts["weight_total_bytes"]
    counts["params_yolo_backbone"] = backbone_params
    counts["params_yolo_neck"] = neck_params
    counts["params_yolo_head"] = head_params
    counts["weight_yolo_backbone_bytes"] = backbone_bytes
    counts["weight_yolo_neck_bytes"] = neck_bytes
    counts["weight_yolo_head_bytes"] = head_bytes
    counts["yolo_backbone_layers"] = _layer_range(split["backbone_indices"])
    counts["yolo_neck_layers"] = _layer_range(split["neck_indices"])
    counts["yolo_head_layers"] = _layer_range(split["head_indices"])
    counts["component_note"] = (
        "Ultralytics YOLO split: backbone ends before the first neck "
        "Upsample/Concat layer; neck excludes the final detection/segment head."
    )
    return counts


def _split_yolo_layers(layers: list[Any]) -> dict[str, Any]:
    head_start = max(0, len(layers) - 1)
    neck_start = _infer_neck_start(layers, head_start)
    return {
        "backbone": layers[:neck_start],
        "neck": layers[neck_start:head_start],
        "head": layers[head_start:],
        "backbone_indices": list(range(0, neck_start)),
        "neck_indices": list(range(neck_start, head_start)),
        "head_indices": list(range(head_start, len(layers))),
    }


def _infer_neck_start(layers: list[Any], head_start: int) -> int:
    for idx, layer in enumerate(layers[:head_start]):
        name = _layer_name(layer)
        if "upsample" in name or "concat" in name:
            return idx
    for idx, layer in enumerate(layers[:head_start]):
        if "spp" in _layer_name(layer):
            return min(idx + 1, head_start)
    return head_start


def _module_list_stats(modules: list[Any]) -> tuple[int, int]:
    seen: set[int] = set()
    params = 0
    bytes_count = 0
    for module in modules:
        if not hasattr(module, "parameters"):
            continue
        for param in module.parameters():
            marker = id(param)
            if marker in seen:
                continue
            seen.add(marker)
            params += param.numel()
            bytes_count += param.numel() * param.element_size()
    return params, bytes_count


def _layer_name(layer: Any) -> str:
    return str(getattr(layer, "type", type(layer).__name__)).lower()


def _layer_range(indices: list[int]) -> str:
    if not indices:
        return ""
    return str(indices[0]) if len(indices) == 1 else f"{indices[0]}-{indices[-1]}"


def _checkpoint_file_bytes(model: Any, weights: str) -> int | str:
    candidates = [Path(weights)]
    for attr in ("ckpt_path", "pt_path"):
        value = getattr(model, attr, None)
        if value:
            candidates.append(Path(str(value)))
    core = getattr(model, "model", None)
    if core is not None:
        for attr in ("ckpt_path", "pt_path"):
            value = getattr(core, attr, None)
            if value:
                candidates.append(Path(str(value)))
    for path in candidates:
        if path.exists() and path.is_file():
            return path.stat().st_size
    return ""


def _sync(torch_module: Any) -> None:
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _summarize(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model_id": args.model_id,
        "family": args.family,
        "weights": args.weights,
        "manifest": str(args.manifest),
        "eval_mode": args.eval_mode,
        "csv": str(args.csv_output),
        "samples": len({row["sample_id"] for row in rows}),
        "rows": len(rows),
        "mean_total_ms": _mean_numeric([row["total_ms"] for row in rows]),
        "effective_fps": _fps(_mean_numeric([row["total_ms"] for row in rows])),
        "miou_best": _mean_numeric([row["best_iou"] for row in rows]),
        "miou_merged": _mean_numeric([row["merged_iou"] for row in rows]),
        "mean_best_box_iou": _mean_numeric([row["best_box_iou"] for row in rows]),
        "mean_target_detection_count": _mean_numeric([row["target_detection_count"] for row in rows]),
    }


def _mean_numeric(values: list[Any]) -> float | str:
    numeric = [float(value) for value in values if value not in ("", None)]
    return mean(numeric) if numeric else ""


def _fps(mean_total_ms: float | str) -> float | str:
    return 1000.0 / mean_total_ms if isinstance(mean_total_ms, float) and mean_total_ms > 0 else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile YOLO segmentation models on a fixed COCO manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Profile only the first N manifest rows; 0 means all rows.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--family", choices=["yoloe-seg", "yolo-seg"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--max-det-for-iou", type=int, default=100)
    parser.add_argument("--agnostic-nms", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--eval-mode", choices=["gt", "overlay", "both", "profile"], default="both")
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--overlay-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
