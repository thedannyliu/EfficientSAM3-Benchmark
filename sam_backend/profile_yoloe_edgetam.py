from __future__ import annotations

import argparse
import csv
import json
import shutil
from contextlib import nullcontext
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .backends import _import_required, _prepend_repo_path
from .overlay import to_numpy
from .profiling import cuda_memory_mb, parameter_counts


FRAME_FIELDS = [
    "source_id",
    "frame_index",
    "track_id",
    "mask_area",
    "tracking_score",
    "edgetam_step_ms",
    "yoloe_validation_ms",
    "yoloe_confidence",
    "yoloe_edgetam_iou",
    "reground_reason",
]

SUMMARY_FIELDS = [
    "source_id",
    "text_prompt",
    "instance_hint",
    "status",
    "frames_tracked",
    "effective_tracking_fps",
    "first_mask_latency_ms",
    "yoloe_set_classes_ms",
    "yoloe_init_ms",
    "yoloe_initial_detection_count",
    "yoloe_initial_top1_confidence",
    "yoloe_initial_top1_gt_iou",
    "yoloe_initial_best_gt_iou",
    "yoloe_initial_best_confidence",
    "yoloe_initial_best_rank",
    "yoloe_initial_localization_note",
    "edgetam_session_init_ms",
    "edgetam_add_prompt_ms",
    "edgetam_propagate_total_ms",
    "mean_edgetam_step_ms",
    "mean_yoloe_validation_ms",
    "reground_count",
    "overlay_video",
]


def main() -> None:
    args = parse_args()
    summary = profile_yoloe_edgetam(args)
    print(json.dumps(summary, indent=2))
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def profile_yoloe_edgetam(args: argparse.Namespace) -> dict[str, Any]:
    sources = _load_sources(args)
    yoloe, predictor, torch_module = _build_models(args)
    yoloe_params = _prefix_dict("yoloe_", parameter_counts(getattr(yoloe, "model", yoloe)))
    edgetam_params = _prefix_dict("edgetam_", parameter_counts(predictor))
    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    if args.overlay_root:
        args.overlay_root.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    with args.csv_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        for source in sources:
            summary, rows = _profile_source(args, source, yoloe, predictor, torch_module, yoloe_params, edgetam_params)
            writer.writerows(rows)
            summaries.append(summary)

    summary_csv = args.csv_output.with_name(args.csv_output.stem + "_summary.csv")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = SUMMARY_FIELDS + sorted(_param_keys(summaries))
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    return {
        "model_id": "yoloe_26m_seg_edgetam",
        "text_prompt": args.text_prompt or "manifest",
        "csv": str(args.csv_output),
        "summary_csv": str(summary_csv),
        "sources": len(summaries),
        "mean_effective_tracking_fps": _mean([row["effective_tracking_fps"] for row in summaries]),
        "mean_first_mask_latency_ms": _mean([row["first_mask_latency_ms"] for row in summaries]),
        "reground_count": sum(int(row["reground_count"]) for row in summaries),
        "cuda_memory_mb": cuda_memory_mb(torch_module),
    }


def _profile_source(
    args: argparse.Namespace,
    source: dict[str, Any],
    yoloe: Any,
    predictor: Any,
    torch_module: Any,
    yoloe_params: dict[str, int],
    edgetam_params: dict[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_id = source["source_id"]
    text_prompt = source.get("text_prompt") or args.text_prompt
    instance_hint = source.get("instance_hint", "")
    if not text_prompt:
        raise ValueError(f"text prompt missing for source {source_id}; pass --text-prompt or add text_prompt to manifest row")
    frames_dir = _materialize_frames(source, args.work_dir / source_id, args.max_frames)
    first_frame = _read_frame(frames_dir, 0)
    if first_frame is None:
        raise RuntimeError(f"failed to read first frame from {frames_dir}")

    _sync(torch_module)
    start = perf_counter()
    yoloe.set_classes([text_prompt])
    _sync(torch_module)
    yoloe_set_classes_ms = (perf_counter() - start) * 1000.0

    start = perf_counter()
    detections = _run_yoloe_candidates(yoloe, first_frame, args)
    _sync(torch_module)
    yoloe_init_ms = (perf_counter() - start) * 1000.0
    gt_mask0 = _read_source_gt_mask(source, int(source.get("prompt_frame_index", 0)))
    localization = _localization_diagnostics(detections, gt_mask0)
    if not detections:
        return _empty_summary(
            args,
            source_id,
            "no_yoloe_detection",
            yoloe_set_classes_ms,
            yoloe_init_ms,
            yoloe_params,
            edgetam_params,
            text_prompt=text_prompt,
            instance_hint=instance_hint,
            localization=localization,
        ), []
    detection = detections[0]

    rows: list[dict[str, Any]] = []
    step_latencies: list[float] = []
    yoloe_validation_latencies: list[float] = []
    reground_count = 0
    overlay_writer = None
    overlay_video = ""
    prev_area: float | None = None
    next_start = 0
    obj_id = 1
    max_frames = args.max_frames if args.max_frames > 0 else _count_frames(frames_dir)

    try:
        _sync(torch_module)
        start = perf_counter()
        state = predictor.init_state(video_path=str(frames_dir))
        _sync(torch_module)
        edgetam_session_init_ms = (perf_counter() - start) * 1000.0

        start = perf_counter()
        with _autocast_context(torch_module, args.autocast_bfloat16):
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                box=detection["box"],
            )
        _sync(torch_module)
        edgetam_add_prompt_ms = (perf_counter() - start) * 1000.0

        propagate_start = perf_counter()
        while len(rows) < max_frames:
            remaining = max_frames - len(rows)
            broke_for_reground = False
            with _autocast_context(torch_module, args.autocast_bfloat16):
                iterator = predictor.propagate_in_video(
                    state,
                    start_frame_idx=next_start,
                    max_frame_num_to_track=remaining,
                )
                while True:
                    step_start = perf_counter()
                    try:
                        frame_idx, out_obj_ids, out_mask_logits = next(iterator)
                    except StopIteration:
                        break
                    _sync(torch_module)
                    step_ms = (perf_counter() - step_start) * 1000.0
                    frame_idx = int(frame_idx)
                    pred_mask = _first_mask(out_mask_logits)
                    area = float(pred_mask.sum())
                    reason = _area_reground_reason(prev_area, area, args.area_jump_ratio)
                    prev_area = area
                    yoloe_ms = ""
                    yoloe_conf = ""
                    yoloe_iou = ""

                    if frame_idx > 0 and args.yoloe_interval > 0 and frame_idx % args.yoloe_interval == 0:
                        frame_bgr = _read_frame(frames_dir, frame_idx)
                        if frame_bgr is not None:
                            validate_start = perf_counter()
                            validation = _run_yoloe(yoloe, frame_bgr, args)
                            _sync(torch_module)
                            yoloe_ms = (perf_counter() - validate_start) * 1000.0
                            yoloe_validation_latencies.append(float(yoloe_ms))
                            if validation is not None:
                                yoloe_conf = validation["confidence"]
                                yoloe_iou = _mask_iou(pred_mask, validation["mask"])
                                if yoloe_iou < args.reground_iou:
                                    reason = reason or "low_yoloe_edgetam_iou"
                                    reground_count += 1
                                    predictor.add_new_points_or_box(
                                        inference_state=state,
                                        frame_idx=frame_idx,
                                        obj_id=obj_id,
                                        box=validation["box"],
                                    )
                                    next_start = frame_idx + 1
                                    broke_for_reground = True

                    if reason and not broke_for_reground:
                        reground_count += 1

                    frame_bgr = _read_frame(frames_dir, frame_idx) if args.overlay_root else None
                    if frame_bgr is not None and args.overlay_root:
                        if overlay_writer is None:
                            overlay_video, overlay_writer = _open_overlay_writer(args.overlay_root, source_id, frame_bgr.shape, args.overlay_fps)
                        overlay_writer.write(_overlay_frame(frame_bgr, pred_mask, text_prompt, source_id, frame_idx, obj_id, reason))

                    rows.append(
                        {
                            "source_id": source_id,
                            "frame_index": frame_idx,
                            "track_id": obj_id if out_obj_ids else "",
                            "mask_area": area,
                            "tracking_score": "",
                            "edgetam_step_ms": step_ms,
                            "yoloe_validation_ms": yoloe_ms,
                            "yoloe_confidence": yoloe_conf,
                            "yoloe_edgetam_iou": yoloe_iou,
                            "reground_reason": reason,
                        }
                    )
                    step_latencies.append(step_ms)
                    if broke_for_reground or len(rows) >= max_frames:
                        break
                if not broke_for_reground and len(rows) < max_frames:
                    break
            if not broke_for_reground:
                break
        _sync(torch_module)
        edgetam_propagate_total_ms = (perf_counter() - propagate_start) * 1000.0
    finally:
        if overlay_writer is not None:
            overlay_writer.release()

    first_mask_latency_ms = yoloe_set_classes_ms + yoloe_init_ms + edgetam_session_init_ms + edgetam_add_prompt_ms
    summary = {
        "source_id": source_id,
        "text_prompt": text_prompt,
        "instance_hint": instance_hint,
        "status": "ok",
        "frames_tracked": len(rows),
        "effective_tracking_fps": len(rows) * 1000.0 / edgetam_propagate_total_ms if edgetam_propagate_total_ms > 0 else "",
        "first_mask_latency_ms": first_mask_latency_ms,
        "yoloe_set_classes_ms": yoloe_set_classes_ms,
        "yoloe_init_ms": yoloe_init_ms,
        **localization,
        "edgetam_session_init_ms": edgetam_session_init_ms,
        "edgetam_add_prompt_ms": edgetam_add_prompt_ms,
        "edgetam_propagate_total_ms": edgetam_propagate_total_ms,
        "mean_edgetam_step_ms": _mean(step_latencies),
        "mean_yoloe_validation_ms": _mean(yoloe_validation_latencies),
        "reground_count": reground_count,
        "overlay_video": overlay_video,
        **yoloe_params,
        **edgetam_params,
    }
    return summary, rows


def _build_models(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    _prepend_repo_path(args.edgetam_external_repo)
    torch_module = _import_required("torch")
    ultralytics = _import_required("ultralytics")
    builder = _import_required("sam2.build_sam")
    yoloe_cls = getattr(ultralytics, "YOLOE")
    yoloe = yoloe_cls(args.yoloe_weights)
    if hasattr(yoloe, "to"):
        yoloe.to(args.device)
    predictor = builder.build_sam2_video_predictor(
        args.edgetam_model_config,
        args.edgetam_checkpoint_path,
        device=args.device,
    )
    if hasattr(predictor, "eval"):
        predictor.eval()
    return yoloe, predictor, torch_module


def _run_yoloe(yoloe: Any, frame_bgr: np.ndarray, args: argparse.Namespace) -> dict[str, Any] | None:
    detections = _run_yoloe_candidates(yoloe, frame_bgr, args)
    return detections[0] if detections else None


def _run_yoloe_candidates(yoloe: Any, frame_bgr: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = yoloe.predict(frame_rgb, imgsz=args.yoloe_imgsz, conf=args.yoloe_conf, verbose=False)
    if not results:
        return []
    result = results[0]
    boxes_obj = getattr(result, "boxes", None)
    masks_obj = getattr(result, "masks", None)
    if boxes_obj is None or masks_obj is None:
        return []
    boxes = to_numpy(getattr(boxes_obj, "xyxy", None))
    scores = to_numpy(getattr(boxes_obj, "conf", None))
    masks = to_numpy(getattr(masks_obj, "data", None))
    if boxes.size == 0 or masks.size == 0:
        return []
    order = np.argsort(-scores) if scores.size else np.arange(len(boxes))
    frame_hw = frame_bgr.shape[:2]
    detections = []
    for rank, idx_value in enumerate(order):
        idx = int(idx_value)
        mask = masks[idx] > 0.5
        if mask.shape != frame_hw:
            mask = cv2.resize(mask.astype("uint8"), (frame_hw[1], frame_hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        detections.append(
            {
                "box": boxes[idx].astype(np.float32),
                "mask": mask,
                "confidence": float(scores[idx]) if scores.size else "",
                "rank": rank + 1,
            }
        )
    return detections


def _materialize_frames(source: dict[str, Any], out_dir: Path, max_frames: int) -> Path:
    if "frames_dir" in source:
        return Path(source["frames_dir"])
    video_path = Path(source["video_path"])
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    frame_idx = 0
    try:
        while max_frames <= 0 or frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if not cv2.imwrite(str(out_dir / f"{frame_idx:05d}.jpg"), frame):
                raise RuntimeError(f"failed to write extracted frame {frame_idx} for {video_path}")
            frame_idx += 1
    finally:
        cap.release()
    if frame_idx == 0:
        raise RuntimeError(f"no frames extracted from {video_path}")
    return out_dir


def _load_sources(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.manifest:
        with args.manifest.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if args.limit > 0:
            rows = rows[: args.limit]
        return [
            {
                **row,
                "source_id": row.get("video_id", f"source_{idx:03d}"),
                "text_prompt": row.get("text_prompt", ""),
                "instance_hint": row.get("text_prompt_instance_hint", row.get("instance_hint", "")),
            }
            for idx, row in enumerate(rows)
        ]
    if args.frames_dir:
        return [{"source_id": args.source_id or Path(args.frames_dir).name, "frames_dir": str(args.frames_dir)}]
    if args.video_path:
        return [{"source_id": args.source_id or Path(args.video_path).stem, "video_path": str(args.video_path)}]
    raise ValueError("one of --manifest, --frames-dir, or --video-path is required")


def _empty_summary(
    args: argparse.Namespace,
    source_id: str,
    status: str,
    yoloe_set_classes_ms: float,
    yoloe_init_ms: float,
    yoloe_params: dict[str, int],
    edgetam_params: dict[str, int],
    text_prompt: str = "",
    instance_hint: str = "",
    localization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    localization = localization or _empty_localization_diagnostics()
    return {
        "source_id": source_id,
        "text_prompt": text_prompt or getattr(args, "text_prompt", ""),
        "instance_hint": instance_hint,
        "status": status,
        "frames_tracked": 0,
        "effective_tracking_fps": "",
        "first_mask_latency_ms": yoloe_set_classes_ms + yoloe_init_ms,
        "yoloe_set_classes_ms": yoloe_set_classes_ms,
        "yoloe_init_ms": yoloe_init_ms,
        **localization,
        "edgetam_session_init_ms": "",
        "edgetam_add_prompt_ms": "",
        "edgetam_propagate_total_ms": "",
        "mean_edgetam_step_ms": "",
        "mean_yoloe_validation_ms": "",
        "reground_count": 0,
        "overlay_video": "",
        **yoloe_params,
        **edgetam_params,
    }


def _first_mask(mask_logits: Any) -> np.ndarray:
    values = to_numpy(mask_logits)
    while values.ndim > 2:
        values = values[0]
    return values > 0


def _read_frame(frames_dir: Path, frame_idx: int) -> np.ndarray | None:
    return cv2.imread(str(frames_dir / f"{frame_idx:05d}.jpg"), cv2.IMREAD_COLOR)


def _read_source_gt_mask(source: dict[str, Any], frame_idx: int) -> np.ndarray | None:
    annotations_dir = source.get("annotations_dir")
    object_id = source.get("object_id")
    if not annotations_dir or object_id is None:
        return None
    path = Path(annotations_dir) / str(object_id) / f"{frame_idx:05d}.png"
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask > 0


def _count_frames(frames_dir: Path) -> int:
    return len(list(frames_dir.glob("*.jpg")))


def _area_reground_reason(previous_area: float | None, current_area: float, max_ratio: float) -> str:
    if previous_area is None or previous_area <= 0 or current_area <= 0:
        return ""
    ratio = max(current_area / previous_area, previous_area / current_area)
    return "mask_area_jump" if ratio > max_ratio else ""


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        left = cv2.resize(left.astype("uint8"), (right.shape[1], right.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    intersection = (left & right).sum()
    union = (left | right).sum()
    return float(intersection / union) if union else 0.0


def _localization_diagnostics(detections: list[dict[str, Any]], gt_mask: np.ndarray | None) -> dict[str, Any]:
    if not detections:
        return _empty_localization_diagnostics(detection_count=0)
    top1 = detections[0]
    values = {
        "yoloe_initial_detection_count": len(detections),
        "yoloe_initial_top1_confidence": top1["confidence"],
        "yoloe_initial_top1_gt_iou": "",
        "yoloe_initial_best_gt_iou": "",
        "yoloe_initial_best_confidence": "",
        "yoloe_initial_best_rank": "",
        "yoloe_initial_localization_note": "gt_unavailable",
    }
    if gt_mask is None:
        return values

    ious = [_mask_iou(detection["mask"], gt_mask) for detection in detections]
    best_idx = int(np.argmax(ious))
    best = detections[best_idx]
    top1_iou = float(ious[0])
    best_iou = float(ious[best_idx])
    values.update(
        {
            "yoloe_initial_top1_gt_iou": top1_iou,
            "yoloe_initial_best_gt_iou": best_iou,
            "yoloe_initial_best_confidence": best["confidence"],
            "yoloe_initial_best_rank": best_idx + 1,
            "yoloe_initial_localization_note": "top1_matches_best" if best_idx == 0 else "same_prompt_different_instance",
        }
    )
    return values


def _empty_localization_diagnostics(detection_count: int | str = "") -> dict[str, Any]:
    return {
        "yoloe_initial_detection_count": detection_count,
        "yoloe_initial_top1_confidence": "",
        "yoloe_initial_top1_gt_iou": "",
        "yoloe_initial_best_gt_iou": "",
        "yoloe_initial_best_confidence": "",
        "yoloe_initial_best_rank": "",
        "yoloe_initial_localization_note": "",
    }


def _open_overlay_writer(overlay_root: Path, source_id: str, frame_shape: tuple[int, int, int], fps: float) -> tuple[str, cv2.VideoWriter]:
    out_dir = overlay_root / source_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "overlay.mp4"
    height, width = frame_shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to create overlay video: {path}")
    return str(path), writer


def _overlay_frame(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    prompt: str,
    source_id: str,
    frame_idx: int,
    track_id: int,
    reason: str,
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = frame_bgr.copy()
    if mask.shape != overlay.shape[:2]:
        mask = cv2.resize(mask.astype("uint8"), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay[mask] = (overlay[mask] * (1.0 - alpha) + np.asarray([40, 220, 60]) * alpha).astype(np.uint8)
    label = f"YOLOE+EdgeTAM {source_id} frame={frame_idx} id={track_id} prompt={prompt}"
    if reason:
        label += f" {reason}"
    cv2.putText(overlay, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def _prefix_dict(prefix: str, values: dict[str, int]) -> dict[str, int]:
    return {prefix + key: value for key, value in values.items()}


def _param_keys(rows: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(key for key in row if key.startswith(("yoloe_params_", "yoloe_weight_", "edgetam_params_", "edgetam_weight_")))
    return keys


def _autocast_context(torch_module: Any, enabled: bool) -> Any:
    if not enabled or not torch_module.cuda.is_available():
        return nullcontext()
    return torch_module.autocast("cuda", dtype=torch_module.bfloat16)


def _sync(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _mean(values: list[float | str | None]) -> float | str:
    numeric = [float(value) for value in values if value not in ("", None)]
    return mean(numeric) if numeric else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile YOLOE-26M-seg text localization plus EdgeTAM video tracking.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="Optional SA-V-style manifest with frames_dir rows.")
    source.add_argument("--frames-dir", type=Path, help="Directory with 00000.jpg-style frames.")
    source.add_argument("--video-path", type=Path, help="Video file to extract into --work-dir before tracking.")
    parser.add_argument("--source-id")
    parser.add_argument("--limit", type=int, default=0, help="Limit manifest rows; 0 means all.")
    parser.add_argument("--text-prompt", help="Text prompt for all sources. Optional for manifests that contain text_prompt per row.")
    parser.add_argument("--yoloe-weights", default="checkpoints/yoloe/yoloe-26m-seg.pt")
    parser.add_argument("--yoloe-imgsz", type=int, default=640)
    parser.add_argument("--yoloe-conf", type=float, default=0.25)
    parser.add_argument("--yoloe-interval", type=int, default=20)
    parser.add_argument("--reground-iou", type=float, default=0.35)
    parser.add_argument("--area-jump-ratio", type=float, default=2.5)
    parser.add_argument("--edgetam-external-repo", default="external/EdgeTAM")
    parser.add_argument("--edgetam-checkpoint-path", default="checkpoints/edgetam/edgetam.pt")
    parser.add_argument("--edgetam-model-config", default="external/EdgeTAM/sam2/configs/edgetam.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--autocast-bfloat16", action="store_true")
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--overlay-root", type=Path)
    parser.add_argument("--overlay-fps", type=float, default=24.0)
    parser.add_argument("--work-dir", type=Path, default=Path("results/yoloe_edgetam/work"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
