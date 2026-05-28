from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .backends import _import_required, _prepend_repo_path
from .overlay import to_numpy
from .profiling import component_timer, cuda_memory_mb, parameter_counts


FIELDNAMES = [
    "model_id",
    "backend",
    "video_id",
    "object_id",
    "frame_index",
    "has_gt",
    "iou",
    "propagate_step_ms",
    "mask_count",
    "width",
    "height",
]


SUMMARY_FIELDS = [
    "model_id",
    "backend",
    "video_id",
    "object_id",
    "frames_tracked",
    "gt_frames_evaluated",
    "mean_iou",
    "session_init_ms",
    "add_prompt_ms",
    "propagate_total_ms",
    "mean_propagate_step_ms",
    "effective_fps",
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
    "other_ms",
    "init_prompt",
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
    "eval_mode",
    "overlay_video",
]


def main() -> None:
    args = parse_args()
    summary = profile_sav_video(args)
    print(json.dumps(summary, indent=2))
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def profile_sav_video(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_manifest(args.manifest)
    predictor, torch_module = _build_predictor(args)
    params = parameter_counts(predictor)
    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    pred_root = getattr(args, "pred_root", None)
    if pred_root:
        pred_root.mkdir(parents=True, exist_ok=True)
    overlay_root = getattr(args, "overlay_root", None)
    if overlay_root:
        overlay_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    with args.csv_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for item in rows:
            summary, frame_rows = _profile_one(args, item, predictor, torch_module, params)
            writer.writerows(frame_rows)
            summaries.append(summary)

    summary_csv = args.csv_output.with_name(args.csv_output.stem + "_summary.csv")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    return {
        "model_id": args.model_id,
        "backend": args.backend,
        "manifest": str(args.manifest),
        "eval_mode": getattr(args, "eval_mode", "both"),
        "csv": str(args.csv_output),
        "summary_csv": str(summary_csv),
        "videos": len(summaries),
        "mean_iou": _mean([row["mean_iou"] for row in summaries]),
        "mean_effective_fps": _mean([row["effective_fps"] for row in summaries]),
        "component_means": {
            key: _mean([row[key] for row in summaries])
            for key in [
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
        },
        "params_and_weights": params,
    }


def _profile_one(
    args: argparse.Namespace,
    item: dict[str, Any],
    predictor: Any,
    torch_module: Any,
    params: dict[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    points = np.asarray([item["point"]], dtype=np.float32)
    labels = np.asarray([int(item.get("point_label", 1))], dtype=np.int32)
    prompt_frame = int(item["prompt_frame_index"])
    max_frames = args.max_frames if args.max_frames > 0 else None
    obj_id = 1

    profile = {}
    frame_rows = []
    step_latencies = []
    ious = []
    overlay_writer = None
    overlay_video = ""
    eval_mode = getattr(args, "eval_mode", "both")
    use_gt = eval_mode in {"gt", "both"}
    use_overlay = eval_mode in {"overlay", "both"} and getattr(args, "overlay_root", None) is not None

    try:
        _sync(torch_module)
        with component_timer(predictor, torch_module) as profile:
            start = perf_counter()
            state = predictor.init_state(video_path=str(item["frames_dir"]))
            _sync(torch_module)
            session_init_ms = (perf_counter() - start) * 1000.0

            start = perf_counter()
            with _autocast_context(torch_module, args.autocast_bfloat16):
                _add_initial_prompt(
                    predictor=predictor,
                    state=state,
                    item=item,
                    frame_idx=prompt_frame,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                    init_prompt=getattr(args, "init_prompt", "point"),
                )
            _sync(torch_module)
            add_prompt_ms = (perf_counter() - start) * 1000.0

            start = perf_counter()
            with _autocast_context(torch_module, args.autocast_bfloat16):
                iterator = predictor.propagate_in_video(
                    state,
                    start_frame_idx=prompt_frame,
                    max_frame_num_to_track=max_frames,
                )
                while True:
                    step_start = perf_counter()
                    try:
                        frame_idx, out_obj_ids, out_mask_logits = next(iterator)
                    except StopIteration:
                        break
                    _sync(torch_module)
                    step_ms = (perf_counter() - step_start) * 1000.0
                    step_latencies.append(step_ms)
                    frame_idx = int(frame_idx)
                    pred_mask = _first_mask(out_mask_logits)
                    gt_mask = _read_gt_mask(item, frame_idx) if use_gt else None
                    iou = _mask_iou(pred_mask, gt_mask) if gt_mask is not None else None
                    if iou is not None:
                        ious.append(iou)
                    pred_root = getattr(args, "pred_root", None)
                    if pred_root:
                        _write_pred_mask(pred_root, item, frame_idx, pred_mask)
                    overlay_root = getattr(args, "overlay_root", None)
                    if use_overlay and overlay_root:
                        frame_bgr = _read_frame(item, frame_idx)
                        if frame_bgr is not None:
                            if overlay_writer is None:
                                overlay_video, overlay_writer = _open_overlay_writer(
                                    overlay_root,
                                    item,
                                    frame_bgr.shape,
                                    getattr(args, "overlay_fps", 24.0),
                                )
                            overlay_writer.write(
                                _overlay_video_frame(
                                    frame_bgr,
                                    pred_mask,
                                    gt_mask,
                                    args.model_id,
                                    item,
                                    frame_idx,
                                    iou,
                                )
                            )
                    frame_rows.append(
                        {
                            "model_id": args.model_id,
                            "backend": args.backend,
                            "video_id": item["video_id"],
                            "object_id": item["object_id"],
                            "frame_index": frame_idx,
                            "has_gt": gt_mask is not None,
                            "iou": iou if iou is not None else "",
                            "propagate_step_ms": step_ms,
                            "mask_count": len(out_obj_ids),
                            "width": pred_mask.shape[1],
                            "height": pred_mask.shape[0],
                        }
                    )
            _sync(torch_module)
            propagate_total_ms = (perf_counter() - start) * 1000.0
    finally:
        if overlay_writer is not None:
            overlay_writer.release()

    component_total = sum(profile.values())
    memory = cuda_memory_mb(torch_module)
    total_ms = session_init_ms + add_prompt_ms + propagate_total_ms
    summary = {
        "model_id": args.model_id,
        "backend": args.backend,
        "video_id": item["video_id"],
        "object_id": item["object_id"],
        "frames_tracked": len(frame_rows),
        "gt_frames_evaluated": len(ious),
        "mean_iou": _mean(ious),
        "session_init_ms": session_init_ms,
        "add_prompt_ms": add_prompt_ms,
        "propagate_total_ms": propagate_total_ms,
        "mean_propagate_step_ms": _mean(step_latencies),
        "effective_fps": len(frame_rows) * 1000.0 / propagate_total_ms if propagate_total_ms > 0 else "",
        "image_encoder_ms": profile.get("image_encoder_ms", 0.0),
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
        "init_prompt": getattr(args, "init_prompt", "point"),
        "cuda_peak_allocated_mb": memory["cuda_peak_allocated_mb"],
        "cuda_peak_reserved_mb": memory["cuda_peak_reserved_mb"],
        "eval_mode": eval_mode,
        "overlay_video": overlay_video,
        **params,
    }
    return summary, frame_rows


def _build_predictor(args: argparse.Namespace) -> tuple[Any, Any]:
    _prepend_repo_path(args.external_repo)
    torch_module = _import_required("torch")
    if args.backend == "efficienttam":
        builder = _import_required("efficient_track_anything.build_efficienttam")
        predictor = builder.build_efficienttam_video_predictor(
            args.model_config,
            args.checkpoint_path,
            device=args.device,
            hydra_overrides_extra=list(_efficienttam_hydra_overrides()),
        )
    elif args.backend in {"sam2", "efficient-sam2"}:
        builder = _import_required("sam2.build_sam")
        predictor = builder.build_sam2_video_predictor(
            args.model_config,
            args.checkpoint_path,
            device=args.device,
        )
    else:
        raise ValueError(f"unsupported video backend: {args.backend}")
    if args.backend == "efficient-sam2":
        _prepare_efficient_sam2_predictor(predictor)
    if hasattr(predictor, "eval"):
        predictor.eval()
    return predictor, torch_module


def _prepare_efficient_sam2_predictor(predictor: Any) -> None:
    if hasattr(predictor, "init_memory_info"):
        predictor.init_memory_info(enable_MeP_info=False)
    elif not hasattr(predictor, "enable_MeP_info"):
        predictor.enable_MeP_info = False
    if not hasattr(predictor, "time_log"):
        predictor.time_log = {}
    if not hasattr(predictor, "Mem_Frame_Prune"):
        predictor.Mem_Frame_Prune = False


def _efficienttam_hydra_overrides() -> tuple[str, ...]:
    return ("++model.compile_image_encoder=False",)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _first_mask(mask_logits: Any) -> np.ndarray:
    values = to_numpy(mask_logits)
    while values.ndim > 2:
        values = values[0]
    return values > 0


def _add_initial_prompt(
    predictor: Any,
    state: Any,
    item: dict[str, Any],
    frame_idx: int,
    obj_id: int,
    points: np.ndarray,
    labels: np.ndarray,
    init_prompt: str,
) -> None:
    if init_prompt == "mask":
        mask = _read_gt_mask(item, frame_idx)
        if mask is None:
            raise RuntimeError(
                f"--init-prompt mask requires a GT mask for video={item['video_id']} "
                f"object={item['object_id']} frame={frame_idx}"
            )
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=mask,
        )
        return
    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
    )


def _read_gt_mask(item: dict[str, Any], frame_idx: int) -> np.ndarray | None:
    if "annotations_dir" not in item:
        return None
    path = Path(item["annotations_dir"]) / str(item["object_id"]) / f"{frame_idx:05d}.png"
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask > 0


def _write_pred_mask(pred_root: Path, item: dict[str, Any], frame_idx: int, mask: np.ndarray) -> None:
    out_dir = pred_root / item["video_id"] / str(item["object_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{frame_idx:05d}.png"), mask.astype(np.uint8) * 255)


def _read_frame(item: dict[str, Any], frame_idx: int) -> np.ndarray | None:
    return cv2.imread(str(Path(item["frames_dir"]) / f"{frame_idx:05d}.jpg"), cv2.IMREAD_COLOR)


def _open_overlay_writer(
    overlay_root: Path,
    item: dict[str, Any],
    frame_shape: tuple[int, int, int],
    fps: float,
) -> tuple[str, cv2.VideoWriter]:
    out_dir = overlay_root / item["video_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{item['object_id']}.mp4"
    height, width = frame_shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to create overlay video: {path}")
    return str(path), writer


def _overlay_video_frame(
    frame_bgr: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray | None,
    model_id: str,
    item: dict[str, Any],
    frame_idx: int,
    iou: float | None,
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = frame_bgr.copy()
    pred = _resize_mask(pred_mask, overlay.shape[:2])
    overlay[pred] = (overlay[pred] * (1.0 - alpha) + np.asarray([40, 220, 60]) * alpha).astype(np.uint8)
    if gt_mask is not None:
        gt = _resize_mask(gt_mask, overlay.shape[:2])
        contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (40, 40, 255), 2)
    label = f"{model_id} {item['video_id']} obj={item['object_id']} frame={frame_idx}"
    if iou is not None:
        label += f" IoU={iou:.3f}"
    cv2.putText(overlay, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def _resize_mask(mask: np.ndarray, frame_hw: tuple[int, int]) -> np.ndarray:
    if mask.shape != frame_hw:
        mask = cv2.resize(mask.astype("uint8"), (frame_hw[1], frame_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def _mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        pred = cv2.resize(pred.astype("uint8"), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(intersection / union) if union else 0.0


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
    parser = argparse.ArgumentParser(description="Profile SAM2-family native video predictors on a fixed SA-V manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--backend", choices=["sam2", "efficient-sam2", "efficienttam"], required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--external-repo", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=96, help="Maximum propagated frames per video; use 0 for all frames.")
    parser.add_argument(
        "--init-prompt",
        choices=["point", "mask"],
        default="point",
        help="Initial SA-V prompt. Use mask to match SAM2-family VOS eval protocols.",
    )
    parser.add_argument("--autocast-bfloat16", action="store_true")
    parser.add_argument(
        "--eval-mode",
        choices=["gt", "overlay", "both", "profile"],
        default="both",
        help="Choose GT metrics, visual overlays, both, or profiling only.",
    )
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--pred-root", type=Path, help="Optional SA-V-style prediction PNG root.")
    parser.add_argument("--overlay-root", type=Path, help="Optional root for per-video overlay MP4s.")
    parser.add_argument("--overlay-fps", type=float, default=24.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
