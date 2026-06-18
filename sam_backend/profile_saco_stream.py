from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .backends import BackendConfig, Prediction, Prompt, create_backend
from .overlay import merge_masks, to_numpy
from .profiling import component_timer, cuda_memory_mb, parameter_counts
from .streaming import masks_to_bbox_xyxy


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

FRAME_FIELDS = [
    "model_id",
    "backend",
    "stream_mode",
    "source_id",
    "video_id",
    "category_id",
    "frame_index",
    "prompt_mode",
    "prompt",
    "has_gt",
    "is_positive",
    "iou",
    "matched_iou",
    "presence_correct",
    "latency_ms",
    "callback_total_ms",
    "end_to_end_ms",
    "mask_count",
    "box_count",
    "score_max",
    *COMPONENT_FIELDS,
    "other_ms",
    "cuda_peak_allocated_mb",
    "cuda_peak_reserved_mb",
]

SUMMARY_FIELDS = [
    "model_id",
    "backend",
    "stream_mode",
    "sources",
    "eval_start_frame",
    "frames",
    "positive_frames",
    "mean_iou",
    "mask_ap_50_95",
    "mask_f1_50",
    "mask_f1_75",
    "presence_accuracy",
    "mean_latency_ms",
    "p95_latency_ms",
    "mean_callback_total_ms",
    "mean_end_to_end_ms",
    "p95_end_to_end_ms",
    "effective_fps",
    "input_fps",
    "overlay_video_count",
    "pred_json",
    "official_eval_json",
    *[f"mean_{field}" for field in COMPONENT_FIELDS],
    "cuda_peak_allocated_mb",
    "cuda_peak_reserved_mb",
    "params_total",
    "weight_total_bytes",
]


@dataclass
class BBoxChainState:
    initial_prompt: Prompt
    bbox_min_area: int = 25
    initial_prompt_frame_index: int = 0
    tracking_bbox: tuple[float, float, float, float] | None = None
    used_initial_prompt: bool = False

    def next_prompt(self, frame_index: int) -> tuple[Prompt | None, str]:
        if frame_index < self.initial_prompt_frame_index:
            return None, "pre_prompt"
        if not self.used_initial_prompt:
            self.used_initial_prompt = True
            if self.initial_prompt.text:
                return self.initial_prompt, "text"
            return self.initial_prompt, "point"
        if self.tracking_bbox is None:
            return None, "lost"
        return Prompt(boxes=[self.tracking_bbox]), "box"

    def update(self, masks: Any, frame_hw: tuple[int, int]) -> tuple[float, float, float, float] | None:
        self.tracking_bbox = masks_to_bbox_xyxy(masks, frame_hw, min_area=self.bbox_min_area)
        return self.tracking_bbox


def main() -> None:
    args = parse_args()
    summary = profile_saco_stream(args)
    print(json.dumps(summary, indent=2))
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def profile_saco_stream(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_manifest(args.manifest)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("manifest did not provide any SA-Co/VEval rows")

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    if args.overlay_root:
        args.overlay_root.mkdir(parents=True, exist_ok=True)
    if args.pred_json:
        args.pred_json.parent.mkdir(parents=True, exist_ok=True)

    if args.stream_mode == "native_video":
        frame_rows, pred_entries, overlay_videos, params, memory = _profile_native(args, rows)
    elif args.stream_mode == "image_per_frame":
        frame_rows, pred_entries, overlay_videos, params, memory = _profile_image_per_frame(args, rows)
    else:
        frame_rows, pred_entries, overlay_videos, params, memory = _profile_bbox_chain(args, rows)

    with args.csv_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FRAME_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(frame_rows)

    if args.pred_json:
        args.pred_json.write_text(json.dumps(pred_entries) + "\n", encoding="utf-8")
    official_eval_json = _run_official_eval(args, frame_rows) if args.pred_json and args.gt_annotation_file and args.official_eval_json else ""

    summary_rows = [_summary_row(args, frame_rows, overlay_videos, params, memory, official_eval_json)]
    summary_csv = args.csv_output.with_name(args.csv_output.stem + "_summary.csv")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    return {
        "model_id": args.model_id,
        "backend": args.backend,
        "stream_mode": args.stream_mode,
        "csv": str(args.csv_output),
        "summary_csv": str(summary_csv),
        "pred_json": str(args.pred_json) if args.pred_json else "",
        "official_eval_json": official_eval_json,
        "sources": len({row["source_id"] for row in frame_rows}),
        "eval_start_frame": _eval_start_frame(frame_rows),
        "frames": len(frame_rows),
        "mean_iou": _mean([row["iou"] for row in frame_rows]),
        "effective_fps": _fps(_mean([row["end_to_end_ms"] for row in frame_rows])),
        "overlay_videos": len(overlay_videos),
    }


def _profile_bbox_chain(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, int], dict[str, float]]:
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
            enable_inst_interactivity=True,
            model_config=args.model_config,
            external_repo=args.external_repo,
            mobile_sam_model_type=args.mobile_sam_model_type,
        )
    )
    torch_module = getattr(backend, "torch", None)
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()
    params = parameter_counts(getattr(backend, "model", None))
    all_rows: list[dict[str, Any]] = []
    pred_entries = []
    overlay_videos = []

    for item in rows:
        frame_rows, pred_entry, overlay_video = _profile_bbox_chain_source(args, item, backend, torch_module)
        all_rows.extend(frame_rows)
        pred_entries.extend(pred_entry)
        if overlay_video:
            overlay_videos.append(overlay_video)
    memory = cuda_memory_mb(torch_module) if torch_module is not None else cuda_memory_mb(None)
    return all_rows, pred_entries, overlay_videos, params, memory


def _profile_image_per_frame(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, int], dict[str, float]]:
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
            enable_inst_interactivity=True,
            model_config=args.model_config,
            external_repo=args.external_repo,
            mobile_sam_model_type=args.mobile_sam_model_type,
        )
    )
    torch_module = getattr(backend, "torch", None)
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()
    params = parameter_counts(getattr(backend, "model", None))
    all_rows: list[dict[str, Any]] = []
    pred_entries = []
    overlay_videos = []

    for item in rows:
        frame_rows, pred_entry, overlay_video = _profile_image_per_frame_source(args, item, backend, torch_module)
        all_rows.extend(frame_rows)
        pred_entries.extend(pred_entry)
        if overlay_video:
            overlay_videos.append(overlay_video)
    memory = cuda_memory_mb(torch_module) if torch_module is not None else cuda_memory_mb(None)
    return all_rows, pred_entries, overlay_videos, params, memory


def _profile_image_per_frame_source(
    args: argparse.Namespace,
    item: dict[str, Any],
    backend: Any,
    torch_module: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    frame_paths = _frame_paths(item, args.max_frames)
    gt_masks = _gt_masks_by_frame(item)
    prompt_type = _resolved_prompt_type(args)
    start_frame = _initial_prompt_frame_index(prompt_type, gt_masks, len(frame_paths), item)

    writer = None
    overlay_video = ""
    rows = []
    pred_masks: list[np.ndarray | None] = []
    pred_scores: list[float] = []
    try:
        for frame_offset, frame_path in enumerate(frame_paths[start_frame:], start=start_frame):
            frame_rgb = _read_rgb(frame_path)
            gt_mask = gt_masks.get(frame_offset)
            prompt, prompt_mode = _image_per_frame_prompt(args, item, gt_mask, prompt_type)
            prediction = Prediction(masks=[], boxes=[], scores=[], latency_ms=0.0, metadata={})
            profile = {}
            callback_start = perf_counter()
            if prompt is not None:
                if torch_module is not None:
                    with component_timer(getattr(backend, "model", None), torch_module) as profile:
                        prediction = backend.predict(frame_rgb, prompt)
                else:
                    prediction = backend.predict(frame_rgb, prompt)
            callback_ms = (perf_counter() - callback_start) * 1000.0
            component_total = sum(profile.values())
            pred_mask = merge_masks(prediction.masks, frame_rgb.shape[:2])
            iou = _mask_iou(pred_mask, gt_mask) if gt_mask is not None else ""
            present_pred = pred_mask is not None and bool(pred_mask.any())
            present_gt = gt_mask is not None and bool(gt_mask.any())
            scores = to_numpy(prediction.scores)
            score_max = float(scores.max()) if scores.size else (1.0 if present_pred else 0.0)
            pred_scores.append(score_max)
            pred_masks.append(pred_mask)
            if args.overlay_root:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                if writer is None:
                    overlay_video, writer = _open_overlay_writer(args.overlay_root, item, frame_bgr.shape, args.overlay_fps)
                writer.write(
                    _overlay_frame(
                        frame_bgr,
                        pred_mask,
                        gt_mask,
                        args.model_id,
                        item,
                        frame_offset,
                        iou,
                        callback_ms,
                        prompt_mode,
                    )
                )
            memory = cuda_memory_mb(torch_module) if torch_module is not None else cuda_memory_mb(None)
            rows.append(
                {
                    "model_id": args.model_id,
                    "backend": args.backend,
                    "stream_mode": args.stream_mode,
                    "source_id": item["source_id"],
                    "video_id": item["video_id"],
                    "category_id": item["category_id"],
                    "frame_index": frame_offset,
                    "prompt_mode": prompt_mode,
                    "prompt": _prompt_text(prompt),
                    "has_gt": gt_mask is not None,
                    "is_positive": bool(item.get("is_positive", True)),
                    "iou": iou,
                    "matched_iou": iou,
                    "presence_correct": present_pred == present_gt,
                    "latency_ms": prediction.latency_ms,
                    "callback_total_ms": callback_ms,
                    "end_to_end_ms": callback_ms,
                    "mask_count": _safe_len(prediction.masks),
                    "box_count": _safe_len(prediction.boxes),
                    "score_max": score_max,
                    **{field: profile.get(field, 0.0) for field in COMPONENT_FIELDS},
                    "other_ms": max(0.0, callback_ms - component_total),
                    **memory,
                }
            )
    finally:
        if writer is not None:
            writer.release()
    return rows, _pred_entries_for_item(item, pred_masks, pred_scores), overlay_video


def _profile_bbox_chain_source(
    args: argparse.Namespace,
    item: dict[str, Any],
    backend: Any,
    torch_module: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    frame_paths = _frame_paths(item, args.max_frames)
    gt_masks = _gt_masks_by_frame(item)
    prompt_type = _resolved_prompt_type(args)
    prompt_frame_index = _initial_prompt_frame_index(prompt_type, gt_masks, len(frame_paths), item)
    prompt_gt = gt_masks.get(prompt_frame_index)
    initial_prompt = _initial_prompt(args, item, prompt_gt, prompt_type)
    state = BBoxChainState(
        initial_prompt=initial_prompt,
        bbox_min_area=args.bbox_min_area,
    )

    writer = None
    overlay_video = ""
    rows = []
    pred_masks: list[np.ndarray | None] = []
    pred_scores: list[float] = []
    try:
        for frame_offset, frame_path in enumerate(frame_paths[prompt_frame_index:], start=prompt_frame_index):
            frame_rgb = _read_rgb(frame_path)
            prompt, prompt_mode = state.next_prompt(frame_offset)
            prediction = Prediction(masks=[], boxes=[], scores=[], latency_ms=0.0, metadata={})
            profile = {}
            callback_start = perf_counter()
            if prompt is not None:
                if torch_module is not None:
                    with component_timer(getattr(backend, "model", None), torch_module) as profile:
                        prediction = backend.predict(frame_rgb, prompt)
                else:
                    prediction = backend.predict(frame_rgb, prompt)
            callback_ms = (perf_counter() - callback_start) * 1000.0
            component_total = sum(profile.values())
            pred_mask = merge_masks(prediction.masks, frame_rgb.shape[:2])
            state.update(prediction.masks, frame_rgb.shape[:2])
            gt_mask = gt_masks.get(frame_offset)
            iou = _mask_iou(pred_mask, gt_mask) if gt_mask is not None else ""
            present_pred = pred_mask is not None and bool(pred_mask.any())
            present_gt = gt_mask is not None and bool(gt_mask.any())
            scores = to_numpy(prediction.scores)
            score_max = float(scores.max()) if scores.size else (1.0 if present_pred else 0.0)
            pred_scores.append(score_max)
            pred_masks.append(pred_mask)
            if args.overlay_root:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                if writer is None:
                    overlay_video, writer = _open_overlay_writer(args.overlay_root, item, frame_bgr.shape, args.overlay_fps)
                writer.write(
                    _overlay_frame(
                        frame_bgr,
                        pred_mask,
                        gt_mask,
                        args.model_id,
                        item,
                        frame_offset,
                        iou,
                        callback_ms,
                        prompt_mode,
                    )
                )
            memory = cuda_memory_mb(torch_module) if torch_module is not None else cuda_memory_mb(None)
            rows.append(
                {
                    "model_id": args.model_id,
                    "backend": args.backend,
                    "stream_mode": args.stream_mode,
                    "source_id": item["source_id"],
                    "video_id": item["video_id"],
                    "category_id": item["category_id"],
                    "frame_index": frame_offset,
                    "prompt_mode": prompt_mode,
                    "prompt": _prompt_text(prompt),
                    "has_gt": gt_mask is not None,
                    "is_positive": bool(item.get("is_positive", True)),
                    "iou": iou,
                    "matched_iou": iou,
                    "presence_correct": present_pred == present_gt,
                    "latency_ms": prediction.latency_ms,
                    "callback_total_ms": callback_ms,
                    "end_to_end_ms": callback_ms,
                    "mask_count": _safe_len(prediction.masks),
                    "box_count": _safe_len(prediction.boxes),
                    "score_max": score_max,
                    **{field: profile.get(field, 0.0) for field in COMPONENT_FIELDS},
                    "other_ms": max(0.0, callback_ms - component_total),
                    **memory,
                }
            )
    finally:
        if writer is not None:
            writer.release()
    return rows, _pred_entries_for_item(item, pred_masks, pred_scores), overlay_video


def _profile_native(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, int], dict[str, float]]:
    if args.backend in {"sam2", "efficient-sam2", "efficienttam"}:
        return _profile_native_sam2(args, rows)
    if args.backend in {"sam3", "sam3p1"}:
        return _profile_native_sam3(args, rows)
    raise ValueError(f"native_video stream mode is not supported for backend={args.backend}")


def _profile_native_sam2(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, int], dict[str, float]]:
    from .profile_sav_video import _build_predictor, _first_mask, _sync

    predictor, torch_module = _build_predictor(args)
    params = parameter_counts(predictor)
    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()
    all_rows = []
    pred_entries = []
    overlays = []
    for item in rows:
        frame_rows, pred_entry, overlay = _profile_native_sam2_source(args, item, predictor, torch_module, _first_mask, _sync)
        all_rows.extend(frame_rows)
        pred_entries.extend(pred_entry)
        if overlay:
            overlays.append(overlay)
    return all_rows, pred_entries, overlays, params, cuda_memory_mb(torch_module)


def _profile_native_sam2_source(
    args: argparse.Namespace,
    item: dict[str, Any],
    predictor: Any,
    torch_module: Any,
    first_mask_fn: Any,
    sync_fn: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    frame_dir = _materialize_frame_dir(item)
    gt_masks = _gt_masks_by_frame(item)
    frame_count = len(_frame_paths(item, args.max_frames))
    prompt_frame_index = _initial_prompt_frame_index("point", gt_masks, frame_count, item)
    point = _mask_centroid(gt_masks[prompt_frame_index])
    rows = []
    pred_masks: list[np.ndarray | None] = []
    pred_scores: list[float] = []
    writer = None
    overlay_video = ""
    try:
        sync_fn(torch_module)
        state = predictor.init_state(video_path=str(frame_dir))
        sync_fn(torch_module)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=prompt_frame_index,
            obj_id=1,
            points=np.asarray([point], dtype=np.float32),
            labels=np.asarray([1], dtype=np.int32),
        )
        max_to_track = (frame_count - prompt_frame_index) if args.max_frames > 0 else None
        iterator = predictor.propagate_in_video(state, start_frame_idx=prompt_frame_index, max_frame_num_to_track=max_to_track)
        for _ in range(max_to_track if max_to_track is not None else len(item["file_names"])):
            start = perf_counter()
            try:
                frame_idx, out_obj_ids, out_mask_logits = next(iterator)
            except StopIteration:
                break
            sync_fn(torch_module)
            latency_ms = (perf_counter() - start) * 1000.0
            frame_idx = int(frame_idx)
            pred_mask = first_mask_fn(out_mask_logits)
            gt_mask = gt_masks.get(frame_idx)
            iou = _mask_iou(pred_mask, gt_mask) if gt_mask is not None else ""
            pred_masks.append(pred_mask)
            pred_scores.append(1.0 if pred_mask is not None and pred_mask.any() else 0.0)
            if args.overlay_root:
                frame_rgb = _read_rgb(_frame_paths(item, 0)[frame_idx])
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                if writer is None:
                    overlay_video, writer = _open_overlay_writer(args.overlay_root, item, frame_bgr.shape, args.overlay_fps)
                writer.write(_overlay_frame(frame_bgr, pred_mask, gt_mask, args.model_id, item, frame_idx, iou, latency_ms, "native"))
            rows.append(_native_row(args, item, frame_idx, pred_mask, gt_mask, iou, latency_ms, len(out_obj_ids), torch_module))
    finally:
        if writer is not None:
            writer.release()
    return rows, _pred_entries_for_item(item, pred_masks, pred_scores), overlay_video


def _profile_native_sam3(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, int], dict[str, float]]:
    from .backends import _import_required, _prepend_repo_path

    _prepend_repo_path(args.external_repo)
    torch_module = _import_required("torch")
    builder = _import_required("sam3.model_builder")
    predictor = builder.build_sam3_predictor(checkpoint_path=args.checkpoint_path, version="sam3.1" if args.backend == "sam3p1" else "sam3")
    params = parameter_counts(getattr(predictor, "model", predictor))
    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()
    all_rows = []
    pred_entries = []
    overlays = []
    for item in rows:
        frame_rows, pred_entry, overlay = _profile_native_sam3_source(args, item, predictor, torch_module)
        all_rows.extend(frame_rows)
        pred_entries.extend(pred_entry)
        if overlay:
            overlays.append(overlay)
    if hasattr(predictor, "shutdown"):
        predictor.shutdown()
    return all_rows, pred_entries, overlays, params, cuda_memory_mb(torch_module)


def _profile_native_sam3_source(args: argparse.Namespace, item: dict[str, Any], predictor: Any, torch_module: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    gt_masks = _gt_masks_by_frame(item)
    response = predictor.handle_request({"type": "start_session", "resource_path": str(_materialize_frame_dir(item))})
    session_id = response["session_id"]
    predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": item.get("text_prompt") or item.get("noun_phrase")})
    rows = []
    pred_masks: list[np.ndarray | None] = []
    pred_scores: list[float] = []
    writer = None
    overlay_video = ""
    try:
        iterator = predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "start_frame_index": 0,
                "max_frame_num_to_track": args.max_frames if args.max_frames > 0 else None,
            }
        )
        for response in iterator:
            start = perf_counter()
            frame_idx = int(response["frame_index"])
            masks = _sam3_output_masks(response.get("outputs", {}), item)
            pred_mask = merge_masks(masks, (int(item["height"]), int(item["width"])))
            latency_ms = (perf_counter() - start) * 1000.0
            gt_mask = gt_masks.get(frame_idx)
            iou = _mask_iou(pred_mask, gt_mask) if gt_mask is not None else ""
            pred_masks.append(pred_mask)
            pred_scores.append(1.0 if pred_mask is not None and pred_mask.any() else 0.0)
            if args.overlay_root:
                frame_rgb = _read_rgb(_frame_paths(item, 0)[frame_idx])
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                if writer is None:
                    overlay_video, writer = _open_overlay_writer(args.overlay_root, item, frame_bgr.shape, args.overlay_fps)
                writer.write(_overlay_frame(frame_bgr, pred_mask, gt_mask, args.model_id, item, frame_idx, iou, latency_ms, "native_text"))
            rows.append(_native_row(args, item, frame_idx, pred_mask, gt_mask, iou, latency_ms, _safe_len(masks), torch_module))
    finally:
        if writer is not None:
            writer.release()
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    return rows, _pred_entries_for_item(item, pred_masks, pred_scores), overlay_video


def _resolved_prompt_type(args: argparse.Namespace) -> str:
    prompt_type = args.prompt_type
    if prompt_type == "auto":
        prompt_type = "point" if args.backend in {"mobilesam", "sam1", "sam2", "efficient-sam2", "efficienttam"} else "text"
    return prompt_type


def _initial_prompt_frame_index(
    prompt_type: str,
    gt_masks: dict[int, np.ndarray],
    frame_count: int,
    item: dict[str, Any],
) -> int:
    if prompt_type == "text":
        return 0
    for frame_idx in sorted(gt_masks):
        if frame_idx >= frame_count:
            break
        mask = gt_masks[frame_idx]
        if mask is not None and bool(mask.any()):
            return frame_idx
    raise ValueError(f"point prompt requested for {item['source_id']} but no GT mask is available in the profiled frames")


def _initial_prompt(args: argparse.Namespace, item: dict[str, Any], gt_mask: np.ndarray | None, prompt_type: str | None = None) -> Prompt:
    prompt_type = prompt_type or _resolved_prompt_type(args)
    if prompt_type == "text":
        return Prompt(text=item.get("text_prompt") or item.get("noun_phrase") or args.prompt)
    if gt_mask is None:
        raise ValueError(f"point prompt requested for {item['source_id']} but no prompt-frame GT mask is available")
    return Prompt(points=[_mask_centroid(gt_mask)], labels=[1])


def _image_per_frame_prompt(
    args: argparse.Namespace,
    item: dict[str, Any],
    gt_mask: np.ndarray | None,
    prompt_type: str,
) -> tuple[Prompt | None, str]:
    if prompt_type == "text":
        return _initial_prompt(args, item, gt_mask, prompt_type), "text"
    if gt_mask is None or not bool(gt_mask.any()):
        return None, "no_prompt"
    return _initial_prompt(args, item, gt_mask, prompt_type), "point"


def _native_row(args: argparse.Namespace, item: dict[str, Any], frame_idx: int, pred_mask: np.ndarray | None, gt_mask: np.ndarray | None, iou: float | str, latency_ms: float, mask_count: int, torch_module: Any) -> dict[str, Any]:
    present_pred = pred_mask is not None and bool(pred_mask.any())
    present_gt = gt_mask is not None and bool(gt_mask.any())
    memory = cuda_memory_mb(torch_module) if torch_module is not None else cuda_memory_mb(None)
    return {
        "model_id": args.model_id,
        "backend": args.backend,
        "stream_mode": args.stream_mode,
        "source_id": item["source_id"],
        "video_id": item["video_id"],
        "category_id": item["category_id"],
        "frame_index": frame_idx,
        "prompt_mode": "native",
        "prompt": item.get("text_prompt", ""),
        "has_gt": gt_mask is not None,
        "is_positive": bool(item.get("is_positive", True)),
        "iou": iou,
        "matched_iou": iou,
        "presence_correct": present_pred == present_gt,
        "latency_ms": latency_ms,
        "callback_total_ms": latency_ms,
        "end_to_end_ms": latency_ms,
        "mask_count": mask_count,
        "box_count": 0,
        "score_max": 1.0 if present_pred else 0.0,
        **{field: 0.0 for field in COMPONENT_FIELDS},
        "other_ms": latency_ms,
        **memory,
    }


def _summary_row(args: argparse.Namespace, rows: list[dict[str, Any]], overlays: list[str], params: dict[str, int], memory: dict[str, float], official_eval_json: str) -> dict[str, Any]:
    ious = [row["iou"] for row in rows if row.get("iou") not in ("", None)]
    callback = [row["callback_total_ms"] for row in rows if row.get("callback_total_ms") not in ("", None)]
    end_to_end = [row["end_to_end_ms"] for row in rows if row.get("end_to_end_ms") not in ("", None)]
    return {
        "model_id": args.model_id,
        "backend": args.backend,
        "stream_mode": args.stream_mode,
        "sources": len({row["source_id"] for row in rows}),
        "eval_start_frame": _eval_start_frame(rows),
        "frames": len(rows),
        "positive_frames": sum(1 for row in rows if row.get("is_positive")),
        "mean_iou": _mean(ious),
        "mask_ap_50_95": _threshold_f1(ious, [x / 100.0 for x in range(50, 100, 5)]),
        "mask_f1_50": _threshold_f1(ious, [0.50]),
        "mask_f1_75": _threshold_f1(ious, [0.75]),
        "presence_accuracy": _mean([1.0 if row.get("presence_correct") else 0.0 for row in rows]),
        "mean_latency_ms": _mean([row["latency_ms"] for row in rows]),
        "p95_latency_ms": _percentile([float(row["latency_ms"]) for row in rows if row.get("latency_ms") not in ("", None)], 0.95),
        "mean_callback_total_ms": _mean(callback),
        "mean_end_to_end_ms": _mean(end_to_end),
        "p95_end_to_end_ms": _percentile([float(value) for value in end_to_end], 0.95),
        "effective_fps": _fps(_mean(end_to_end)),
        "input_fps": args.input_fps,
        "overlay_video_count": len(overlays),
        "pred_json": str(args.pred_json) if args.pred_json else "",
        "official_eval_json": official_eval_json,
        **{f"mean_{field}": _mean([row[field] for row in rows]) for field in COMPONENT_FIELDS},
        "cuda_peak_allocated_mb": memory.get("cuda_peak_allocated_mb", ""),
        "cuda_peak_reserved_mb": memory.get("cuda_peak_reserved_mb", ""),
        "params_total": params.get("params_total", ""),
        "weight_total_bytes": params.get("weight_total_bytes", ""),
    }


def _eval_start_frame(rows: list[dict[str, Any]]) -> int | str:
    frame_indices = [int(row["frame_index"]) for row in rows if row.get("frame_index") not in ("", None)]
    return min(frame_indices) if frame_indices else ""


def _run_official_eval(args: argparse.Namespace, rows: list[dict[str, Any]]) -> str:
    assert args.pred_json is not None and args.gt_annotation_file is not None and args.official_eval_json is not None
    eval_start = _eval_start_frame(rows)
    if eval_start not in ("", 0):
        return f"skipped: eval starts at frame {eval_start}"
    args.official_eval_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(args.external_repo or "external/sam3") / "sam3" / "eval" / "saco_veval_eval.py"),
        "one",
        "--gt_annot_file",
        str(args.gt_annotation_file),
        "--pred_file",
        str(args.pred_json),
        "--eval_res_file",
        str(args.official_eval_json),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return f"failed: {exc}"
    return str(args.official_eval_json)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _frame_paths(item: dict[str, Any], max_frames: int) -> list[Path]:
    media_root = Path(item["media_root"])
    paths = [media_root / file_name for file_name in item["file_names"]]
    if max_frames > 0:
        paths = paths[:max_frames]
    return paths


def _materialize_frame_dir(item: dict[str, Any]) -> Path:
    first = _frame_paths(item, 1)[0]
    return first.parent


def _read_rgb(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"failed to read frame: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _gt_masks_by_frame(item: dict[str, Any]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for ann in item.get("annotations", []):
        for idx, rle in enumerate(ann.get("segmentations", [])):
            mask = _decode_rle(rle)
            if mask is None:
                continue
            if idx in out:
                out[idx] = out[idx] | mask
            else:
                out[idx] = mask
    return out


def _decode_rle(rle: dict[str, Any] | None) -> np.ndarray | None:
    if not rle:
        return None
    counts = rle.get("counts")
    size = rle.get("size")
    if not size:
        return None
    if isinstance(counts, list):
        return _decode_uncompressed_rle(counts, (int(size[0]), int(size[1])))
    try:
        import pycocotools.mask as mask_utils
    except ImportError as exc:
        raise RuntimeError("pycocotools is required to decode compressed SA-Co/VEval RLE masks") from exc
    return mask_utils.decode(rle).astype(bool)


def _decode_uncompressed_rle(counts: list[int], shape: tuple[int, int]) -> np.ndarray:
    total = int(shape[0] * shape[1])
    values = np.zeros(total, dtype=np.uint8)
    index = 0
    value = 0
    for count in counts:
        end = min(total, index + int(count))
        if value == 1:
            values[index:end] = 1
        index = end
        value = 1 - value
    return values.reshape((shape[1], shape[0])).T.astype(bool)


def _encode_rle(mask: np.ndarray | None, shape: tuple[int, int]) -> dict[str, Any]:
    if mask is None:
        return {"size": [shape[0], shape[1]], "counts": [shape[0] * shape[1]]}
    mask = _resize_mask(mask, shape).astype(np.uint8)
    try:
        import pycocotools.mask as mask_utils

        rle = mask_utils.encode(np.asfortranarray(mask))
        counts = rle["counts"]
        if isinstance(counts, bytes):
            rle["counts"] = counts.decode("ascii")
        return rle
    except ImportError:
        flat = mask.T.reshape(-1)
        counts = []
        last = 0
        run = 0
        for value in flat:
            value = int(value)
            if value == last:
                run += 1
            else:
                counts.append(run)
                run = 1
                last = value
        counts.append(run)
        return {"size": [shape[0], shape[1]], "counts": counts}


def _pred_entries_for_item(item: dict[str, Any], masks: list[np.ndarray | None], scores: list[float]) -> list[dict[str, Any]]:
    shape = (int(item["height"]), int(item["width"]))
    bboxes = [_bbox_xywh(mask, shape) for mask in masks]
    areas = [int(_resize_mask(mask, shape).sum()) if mask is not None else 0 for mask in masks]
    return [
        {
            "video_id": int(item["video_id"]),
            "category_id": int(item["category_id"]),
            "bboxes": bboxes,
            "score": max(scores) if scores else 0.0,
            "segmentations": [_encode_rle(mask, shape) for mask in masks],
            "areas": areas,
        }
    ]


def _bbox_xywh(mask: np.ndarray | None, shape: tuple[int, int]) -> list[int]:
    if mask is None:
        return [0, 0, 0, 0]
    mask = _resize_mask(mask, shape)
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def _mask_centroid(mask: np.ndarray | None) -> tuple[float, float]:
    if mask is None:
        raise ValueError("cannot compute centroid for missing mask")
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("cannot compute centroid for empty mask")
    return float(xs.mean()), float(ys.mean())


def _mask_iou(pred: np.ndarray | None, gt: np.ndarray | None) -> float:
    if pred is None or gt is None:
        return 0.0
    pred = _resize_mask(pred, gt.shape)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(intersection / union) if union else 0.0


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape != shape:
        mask = cv2.resize(mask.astype("uint8"), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def _sam3_output_masks(outputs: dict[str, Any], item: dict[str, Any]) -> Any:
    for key in ("out_binary_masks", "pred_masks", "masks"):
        if key in outputs:
            return outputs[key]
    return np.zeros((0, int(item["height"]), int(item["width"])), dtype=np.uint8)


def _open_overlay_writer(overlay_root: Path, item: dict[str, Any], frame_shape: tuple[int, int, int], fps: float) -> tuple[str, cv2.VideoWriter]:
    out_dir = overlay_root / item["source_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "overlay.mp4"
    height, width = frame_shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to create overlay video: {path}")
    return str(path), writer


def _overlay_frame(frame_bgr: np.ndarray, pred_mask: np.ndarray | None, gt_mask: np.ndarray | None, model_id: str, item: dict[str, Any], frame_idx: int, iou: float | str, latency_ms: float, prompt_mode: str, alpha: float = 0.45) -> np.ndarray:
    overlay = frame_bgr.copy()
    if pred_mask is not None:
        pred = _resize_mask(pred_mask, overlay.shape[:2])
        overlay[pred] = (overlay[pred] * (1.0 - alpha) + np.asarray([40, 220, 60]) * alpha).astype(np.uint8)
    if gt_mask is not None:
        gt = _resize_mask(gt_mask, overlay.shape[:2])
        contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (40, 40, 255), 2)
    label = f"{model_id} {item.get('noun_phrase', '')} frame={frame_idx} {prompt_mode} {latency_ms:.1f}ms"
    if iou not in ("", None):
        label += f" IoU={float(iou):.3f}"
    cv2.putText(overlay, label[:120], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, label[:120], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def _threshold_f1(ious: list[Any], thresholds: list[float]) -> float | str:
    numeric = [float(value) for value in ious if value not in ("", None)]
    if not numeric:
        return ""
    values = []
    for threshold in thresholds:
        tp = sum(1 for value in numeric if value >= threshold)
        fp = len(numeric) - tp
        fn = fp
        denom = (2 * tp) + fp + fn
        values.append((2 * tp) / denom if denom else 0.0)
    return mean(values)


def _mean(values: list[Any]) -> float | str:
    numeric = [float(value) for value in values if value not in ("", None)]
    return mean(numeric) if numeric else ""


def _percentile(values: list[float], q: float) -> float | str:
    if not values:
        return ""
    values = sorted(values)
    return values[int((len(values) - 1) * q)]


def _fps(mean_ms: float | str) -> float | str:
    return 1000.0 / mean_ms if isinstance(mean_ms, float) and mean_ms > 0 else ""


def _safe_len(value: object) -> int:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _prompt_text(prompt: Prompt | None) -> str:
    if prompt is None:
        return ""
    if prompt.text:
        return prompt.text
    if prompt.points:
        return json.dumps({"points": prompt.points, "labels": prompt.labels})
    if prompt.boxes:
        return json.dumps({"boxes": prompt.boxes})
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile SA-Co/VEval-SAV stream segmentation with overlay videos.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--backend", choices=["null", "mobilesam", "sam1", "sam2", "efficient-sam2", "efficienttam", "sam3", "sam3p1", "efficientsam3"], required=True)
    parser.add_argument("--stream-mode", choices=["bbox_chain", "text_bbox_chain", "native_video", "image_per_frame"], default="bbox_chain")
    parser.add_argument("--prompt-type", choices=["auto", "point", "text"], default="auto")
    parser.add_argument("--prompt", default="")
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
    parser.add_argument("--bbox-min-area", type=int, default=25)
    parser.add_argument("--input-fps", type=float, default=30.0)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--pred-json", type=Path)
    parser.add_argument("--gt-annotation-file", type=Path)
    parser.add_argument("--official-eval-json", type=Path)
    parser.add_argument("--overlay-root", type=Path)
    parser.add_argument("--overlay-fps", type=float, default=30.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
