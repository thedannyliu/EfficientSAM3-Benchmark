from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Callable

import cv2
import numpy as np

from .backends import BackendConfig, create_backend
from .coco_manifest import ann_to_mask
from .profile_coco import _build_prompt, _prediction_iou, _prompt_modes
from .profile_yolo_coco import (
    _best_box_iou,
    _build_model as _build_yolo_model,
    _extract_detections,
    _filter_detections_by_class,
    _mask_ious,
    _predict_kwargs,
    _set_open_vocab_classes,
)
from .profiling import cuda_memory_mb


FIELDNAMES = [
    "suite",
    "model_id",
    "backend",
    "family",
    "weights",
    "sample_id",
    "iteration",
    "prompt_mode",
    "prompt",
    "image",
    "width",
    "height",
    "read_ms",
    "color_convert_ms",
    "gt_decode_ms",
    "prompt_build_ms",
    "set_classes_ms",
    "predict_wall_ms",
    "predict_cuda_window_ms",
    "predict_torch_cuda_kernel_ms",
    "predict_torch_cpu_self_ms",
    "predict_cpu_gap_ms",
    "postprocess_ms",
    "total_pipeline_ms",
    "pipeline_without_read_ms",
    "mask_count",
    "box_count",
    "score_max",
    "best_iou",
    "merged_iou",
    "best_box_iou",
    "cuda_allocated_mb",
    "cuda_reserved_mb",
    "cuda_peak_allocated_mb",
    "cuda_peak_reserved_mb",
]


def main() -> None:
    args = parse_args()
    summary = profile_pipeline(args)
    print(json.dumps(summary, indent=2))
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def profile_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_manifest(args.manifest)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("manifest did not provide any rows to profile")

    runner = _build_runner(args)
    torch_module = runner.torch_module
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()

    cached_frames = _preload_frames(rows) if args.input_mode == "preload" else {}
    _warmup(args, runner, rows[0], cached_frames)

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    with args.csv_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for iteration in range(args.repeat):
            for item in rows:
                frame_bgr, read_ms = _load_frame(item, cached_frames)
                start = perf_counter()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                color_convert_ms = _elapsed_ms(start)

                gt_mask = None
                gt_decode_ms = 0.0
                if args.with_gt:
                    start = perf_counter()
                    gt_mask = ann_to_mask(item, int(item["width"]), int(item["height"]))
                    gt_decode_ms = _elapsed_ms(start)
                    if gt_mask is None:
                        raise RuntimeError(f"failed to decode ground-truth mask for sample {item['sample_id']}")

                row = runner.profile_item(item, frame_rgb, gt_mask)
                total_pipeline_ms = (
                    read_ms
                    + color_convert_ms
                    + gt_decode_ms
                    + row["prompt_build_ms"]
                    + row["set_classes_ms"]
                    + row["predict_wall_ms"]
                    + row["postprocess_ms"]
                )
                memory = cuda_memory_mb(torch_module)
                full_row = {
                    "suite": args.suite,
                    "model_id": args.model_id,
                    "backend": getattr(args, "backend", ""),
                    "family": getattr(args, "family", ""),
                    "weights": getattr(args, "weights", ""),
                    "sample_id": item["sample_id"],
                    "iteration": iteration,
                    "image": item["image_path"],
                    "width": frame_rgb.shape[1],
                    "height": frame_rgb.shape[0],
                    "read_ms": read_ms,
                    "color_convert_ms": color_convert_ms,
                    "gt_decode_ms": gt_decode_ms,
                    "total_pipeline_ms": total_pipeline_ms,
                    "pipeline_without_read_ms": total_pipeline_ms - read_ms,
                    **row,
                    **memory,
                }
                writer.writerow(full_row)
                output_rows.append(full_row)

    return _summarize(args, output_rows, torch_module)


class _Runner:
    torch_module: Any

    def profile_item(self, item: dict[str, Any], frame_rgb: np.ndarray, gt_mask: np.ndarray | None) -> dict[str, Any]:
        raise NotImplementedError

    def warmup_item(self, item: dict[str, Any], frame_rgb: np.ndarray) -> None:
        self.profile_item(item, frame_rgb, None)


class _SamRunner(_Runner):
    def __init__(self, args: argparse.Namespace) -> None:
        self.backend = create_backend(
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
                mobile_sam_model_type=args.mobile_sam_model_type,
            )
        )
        self.torch_module = getattr(self.backend, "torch", None)
        prompt_modes = _prompt_modes(args.prompt_mode)
        if args.backend in {"sam2", "efficient-sam2", "efficienttam", "mobilesam"}:
            if args.prompt_mode == "text":
                raise ValueError(f"{args.backend} supports point/box prompts in this benchmark, not text prompts")
            if args.prompt_mode == "both":
                prompt_modes = ["point"]
            if args.prompt_mode == "all":
                prompt_modes = ["point", "box"]
        self.prompt_modes = prompt_modes
        self.use_torch_profiler = args.with_torch_profiler

    def profile_item(self, item: dict[str, Any], frame_rgb: np.ndarray, gt_mask: np.ndarray | None) -> dict[str, Any]:
        rows = []
        for prompt_mode in self.prompt_modes:
            start = perf_counter()
            prompt = _build_prompt(item, prompt_mode)
            prompt_build_ms = _elapsed_ms(start)

            prediction, timing = _time_prediction(
                self.torch_module,
                lambda: self.backend.predict(frame_rgb, prompt),
                self.use_torch_profiler,
            )

            start = perf_counter()
            best_iou, merged_iou = _prediction_iou(prediction.masks, gt_mask) if gt_mask is not None else ("", "")
            postprocess_ms = _elapsed_ms(start)
            scores = _to_numpy(getattr(prediction, "scores", None))
            rows.append(
                {
                    "prompt_mode": prompt_mode,
                    "prompt": prompt.text or "",
                    "prompt_build_ms": prompt_build_ms,
                    "set_classes_ms": 0.0,
                    "predict_wall_ms": timing["wall_ms"],
                    "predict_cuda_window_ms": timing["cuda_window_ms"],
                    "predict_torch_cuda_kernel_ms": timing["torch_cuda_kernel_ms"],
                    "predict_torch_cpu_self_ms": timing["torch_cpu_self_ms"],
                    "predict_cpu_gap_ms": _gap_ms(timing["wall_ms"], timing["best_gpu_ms"]),
                    "postprocess_ms": postprocess_ms,
                    "mask_count": _safe_len(getattr(prediction, "masks", None)),
                    "box_count": _safe_len(getattr(prediction, "boxes", None)),
                    "score_max": float(scores.max()) if scores.size else "",
                    "best_iou": best_iou,
                    "merged_iou": merged_iou,
                    "best_box_iou": "",
                }
            )
        if len(rows) != 1:
            return _collapse_prompt_rows(rows)
        return rows[0]


class _YoloRunner(_Runner):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model, self.torch_module = _build_yolo_model(args)
        self.current_prompt: str | None = None

    def profile_item(self, item: dict[str, Any], frame_rgb: np.ndarray, gt_mask: np.ndarray | None) -> dict[str, Any]:
        start = perf_counter()
        prompt = str(item.get("text_prompt") or item["category_name"])
        prompt_build_ms = _elapsed_ms(start)

        set_classes_ms = 0.0
        if self.args.family == "yoloe-seg" and prompt != self.current_prompt:
            start = perf_counter()
            _set_open_vocab_classes(self.model, [prompt])
            _sync(self.torch_module)
            set_classes_ms = _elapsed_ms(start)
            self.current_prompt = prompt

        predict_kwargs = _predict_kwargs(self.args)
        results, timing = _time_prediction(
            self.torch_module,
            lambda: self.model.predict(frame_rgb, **predict_kwargs),
            self.args.with_torch_profiler,
        )

        start = perf_counter()
        result = results[0] if results else None
        all_detections = _extract_detections(result, frame_rgb.shape[:2])
        detections = all_detections
        if self.args.family == "yolo-seg":
            detections = _filter_detections_by_class(all_detections, prompt)
        detections = detections[: self.args.max_det_for_iou] if self.args.max_det_for_iou > 0 else detections
        best_iou, merged_iou = _mask_ious([det["mask"] for det in detections], gt_mask) if gt_mask is not None else ("", "")
        best_box_iou = _best_box_iou([det["box"] for det in detections], item) if gt_mask is not None else ""
        postprocess_ms = _elapsed_ms(start)
        scores = [det["score"] for det in detections if det["score"] != ""]

        return {
            "prompt_mode": "text",
            "prompt": prompt,
            "prompt_build_ms": prompt_build_ms,
            "set_classes_ms": set_classes_ms,
            "predict_wall_ms": timing["wall_ms"],
            "predict_cuda_window_ms": timing["cuda_window_ms"],
            "predict_torch_cuda_kernel_ms": timing["torch_cuda_kernel_ms"],
            "predict_torch_cpu_self_ms": timing["torch_cpu_self_ms"],
            "predict_cpu_gap_ms": _gap_ms(timing["wall_ms"], timing["best_gpu_ms"]),
            "postprocess_ms": postprocess_ms,
            "mask_count": sum(1 for det in detections if det["mask"] is not None),
            "box_count": sum(1 for det in detections if det["box"] is not None),
            "score_max": max(scores) if scores else "",
            "best_iou": best_iou,
            "merged_iou": merged_iou,
            "best_box_iou": best_box_iou,
        }


def _build_runner(args: argparse.Namespace) -> _Runner:
    if args.suite == "sam":
        return _SamRunner(args)
    if args.suite == "yolo":
        return _YoloRunner(args)
    raise ValueError(f"unsupported suite: {args.suite}")


def _warmup(
    args: argparse.Namespace,
    runner: _Runner,
    item: dict[str, Any],
    cached_frames: dict[str, np.ndarray],
) -> None:
    if args.warmup <= 0:
        return
    frame_bgr, _ = _load_frame(item, cached_frames)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    for _ in range(args.warmup):
        runner.warmup_item(item, frame_rgb)


def _preload_frames(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    frames = {}
    for item in rows:
        path = str(item["image_path"])
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"failed to read image: {path}")
        frames[path] = frame
    return frames


def _load_frame(item: dict[str, Any], cached_frames: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    path = str(item["image_path"])
    if path in cached_frames:
        return cached_frames[path], 0.0
    start = perf_counter()
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    read_ms = _elapsed_ms(start)
    if frame is None:
        raise RuntimeError(f"failed to read image: {path}")
    return frame, read_ms


def _time_prediction(torch_module: Any, func: Callable[[], Any], with_torch_profiler: bool) -> tuple[Any, dict[str, Any]]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        start = perf_counter()
        result = func()
        return result, {
            "wall_ms": _elapsed_ms(start),
            "cuda_window_ms": "",
            "torch_cuda_kernel_ms": "",
            "torch_cpu_self_ms": "",
            "best_gpu_ms": "",
        }

    cuda.synchronize()
    start_event = cuda.Event(enable_timing=True)
    end_event = cuda.Event(enable_timing=True)
    start_event.record()
    start = perf_counter()
    if with_torch_profiler:
        result, torch_cuda_kernel_ms, torch_cpu_self_ms = _time_with_torch_profiler(torch_module, func)
    else:
        result = func()
        torch_cuda_kernel_ms = ""
        torch_cpu_self_ms = ""
    end_event.record()
    end_event.synchronize()
    wall_ms = _elapsed_ms(start)
    cuda_window_ms = float(start_event.elapsed_time(end_event))
    best_gpu_ms = torch_cuda_kernel_ms if torch_cuda_kernel_ms != "" else cuda_window_ms
    return result, {
        "wall_ms": wall_ms,
        "cuda_window_ms": cuda_window_ms,
        "torch_cuda_kernel_ms": torch_cuda_kernel_ms,
        "torch_cpu_self_ms": torch_cpu_self_ms,
        "best_gpu_ms": best_gpu_ms,
    }


def _time_with_torch_profiler(torch_module: Any, func: Callable[[], Any]) -> tuple[Any, float | str, float | str]:
    profiler_mod = getattr(torch_module, "profiler", None)
    if profiler_mod is None:
        result = func()
        return result, "", ""
    activities = [profiler_mod.ProfilerActivity.CPU]
    if torch_module.cuda.is_available():
        activities.append(profiler_mod.ProfilerActivity.CUDA)
    with profiler_mod.profile(activities=activities) as prof:
        result = func()
        _sync(torch_module)
    cuda_us = 0.0
    cpu_us = 0.0
    for event in prof.key_averages():
        cuda_us += float(getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0)) or 0.0)
        cpu_us += float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
    return result, cuda_us / 1000.0 if cuda_us > 0 else "", cpu_us / 1000.0 if cpu_us > 0 else ""


def _collapse_prompt_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_fields = [
        "prompt_build_ms",
        "set_classes_ms",
        "predict_wall_ms",
        "postprocess_ms",
        "mask_count",
        "box_count",
    ]
    out = dict(rows[0])
    out["prompt_mode"] = "+".join(str(row["prompt_mode"]) for row in rows)
    out["prompt"] = "+".join(str(row["prompt"]) for row in rows if row["prompt"])
    for field in numeric_fields:
        out[field] = sum(float(row[field]) for row in rows)
    window_values = [row["predict_cuda_window_ms"] for row in rows if row["predict_cuda_window_ms"] != ""]
    kernel_values = [row["predict_torch_cuda_kernel_ms"] for row in rows if row["predict_torch_cuda_kernel_ms"] != ""]
    cpu_values = [row["predict_torch_cpu_self_ms"] for row in rows if row["predict_torch_cpu_self_ms"] != ""]
    out["predict_cuda_window_ms"] = sum(float(value) for value in window_values) if window_values else ""
    out["predict_torch_cuda_kernel_ms"] = sum(float(value) for value in kernel_values) if kernel_values else ""
    out["predict_torch_cpu_self_ms"] = sum(float(value) for value in cpu_values) if cpu_values else ""
    best_gpu_ms = out["predict_torch_cuda_kernel_ms"] or out["predict_cuda_window_ms"]
    out["predict_cpu_gap_ms"] = _gap_ms(out["predict_wall_ms"], best_gpu_ms)
    for field in ["score_max", "best_iou", "merged_iou", "best_box_iou"]:
        values = [row[field] for row in rows if row[field] != ""]
        out[field] = max(float(value) for value in values) if values else ""
    return out


def _summarize(args: argparse.Namespace, rows: list[dict[str, Any]], torch_module: Any) -> dict[str, Any]:
    numeric = {
        field: _numeric_values(rows, field)
        for field in [
            "read_ms",
            "color_convert_ms",
            "gt_decode_ms",
            "prompt_build_ms",
            "set_classes_ms",
            "predict_wall_ms",
            "predict_cuda_window_ms",
            "predict_torch_cuda_kernel_ms",
            "predict_torch_cpu_self_ms",
            "predict_cpu_gap_ms",
            "postprocess_ms",
            "total_pipeline_ms",
            "pipeline_without_read_ms",
        ]
    }
    mean_total = _mean(numeric["total_pipeline_ms"])
    mean_predict_wall = _mean(numeric["predict_wall_ms"])
    mean_cuda_window = _mean(numeric["predict_cuda_window_ms"])
    mean_torch_kernel = _mean(numeric["predict_torch_cuda_kernel_ms"])
    mean_gpu_for_hint = mean_torch_kernel if mean_torch_kernel != "" else mean_cuda_window
    mean_non_predict = mean_total - mean_predict_wall if mean_total != "" and mean_predict_wall != "" else ""
    summary = {
        "suite": args.suite,
        "model_id": args.model_id,
        "manifest": str(args.manifest),
        "csv": str(args.csv_output),
        "input_mode": args.input_mode,
        "with_gt": args.with_gt,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "samples": len({row["sample_id"] for row in rows}),
        "rows": len(rows),
        "cuda_available": bool(torch_module is not None and torch_module.cuda.is_available()),
        "mean_total_pipeline_ms": mean_total,
        "p50_total_pipeline_ms": _median(numeric["total_pipeline_ms"]),
        "effective_pipeline_fps": _fps(mean_total),
        "mean_pipeline_without_read_ms": _mean(numeric["pipeline_without_read_ms"]),
        "mean_predict_wall_ms": mean_predict_wall,
        "effective_predict_wall_fps": _fps(mean_predict_wall),
        "mean_predict_cuda_window_ms": mean_cuda_window,
        "effective_cuda_window_fps": _fps(mean_cuda_window),
        "mean_predict_torch_cuda_kernel_ms": mean_torch_kernel,
        "effective_torch_cuda_kernel_fps": _fps(mean_torch_kernel),
        "mean_predict_torch_cpu_self_ms": _mean(numeric["predict_torch_cpu_self_ms"]),
        "mean_predict_cpu_gap_ms": _mean(numeric["predict_cpu_gap_ms"]),
        "mean_non_predict_pipeline_ms": mean_non_predict,
        "mean_read_ms": _mean(numeric["read_ms"]),
        "mean_color_convert_ms": _mean(numeric["color_convert_ms"]),
        "mean_gt_decode_ms": _mean(numeric["gt_decode_ms"]),
        "mean_prompt_build_ms": _mean(numeric["prompt_build_ms"]),
        "mean_set_classes_ms": _mean(numeric["set_classes_ms"]),
        "mean_postprocess_ms": _mean(numeric["postprocess_ms"]),
    }
    if mean_torch_kernel != "":
        summary["gpu_time_source"] = "torch_profiler_cuda_kernel"
    elif mean_cuda_window != "":
        summary["gpu_time_source"] = "cuda_window"
    else:
        summary["gpu_time_source"] = ""
    summary["gpu_time_fraction_of_pipeline"] = _ratio(mean_gpu_for_hint, mean_total)
    summary["gpu_time_fraction_of_predict"] = _ratio(mean_gpu_for_hint, mean_predict_wall)
    summary["bottleneck_hint"] = _bottleneck_hint(summary)
    summary["next_checks"] = _next_checks(summary)
    return summary


def _bottleneck_hint(summary: dict[str, Any]) -> str:
    if (
        not summary["cuda_available"]
        or (summary["mean_predict_cuda_window_ms"] == "" and summary["mean_predict_torch_cuda_kernel_ms"] == "")
    ):
        return "cpu_or_no_cuda_timing"
    gpu_fraction = float(summary["gpu_time_fraction_of_pipeline"])
    predict_gap = float(summary["mean_predict_cpu_gap_ms"])
    total = float(summary["mean_total_pipeline_ms"])
    non_predict = float(summary["mean_non_predict_pipeline_ms"])
    postprocess = float(summary["mean_postprocess_ms"])
    if gpu_fraction >= 0.70:
        return "gpu_bound_compute_or_memory"
    if predict_gap >= max(10.0, total * 0.25):
        return "cpu_wrapper_sync_or_copy_bound"
    if postprocess >= max(10.0, total * 0.25):
        return "postprocess_bound"
    if non_predict >= max(10.0, total * 0.30):
        return "preprocess_or_fixed_pipeline_bound"
    return "mixed_or_kernel_launch_bound"


def _next_checks(summary: dict[str, Any]) -> list[str]:
    hint = summary["bottleneck_hint"]
    if hint == "gpu_bound_compute_or_memory":
        return [
            "Run Nsight Compute to separate SM/Tensor utilization from DRAM bandwidth.",
            "Repeat a resolution sweep; strong pixel scaling points toward image-path or bandwidth pressure.",
        ]
    if hint == "cpu_wrapper_sync_or_copy_bound":
        return [
            "Inspect Nsight Systems for CPU gaps, cudaMemcpy, and per-frame synchronize calls.",
            "Compare PyTorch with TensorRT for YOLO to isolate framework and Python overhead.",
        ]
    if hint == "postprocess_bound":
        return [
            "Move mask resize/filtering/NMS-like work off CPU or reduce returned detections.",
            "Run with --with-gt disabled when measuring deployment latency.",
        ]
    if hint == "preprocess_or_fixed_pipeline_bound":
        return [
            "Run --input-mode preload to remove image decode from the timed path.",
            "Move resize/color conversion/normalization to the GPU or hardware pipeline.",
        ]
    return [
        "Run batch/resolution sweeps on the same command line before changing code.",
        "Use Nsight Systems when wall time and GPU timing disagree.",
    ]


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value in ("", None):
            continue
        values.append(float(value))
    return values


def _mean(values: list[float]) -> float | str:
    return mean(values) if values else ""


def _median(values: list[float]) -> float | str:
    return median(values) if values else ""


def _ratio(numerator: Any, denominator: Any) -> float | str:
    if numerator == "" or denominator == "":
        return ""
    denominator = float(denominator)
    return float(numerator) / denominator if denominator > 0 else ""


def _fps(mean_ms: Any) -> float | str:
    return 1000.0 / float(mean_ms) if mean_ms not in ("", None) and float(mean_ms) > 0 else ""


def _gap_ms(wall_ms: float, gpu_ms: float | str) -> float | str:
    if gpu_ms == "":
        return ""
    return max(0.0, wall_ms - float(gpu_ms))


def _elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000.0


def _safe_len(value: object) -> int:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _sync(torch_module: Any) -> None:
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile fixed pipeline costs versus model GPU time on COCO images.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Profile only the first N manifest rows; 0 means all rows.")
    parser.add_argument("--suite", choices=["sam", "yolo"], required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--input-mode", choices=["read-each-time", "preload"], default="read-each-time")
    parser.add_argument("--with-gt", action="store_true", help="Decode GT masks and compute IoU inside the timed path.")
    parser.add_argument(
        "--with-torch-profiler",
        action="store_true",
        help="Use PyTorch profiler around predict() to estimate CUDA kernel time on short diagnostic runs.",
    )
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)

    parser.add_argument(
        "--backend",
        choices=["null", "sam3", "efficientsam3", "sam2", "efficient-sam2", "efficienttam", "mobilesam"],
        default="null",
        help="SAM-family backend. Used when --suite sam.",
    )
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--model-config")
    parser.add_argument("--external-repo")
    parser.add_argument("--backbone-type", default="efficientvit")
    parser.add_argument("--model-name", default="b0")
    parser.add_argument("--text-encoder-type")
    parser.add_argument("--text-encoder-context-length", type=int, default=77)
    parser.add_argument("--text-encoder-pos-embed-table-size", type=int)
    parser.add_argument("--interpolate-pos-embed", action="store_true")
    parser.add_argument("--mobile-sam-model-type", default="vit_t")
    parser.add_argument("--prompt-mode", choices=["text", "point", "box", "both", "all"], default="point")

    parser.add_argument("--family", choices=["yoloe-seg", "yolo-seg"], default="yolo-seg")
    parser.add_argument("--weights", default="")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--max-det-for-iou", type=int, default=100)
    parser.add_argument("--agnostic-nms", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
