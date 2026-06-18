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

from .backends import _import_required, _prepend_repo_path
from .overlay import overlay_prediction


POINT_COUNTS = (1, 2, 3, 5, 10, 15)
SUMMARY_FIELDS = [
    "suite",
    "model_id",
    "target_count",
    "images",
    "mean_model_ms",
    "p50_model_ms",
    "p95_model_ms",
    "mean_fps",
    "mean_image_encoder_ms",
    "mean_prompt_decode_ms",
    "mean_mask_count",
]


def main() -> None:
    args = parse_args()
    rows = profile_multi_prompt_image(args)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = summarize_rows(rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps({"rows": len(rows), "csv": str(args.csv_output), "summary": str(args.summary_output)}, indent=2))


def profile_multi_prompt_image(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = _load_samples(args)
    rows: list[dict[str, Any]] = []
    if args.suite in {"mobilesam", "all"}:
        rows.extend(_profile_mobilesam(args, samples))
    if args.suite in {"sam3_text", "all"}:
        rows.extend(_profile_sam3_text(args, samples))
    return rows


def _profile_mobilesam(args: argparse.Namespace, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    torch = _import_required("torch")
    _prepend_repo_path(args.mobilesam_external_repo)
    mobile_sam = _import_required("mobile_sam")
    model = mobile_sam.sam_model_registry[args.mobilesam_model_type](checkpoint=args.mobilesam_checkpoint)
    if args.device and hasattr(model, "to"):
        model.to(device=args.device)
    if hasattr(model, "eval"):
        model.eval()
    predictor = mobile_sam.SamPredictor(model)

    rows: list[dict[str, Any]] = []
    counts = _parse_counts(args.point_counts)
    for sample in samples:
        frame_bgr = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        all_points = _grid_points(width, height, max(counts))

        for _ in range(args.warmup):
            _mobilesam_multi_point_predict(torch, predictor, frame_rgb, all_points[: min(counts)])

        for count in counts:
            points = all_points[:count]
            result = _mobilesam_multi_point_predict(torch, predictor, frame_rgb, points)
            overlay_path = ""
            if args.overlay_root and len(rows) < args.max_overlays:
                overlay_path = str(_write_overlay(args.overlay_root, "mobilesam", sample, count, frame_rgb, result["masks"]))
            rows.append(
                {
                    "suite": "mobilesam_points",
                    "model_id": "mobilesam_vit_t",
                    "backend": "mobilesam",
                    "image_id": sample["image_id"],
                    "image_path": str(sample["image_path"]),
                    "text_prompt": sample.get("text_prompt", ""),
                    "target_count": count,
                    "point_count": count,
                    "points_json": json.dumps(points),
                    "model_ms": result["model_ms"],
                    "image_encoder_ms": result["image_encoder_ms"],
                    "prompt_decode_ms": result["prompt_decode_ms"],
                    "mask_count": result["mask_count"],
                    "overlay": overlay_path,
                }
            )
    return rows


def _mobilesam_multi_point_predict(torch: Any, predictor: Any, frame_rgb: np.ndarray, points: list[tuple[float, float]]) -> dict[str, Any]:
    _sync(torch)
    start = perf_counter()
    with torch.inference_mode():
        image_start = perf_counter()
        predictor.set_image(frame_rgb)
        _sync(torch)
        image_encoder_ms = (perf_counter() - image_start) * 1000.0
        masks = []
        decode_ms = 0.0
        for point in points:
            point_coords = np.asarray([point], dtype=np.float32)
            point_labels = np.asarray([1], dtype=np.int32)
            decode_start = perf_counter()
            mask, _scores, _logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
            _sync(torch)
            decode_ms += (perf_counter() - decode_start) * 1000.0
            masks.append(mask)
    _sync(torch)
    return {
        "model_ms": (perf_counter() - start) * 1000.0,
        "image_encoder_ms": image_encoder_ms,
        "prompt_decode_ms": decode_ms,
        "mask_count": len(masks),
        "masks": _flatten_masks(masks),
    }


def _profile_sam3_text(args: argparse.Namespace, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id, version, checkpoint in (
        ("sam3_ref_native_text_image", "sam3", args.sam3_checkpoint),
        ("sam3p1_ref_native_text_image", "sam3.1", args.sam3p1_checkpoint),
    ):
        if checkpoint and not Path(checkpoint).exists() and args.skip_missing:
            continue
        predictor, torch = _build_sam3_predictor(args, checkpoint, version)
        for sample in samples:
            frame_bgr = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            text_prompt = args.text_prompt or sample.get("text_prompt") or sample.get("noun_phrase") or "monitor"
            frame_dir = args.work_dir / model_id / sample["image_id"]
            _write_single_frame(frame_dir, frame_bgr)
            for _ in range(args.warmup):
                _sam3_single_frame_text_predict(torch, predictor, frame_dir, text_prompt, version)
            result = _sam3_single_frame_text_predict(torch, predictor, frame_dir, text_prompt, version)
            overlay_path = ""
            if args.overlay_root and len(rows) < args.max_overlays:
                overlay_path = str(_write_overlay(args.overlay_root, model_id, sample, 1, frame_rgb, result["masks"]))
            rows.append(
                {
                    "suite": "sam3_text",
                    "model_id": model_id,
                    "backend": version,
                    "image_id": sample["image_id"],
                    "image_path": str(sample["image_path"]),
                    "text_prompt": text_prompt,
                    "target_count": sample.get("num_masklets", ""),
                    "point_count": "",
                    "model_ms": result["model_ms"],
                    "start_session_ms": result["start_session_ms"],
                    "add_prompt_ms": result["add_prompt_ms"],
                    "propagate_ms": result["propagate_ms"],
                    "mask_count": result["mask_count"],
                    "overlay": overlay_path,
                }
            )
    return rows


def _build_sam3_predictor(args: argparse.Namespace, checkpoint: str, version: str) -> tuple[Any, Any]:
    _prepend_repo_path(args.sam3_external_repo)
    torch = _import_required("torch")
    builder = _import_required("sam3.model_builder")
    predictor = builder.build_sam3_predictor(
        checkpoint_path=checkpoint,
        version=version,
        warm_up=False,
        async_loading_frames=False,
    )
    return predictor, torch


def _sam3_single_frame_text_predict(torch: Any, predictor: Any, frame_dir: Path, text_prompt: str, version: str) -> dict[str, Any]:
    session_id = None
    _sync(torch)
    start = perf_counter()
    try:
        start_session_start = perf_counter()
        response = predictor.handle_request({"type": "start_session", "resource_path": str(frame_dir)})
        _sync(torch)
        start_session_ms = (perf_counter() - start_session_start) * 1000.0
        session_id = response["session_id"]

        add_prompt_start = perf_counter()
        predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": text_prompt})
        _sync(torch)
        add_prompt_ms = (perf_counter() - add_prompt_start) * 1000.0

        propagate_start = perf_counter()
        responses = list(
            predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "start_frame_index": 0,
                    "max_frame_num_to_track": 1,
                }
            )
        )
        _sync(torch)
        propagate_ms = (perf_counter() - propagate_start) * 1000.0
    finally:
        if session_id is not None:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
    masks = _sam3_response_masks(responses[-1] if responses else {}, frame_dir)
    return {
        "model_ms": (perf_counter() - start) * 1000.0,
        "start_session_ms": start_session_ms,
        "add_prompt_ms": add_prompt_ms,
        "propagate_ms": propagate_ms,
        "mask_count": _safe_len(masks),
        "masks": masks,
        "version": version,
    }


def _sam3_response_masks(response: dict[str, Any], frame_dir: Path) -> Any:
    outputs = response.get("outputs", response)
    for key in ("out_binary_masks", "pred_masks", "masks"):
        if key in outputs:
            return outputs[key]
    first = next(frame_dir.glob("*.jpg"), None)
    if first is not None:
        frame = cv2.imread(str(first), cv2.IMREAD_COLOR)
        if frame is not None:
            return np.zeros((0, frame.shape[0], frame.shape[1]), dtype=np.uint8)
    return np.asarray([])


def _load_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.image_dir:
        paths = sorted(path for path in args.image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        return [{"image_id": path.stem, "image_path": path, "text_prompt": args.text_prompt or "monitor"} for path in paths[: args.image_count]]
    rows = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    samples = []
    for row in rows:
        media_root = Path(row["media_root"])
        file_names = row.get("file_names", [])
        if not file_names:
            continue
        image_path = media_root / file_names[0]
        if image_path.exists():
            samples.append(
                {
                    "image_id": str(row.get("source_id") or row.get("video_name") or len(samples)),
                    "image_path": image_path,
                    "text_prompt": row.get("text_prompt") or row.get("noun_phrase") or args.text_prompt or "monitor",
                    "noun_phrase": row.get("noun_phrase", ""),
                    "num_masklets": int(row.get("num_masklets", 0)),
                }
            )
        if len(samples) >= args.image_count:
            break
    if not samples:
        raise RuntimeError(f"no readable images found from {args.manifest}")
    return samples


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["suite"]), str(row["model_id"]), str(row.get("target_count", ""))), []).append(row)
    summary = []
    for (suite, model_id, target_count), group in sorted(groups.items()):
        model_ms = _float_values(group, "model_ms")
        image_ms = _float_values(group, "image_encoder_ms")
        decode_ms = _float_values(group, "prompt_decode_ms")
        mask_counts = _float_values(group, "mask_count")
        mean_model = mean(model_ms) if model_ms else 0.0
        summary.append(
            {
                "suite": suite,
                "model_id": model_id,
                "target_count": target_count,
                "images": len(group),
                "mean_model_ms": mean_model,
                "p50_model_ms": _percentile(model_ms, 0.50),
                "p95_model_ms": _percentile(model_ms, 0.95),
                "mean_fps": 1000.0 / mean_model if mean_model > 0 else "",
                "mean_image_encoder_ms": mean(image_ms) if image_ms else "",
                "mean_prompt_decode_ms": mean(decode_ms) if decode_ms else "",
                "mean_mask_count": mean(mask_counts) if mask_counts else "",
            }
        )
    return summary


def _parse_counts(value: str) -> list[int]:
    counts = [int(item) for item in value.replace(",", " ").split()]
    if not counts or min(counts) <= 0:
        raise ValueError("--point-counts must contain positive integers")
    return counts


def _grid_points(width: int, height: int, count: int) -> list[tuple[float, float]]:
    cols = int(np.ceil(np.sqrt(count * width / max(height, 1))))
    rows = int(np.ceil(count / max(cols, 1)))
    xs = np.linspace(0.18 * width, 0.82 * width, cols)
    ys = np.linspace(0.18 * height, 0.82 * height, rows)
    points = [(float(x), float(y)) for y in ys for x in xs]
    return points[:count]


def _flatten_masks(masks: list[Any]) -> np.ndarray:
    arrays = []
    for mask in masks:
        array = np.asarray(mask)
        if array.ndim == 2:
            array = array[None, ...]
        arrays.append(array)
    if not arrays:
        return np.asarray([])
    return np.concatenate(arrays, axis=0)


def _write_single_frame(frame_dir: Path, frame_bgr: np.ndarray) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("*.jpg"):
        old.unlink()
    cv2.imwrite(str(frame_dir / "000000.jpg"), frame_bgr)


def _write_overlay(root: Path, model_id: str, sample: dict[str, Any], count: int, frame_rgb: np.ndarray, masks: Any) -> Path:
    path = root / model_id / f"{sample['image_id']}_count{count}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_prediction(frame_rgb, masks)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return path


def _sync(torch: Any) -> None:
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.synchronize()


def _safe_len(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def _float_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value in ("", None):
            continue
        values.append(float(value))
    return values


def _percentile(values: list[float], q: float) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * q)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thor multi-prompt single-image latency benchmark.")
    parser.add_argument("--suite", choices=["mobilesam", "sam3_text", "all"], default="all")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/saco_veval_sav_fixed20.jsonl"))
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--image-count", type=int, default=10)
    parser.add_argument("--point-counts", default="1,2,3,5,10,15")
    parser.add_argument("--text-prompt", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--mobilesam-checkpoint", default="checkpoints/mobilesam/mobile_sam.pt")
    parser.add_argument("--mobilesam-external-repo", default="external/MobileSAM")
    parser.add_argument("--mobilesam-model-type", default="vit_t")
    parser.add_argument("--sam3-checkpoint", default="checkpoints/sam3/sam3.pt")
    parser.add_argument("--sam3p1-checkpoint", default="checkpoints/sam3p1/sam3.1_multiplex.pt")
    parser.add_argument("--sam3-external-repo", default="external/sam3")
    parser.add_argument("--work-dir", type=Path, default=Path("results/thor/multi_prompt_image/work"))
    parser.add_argument("--csv-output", type=Path, default=Path("results/thor/multi_prompt_image/frames.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/thor/multi_prompt_image/summary.csv"))
    parser.add_argument("--overlay-root", type=Path)
    parser.add_argument("--max-overlays", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    main()
