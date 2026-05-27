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
from .overlay import overlay_prediction, to_numpy
from .profiling import component_timer, cuda_memory_mb, parameter_counts


FIELDNAMES = [
    "model_id",
    "backend",
    "video",
    "frame_index",
    "prompt",
    "width",
    "height",
    "mask_count",
    "score_max",
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
]


def main() -> None:
    args = parse_args()
    summary = profile_video(args)
    print(json.dumps(summary, indent=2))


def profile_video(args: argparse.Namespace) -> dict[str, Any]:
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
        )
    )
    prompt = Prompt(text=args.prompt)
    torch_module = getattr(backend, "torch", None)
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()
    params = parameter_counts(getattr(backend, "model", None))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")

    output_video = None
    writer = None
    if args.overlay_output:
        args.overlay_output.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.overlay_output), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"failed to create overlay video: {args.overlay_output}")
        output_video = str(args.overlay_output)

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    latencies: list[float] = []
    with args.csv_output.open("w", newline="", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        csv_writer.writeheader()
        try:
            frame_index = 0
            while frame_index < args.max_frames:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if frame_index % args.frame_stride != 0:
                    frame_index += 1
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
                row = {
                    "model_id": args.model_id,
                    "backend": args.backend,
                    "video": str(args.video),
                    "frame_index": frame_index,
                    "prompt": args.prompt,
                    "width": frame_rgb.shape[1],
                    "height": frame_rgb.shape[0],
                    "mask_count": _safe_len(prediction.masks),
                    "score_max": float(scores.max()) if scores.size else "",
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
                    **memory,
                    **params,
                }
                csv_writer.writerow(row)
                latencies.append(total_ms)

                if writer is not None:
                    overlay_rgb = overlay_prediction(
                        frame_rgb,
                        prediction.masks,
                        prediction.boxes,
                        prediction.scores,
                    )
                    writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
                frame_index += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()

    return {
        "model_id": args.model_id,
        "video": str(args.video),
        "csv": str(args.csv_output),
        "overlay": output_video,
        "frames_profiled": len(latencies),
        "mean_total_ms": mean(latencies) if latencies else None,
        "params_total": params["params_total"],
    }


def _safe_len(value: object) -> int:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile SAM3/EfficientSAM3 on a video.")
    parser.add_argument("--model-id", default="sam3")
    parser.add_argument("--backend", choices=["null", "sam3", "efficientsam3"], default="sam3")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone-type", default="efficientvit")
    parser.add_argument("--model-name", default="b0")
    parser.add_argument("--text-encoder-type")
    parser.add_argument("--text-encoder-context-length", type=int, default=77)
    parser.add_argument("--text-encoder-pos-embed-table-size", type=int)
    parser.add_argument("--interpolate-pos-embed", action="store_true")
    parser.add_argument("--prompt", default="monitor")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--overlay-output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
