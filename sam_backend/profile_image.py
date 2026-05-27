from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2

from .backends import BackendConfig, Prompt, create_backend
from .overlay import overlay_prediction, to_numpy
from .profiling import component_timer, cuda_memory_mb, parameter_counts


def main() -> None:
    args = parse_args()
    summary = profile_image(args)
    print(json.dumps(summary, indent=2))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def profile_image(args: argparse.Namespace) -> dict[str, Any]:
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
            enable_inst_interactivity=bool(args.point),
            model_config=args.model_config,
            external_repo=args.external_repo,
        )
    )
    torch_module = getattr(backend, "torch", None)
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()
    params = parameter_counts(getattr(backend, "model", None))

    frame_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(f"failed to read image: {args.image}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    prompt = _build_prompt(args, frame_rgb.shape[1], frame_rgb.shape[0])

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

    if args.overlay_output:
        args.overlay_output.parent.mkdir(parents=True, exist_ok=True)
        overlay_rgb = overlay_prediction(frame_rgb, prediction.masks, prediction.boxes, prediction.scores)
        ok = cv2.imwrite(str(args.overlay_output), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"failed to write overlay image: {args.overlay_output}")

    return {
        "model_id": args.model_id,
        "backend": args.backend,
        "image": str(args.image),
        "prompt": args.prompt,
        "points": prompt.points,
        "labels": prompt.labels,
        "width": frame_rgb.shape[1],
        "height": frame_rgb.shape[0],
        "mask_count": _safe_len(prediction.masks),
        "box_count": _safe_len(prediction.boxes),
        "score_max": float(scores.max()) if scores.size else None,
        "total_ms": total_ms,
        "image_encoder_ms": profile.get("image_encoder_ms", 0.0),
        "text_encoder_ms": profile.get("text_encoder_ms", 0.0),
        "prompt_encoder_ms": profile.get("prompt_encoder_ms", 0.0),
        "mask_decoder_ms": profile.get("mask_decoder_ms", 0.0),
        "grounding_ms": profile.get("grounding_ms", 0.0),
        "memory_attention_ms": profile.get("memory_attention_ms", 0.0),
        "memory_encoder_ms": profile.get("memory_encoder_ms", 0.0),
        "other_ms": max(0.0, total_ms - component_total),
        "overlay": str(args.overlay_output) if args.overlay_output else None,
        **memory,
        **params,
    }


def _safe_len(value: object) -> int:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _build_prompt(args: argparse.Namespace, width: int, height: int) -> Prompt:
    if args.point:
        points = [_parse_point(value, width, height, args.point_normalized) for value in args.point]
        labels = args.point_label or [1] * len(points)
        if len(labels) != len(points):
            raise ValueError("--point-label count must match --point count")
        return Prompt(points=points, labels=labels)
    if args.prompt:
        return Prompt(text=args.prompt)
    raise ValueError("provide --prompt or at least one --point")


def _parse_point(value: str, width: int, height: int, normalized: bool) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError(f"point must be formatted as x,y: {value!r}")
    x = float(parts[0])
    y = float(parts[1])
    if normalized:
        x *= width
        y *= height
    return (x, y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a supported SAM backend on a single image.")
    parser.add_argument("--model-id", default="sam3-image")
    parser.add_argument(
        "--backend",
        choices=["null", "sam3", "efficientsam3", "sam2", "efficient-sam2", "efficienttam"],
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
    parser.add_argument("--prompt", default="cats")
    parser.add_argument(
        "--point",
        action="append",
        help="Point prompt as x,y. May be repeated. Use --point-normalized for 0..1 coordinates.",
    )
    parser.add_argument(
        "--point-label",
        action="append",
        type=int,
        help="Point label for each --point, usually 1 for positive or 0 for negative.",
    )
    parser.add_argument("--point-normalized", action="store_true", help="Interpret --point values as 0..1 image fractions.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--overlay-output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
