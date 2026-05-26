from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Iterable

import cv2
import numpy as np

from .backends import BackendConfig, Prompt, create_backend


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> dict:
    backend = create_backend(
        BackendConfig(
            backend=args.backend,
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            backbone_type=args.backbone_type,
            model_name=args.model_name,
        )
    )
    prompt = Prompt(text=args.prompt)
    frames = list(iter_frames(args))
    if not frames:
        raise RuntimeError("no frames available for benchmark")

    total_iters = args.warmup + args.runs
    latencies: list[float] = []
    start_total = perf_counter()
    for idx in range(total_iters):
        frame = frames[idx % len(frames)]
        prediction = backend.predict(frame, prompt)
        if idx >= args.warmup:
            latencies.append(prediction.latency_ms)
    total_s = perf_counter() - start_total

    return {
        "backend": args.backend,
        "prompt": args.prompt,
        "frames_loaded": len(frames),
        "warmup": args.warmup,
        "runs": args.runs,
        "latency_ms": summarize(latencies),
        "throughput_fps": args.runs / total_s if total_s > 0 else None,
    }


def iter_frames(args: argparse.Namespace) -> Iterable[np.ndarray]:
    if args.synthetic_frames:
        for idx in range(args.synthetic_frames):
            yield synthetic_frame(idx, args.width, args.height)
        return
    if args.image:
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {args.image}")
        yield cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return
    if args.video:
        cap = cv2.VideoCapture(str(args.video))
        try:
            count = 0
            while count < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                count += 1
        finally:
            cap.release()
        return
    raise RuntimeError("provide --image, --video, or --synthetic-frames")


def synthetic_frame(idx: int, width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = 20 + (idx * 7) % max(1, width - 80)
    y0 = 20 + (idx * 5) % max(1, height - 80)
    frame[y0 : y0 + 60, x0 : x0 + 60] = (220, 80, 40)
    frame[:, :, 1] = np.linspace(0, 120, width, dtype=np.uint8)
    return frame


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SAM3/EfficientSAM3 image inference.")
    parser.add_argument("--backend", choices=["null", "sam3", "efficientsam3"], default="null")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--device")
    parser.add_argument("--backbone-type", default="efficientvit")
    parser.add_argument("--model-name", default="b0")
    parser.add_argument("--prompt", default="object")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--synthetic-frames", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
