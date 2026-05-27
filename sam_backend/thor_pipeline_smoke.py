from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter, time_ns

import cv2

from .backends import BackendConfig, Prompt, create_backend
from .overlay import overlay_prediction


@dataclass(slots=True)
class ImageEnvelope:
    frame_index: int
    stamp_ns: int
    frame_id: str
    image_rgb: object


@dataclass(slots=True)
class ResultEnvelope:
    frame_index: int
    stamp_ns: int
    frame_id: str
    backend: str
    prompt: str
    latency_ms: float
    end_to_end_ms: float
    mask_count: int
    box_count: int


class VideoStreamNodeShim:
    def __init__(self, source: str, frame_id: str = "camera") -> None:
        self.source = source
        self.frame_id = frame_id
        self.capture = cv2.VideoCapture(_capture_source(source))
        if not self.capture.isOpened():
            raise RuntimeError(f"failed to open video source: {source}")

    def read(self, frame_index: int) -> ImageEnvelope | None:
        ok, frame_bgr = self.capture.read()
        if not ok:
            return None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return ImageEnvelope(
            frame_index=frame_index,
            stamp_ns=time_ns(),
            frame_id=self.frame_id,
            image_rgb=frame_rgb,
        )

    def close(self) -> None:
        self.capture.release()


class SamBackendNodeShim:
    def __init__(self, config: BackendConfig, prompt: Prompt) -> None:
        self.config = config
        self.prompt = prompt
        self.backend = create_backend(config)

    def handle(self, envelope: ImageEnvelope) -> tuple[ResultEnvelope, object]:
        start = perf_counter()
        prediction = self.backend.predict(envelope.image_rgb, self.prompt)
        prompt_text = self.prompt.text or json.dumps({"points": self.prompt.points, "labels": self.prompt.labels})
        return (
            ResultEnvelope(
                frame_index=envelope.frame_index,
                stamp_ns=envelope.stamp_ns,
                frame_id=envelope.frame_id,
                backend=self.config.backend,
                prompt=prompt_text,
                latency_ms=prediction.latency_ms,
                end_to_end_ms=(perf_counter() - start) * 1000.0,
                mask_count=_safe_len(prediction.masks),
                box_count=_safe_len(prediction.boxes),
            ),
            prediction,
        )


def main() -> None:
    args = parse_args()
    run_smoke(args)


def run_smoke(args: argparse.Namespace) -> None:
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    video_node = VideoStreamNodeShim(args.video, frame_id=args.frame_id)
    width = int(video_node.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_node.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    prompt = _build_prompt(args, width, height)
    backend_node = SamBackendNodeShim(
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
            enable_inst_interactivity=bool(prompt.points),
            model_config=args.model_config,
            external_repo=args.external_repo,
        ),
        prompt,
    )
    if args.overlay_output:
        args.overlay_output.parent.mkdir(parents=True, exist_ok=True)
        fps = video_node.capture.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(
            str(args.overlay_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to create overlay video: {args.overlay_output}")

    frames = 0
    try:
        with args.output_jsonl.open("w", encoding="utf-8") as f:
            while frames < args.max_frames:
                image_msg = video_node.read(frames)
                if image_msg is None:
                    break
                result, prediction = backend_node.handle(image_msg)
                f.write(json.dumps(asdict(result)) + "\n")
                if args.sample_frame_dir:
                    args.sample_frame_dir.mkdir(parents=True, exist_ok=True)
                    frame_path = args.sample_frame_dir / f"frame_{frames:06d}.png"
                    cv2.imwrite(str(frame_path), cv2.cvtColor(image_msg.image_rgb, cv2.COLOR_RGB2BGR))
                if writer is not None:
                    overlay_rgb = overlay_prediction(
                        image_msg.image_rgb,
                        prediction.masks,
                        prediction.boxes,
                        prediction.scores,
                    )
                    writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
                    if args.overlay_frame_dir:
                        args.overlay_frame_dir.mkdir(parents=True, exist_ok=True)
                        overlay_path = args.overlay_frame_dir / f"frame_{frames:06d}.png"
                        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
                frames += 1
    finally:
        video_node.close()
        if writer is not None:
            writer.release()
    print(
        json.dumps(
            {
                "frames": frames,
                "jsonl": str(args.output_jsonl),
                "sample_frame_dir": str(args.sample_frame_dir) if args.sample_frame_dir else None,
                "overlay": str(args.overlay_output) if args.overlay_output else None,
                "overlay_frame_dir": str(args.overlay_frame_dir) if args.overlay_frame_dir else None,
            },
            indent=2,
        )
    )


def _safe_len(value: object) -> int:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _capture_source(source: str) -> int | str:
    if source.isdigit():
        return int(source)
    return source


def _build_prompt(args: argparse.Namespace, width: int, height: int) -> Prompt:
    if args.point:
        points = [_parse_point(value, width, height, args.point_normalized) for value in args.point]
        labels = args.point_label or [1] * len(points)
        if len(labels) != len(points):
            raise ValueError("--point-label count must match --point count")
        return Prompt(points=points, labels=labels)
    return Prompt(text=args.prompt)


def _parse_point(value: str, width: int, height: int, normalized: bool) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError(f"point must be formatted as x,y: {value!r}")
    x = float(parts[0])
    y = float(parts[1])
    if normalized:
        if width <= 0 or height <= 0:
            raise ValueError("normalized points require a video source with known width and height")
        x *= width
        y *= height
    return (x, y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a ROS-free Thor pipeline smoke test on PACE.")
    parser.add_argument(
        "--backend",
        choices=["null", "sam3", "efficientsam3", "sam2", "efficient-sam2", "efficienttam"],
        default="null",
    )
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-config", help="Native config path for SAM2/EfficientTAM-style backends.")
    parser.add_argument("--external-repo", help="Optional repo root to prepend to PYTHONPATH for external backends.")
    parser.add_argument("--backbone-type", default="efficientvit")
    parser.add_argument("--model-name", default="b0")
    parser.add_argument("--text-encoder-type")
    parser.add_argument("--text-encoder-context-length", type=int, default=77)
    parser.add_argument("--text-encoder-pos-embed-table-size", type=int)
    parser.add_argument("--interpolate-pos-embed", action="store_true")
    parser.add_argument("--prompt", default="monitor")
    parser.add_argument("--point", action="append", help="Point prompt as x,y. May be repeated.")
    parser.add_argument("--point-label", action="append", type=int, help="Point label for each --point.")
    parser.add_argument("--point-normalized", action="store_true", help="Interpret --point values as 0..1 image fractions.")
    parser.add_argument("--video", required=True, help="Video path or camera index, e.g. videos/test1.mov or 0.")
    parser.add_argument("--frame-id", default="camera")
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--overlay-output", type=Path)
    parser.add_argument("--sample-frame-dir", type=Path, help="Optional directory for sampled input frames.")
    parser.add_argument("--overlay-frame-dir", type=Path, help="Optional directory for overlay PNG frames.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
