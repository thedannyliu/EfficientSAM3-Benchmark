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
    def __init__(self, video_path: Path, frame_id: str = "camera") -> None:
        self.video_path = video_path
        self.frame_id = frame_id
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"failed to open video source: {video_path}")

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
        return (
            ResultEnvelope(
                frame_index=envelope.frame_index,
                stamp_ns=envelope.stamp_ns,
                frame_id=envelope.frame_id,
                backend=self.config.backend,
                prompt=self.prompt.text or "",
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
        ),
        Prompt(text=args.prompt),
    )
    if args.overlay_output:
        args.overlay_output.parent.mkdir(parents=True, exist_ok=True)
        fps = video_node.capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(video_node.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_node.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
                if writer is not None:
                    overlay_rgb = overlay_prediction(
                        image_msg.image_rgb,
                        prediction.masks,
                        prediction.boxes,
                        prediction.scores,
                    )
                    writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
                frames += 1
    finally:
        video_node.close()
        if writer is not None:
            writer.release()
    print(json.dumps({"frames": frames, "jsonl": str(args.output_jsonl), "overlay": str(args.overlay_output) if args.overlay_output else None}, indent=2))


def _safe_len(value: object) -> int:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a ROS-free Thor pipeline smoke test on PACE.")
    parser.add_argument("--backend", choices=["null", "sam3", "efficientsam3"], default="null")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone-type", default="efficientvit")
    parser.add_argument("--model-name", default="b0")
    parser.add_argument("--text-encoder-type")
    parser.add_argument("--text-encoder-context-length", type=int, default=77)
    parser.add_argument("--text-encoder-pos-embed-table-size", type=int)
    parser.add_argument("--interpolate-pos-embed", action="store_true")
    parser.add_argument("--prompt", default="monitor")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame-id", default="camera")
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--overlay-output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
