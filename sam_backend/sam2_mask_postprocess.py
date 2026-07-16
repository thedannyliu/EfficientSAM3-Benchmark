from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from .sam2_video_demo import (
    FFmpegVideoWriter,
    OBJECT_COLORS,
    probe_video,
    resolve_ffmpeg_codec,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay saved SAM2 masks on the source video without model metadata."
    )
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-path", required=True)
    lead_group = parser.add_mutually_exclusive_group()
    lead_group.add_argument("--lead-seconds", type=float)
    lead_group.add_argument("--lead-frames", type=int)
    lead_group.add_argument("--lead-seconds-list", type=float, nargs="+")
    parser.add_argument("--alpha", type=float, default=0.22)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--crf", type=int, default=18)
    return parser


def resolve_lead_frames(
    fps: float,
    *,
    lead_seconds: float | None,
    lead_frames: int | None,
) -> int:
    return resolve_lead_frame_offsets(
        fps,
        lead_seconds=lead_seconds,
        lead_frames=lead_frames,
        lead_seconds_list=None,
    )[0]


def resolve_lead_frame_offsets(
    fps: float,
    *,
    lead_seconds: float | None,
    lead_frames: int | None,
    lead_seconds_list: list[float] | None,
) -> list[int]:
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if lead_frames is not None:
        if lead_frames < 0:
            raise ValueError("lead-frames must not be negative")
        return [lead_frames]
    seconds_values = (
        lead_seconds_list
        if lead_seconds_list is not None
        else [0.1 if lead_seconds is None else lead_seconds]
    )
    if not seconds_values:
        raise ValueError("lead-seconds-list must not be empty")
    if any(seconds < 0.0 or not math.isfinite(seconds) for seconds in seconds_values):
        raise ValueError("lead seconds must be finite and not negative")
    offsets = sorted(
        {
            max(0, int(math.floor(seconds * fps + 0.5)))
            for seconds in seconds_values
        }
    )
    return offsets


def prediction_layer_alpha(
    layer_index: int,
    layer_count: int,
    max_alpha: float,
) -> float:
    if layer_count < 1 or not 0 <= layer_index < layer_count:
        raise ValueError("prediction layer index is out of range")
    if not 0.0 <= max_alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    return max_alpha * (layer_index + 1) / layer_count


def overlay_label_mask(
    frame_bgr: np.ndarray,
    label_map: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if label_map.shape != frame_bgr.shape[:2]:
        label_map = cv2.resize(
            label_map,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    output = frame_bgr.copy()
    for object_id in sorted(int(value) for value in np.unique(label_map) if value > 0):
        mask = label_map == object_id
        color = np.asarray(
            OBJECT_COLORS[(object_id - 1) % len(OBJECT_COLORS)],
            dtype=np.float32,
        )
        output[mask] = (
            output[mask].astype(np.float32) * (1.0 - alpha) + color * alpha
        ).astype(np.uint8)
    return output


def load_mask_paths(mask_dir: Path) -> dict[int, Path]:
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"mask directory does not exist: {mask_dir}")
    mask_paths: dict[int, Path] = {}
    for path in mask_dir.glob("*.png"):
        try:
            frame_index = int(path.stem)
        except ValueError:
            continue
        mask_paths[frame_index] = path
    if not mask_paths:
        raise RuntimeError(f"mask directory contains no frame PNGs: {mask_dir}")
    return mask_paths


def main() -> None:
    args = build_parser().parse_args()
    video_path = Path(args.video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video does not exist: {video_path}")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    video_info, _ = probe_video(video_path)
    lead_frame_offsets = resolve_lead_frame_offsets(
        video_info.fps,
        lead_seconds=args.lead_seconds,
        lead_frames=args.lead_frames,
        lead_seconds_list=args.lead_seconds_list,
    )
    mask_paths = load_mask_paths(Path(args.mask_dir))
    output_path = Path(args.output_path)
    codec = resolve_ffmpeg_codec(args.codec)
    writer = FFmpegVideoWriter(
        output_path,
        width=video_info.width,
        height=video_info.height,
        fps=video_info.fps,
        source_path=video_path,
        codec=codec,
        preset=args.preset,
        crf=args.crf,
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        writer.close()
        raise RuntimeError(f"failed to open video: {video_path}")

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for layer_index, lead_frames in enumerate(lead_frame_offsets):
                mask_path = mask_paths.get(frame_index + lead_frames)
                if mask_path is None:
                    continue
                label_map = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if label_map is None:
                    raise RuntimeError(f"failed to decode mask: {mask_path}")
                frame = overlay_label_mask(
                    frame,
                    label_map,
                    alpha=prediction_layer_alpha(
                        layer_index,
                        len(lead_frame_offsets),
                        args.alpha,
                    ),
                )
            writer.write(frame)
            frame_index += 1
            if frame_index % 100 == 0:
                print(f"Encoded {frame_index} frames")
    finally:
        capture.release()
        writer.close()
    print(
        f"Wrote {output_path} at {video_info.fps:g} FPS with mask leads "
        f"{lead_frame_offsets} frames"
    )


if __name__ == "__main__":
    main()
