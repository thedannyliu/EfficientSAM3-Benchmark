from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def stamp_to_seconds(stamp: Any) -> float | None:
    seconds = float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0
    if seconds <= 0.0:
        return None
    return seconds


class TimedVideoRecorder:
    def __init__(
        self,
        enabled: bool,
        output_path: Path,
        fps: float,
        max_frames: int = 0,
        preserve_timing: bool = True,
    ) -> None:
        self.enabled = enabled
        self.output_path = output_path
        self.fps = fps
        self.max_frames = max_frames
        self.preserve_timing = preserve_timing
        self.writer: cv2.VideoWriter | None = None
        self.frames = 0
        self.last_stamp_seconds: float | None = None

    def write(self, frame_rgb: np.ndarray, stamp_seconds: float | None = None) -> None:
        if not self.enabled:
            return
        self._ensure_writer(frame_rgb)
        repeat_count = self._repeat_count(stamp_seconds)
        if repeat_count <= 0:
            return
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        for _ in range(repeat_count):
            if self.max_frames > 0 and self.frames >= self.max_frames:
                raise SystemExit
            assert self.writer is not None
            self.writer.write(frame_bgr)
            self.frames += 1

    def release(self, logger: Any) -> None:
        if self.writer is not None:
            self.writer.release()
            logger.info(f"wrote {self.frames} overlay frames to {self.output_path}")

    def _ensure_writer(self, frame_rgb: np.ndarray) -> None:
        if self.writer is not None:
            return
        if self.fps <= 0:
            raise ValueError("video recorder fps must be positive")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame_rgb.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"failed to create overlay video: {self.output_path}")

    def _repeat_count(self, stamp_seconds: float | None) -> int:
        if not self.preserve_timing or stamp_seconds is None:
            return 1
        if self.last_stamp_seconds is None:
            self.last_stamp_seconds = stamp_seconds
            return 1
        delta = stamp_seconds - self.last_stamp_seconds
        if delta <= 0.0:
            return 0
        self.last_stamp_seconds = stamp_seconds
        return max(1, int(round(delta * self.fps)))
