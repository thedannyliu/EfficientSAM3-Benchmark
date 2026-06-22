from __future__ import annotations

import json
import shutil
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from sam_backend.streaming import parse_tegrastats_gr3d


class LiveViewerNode(Node):
    def __init__(self) -> None:
        super().__init__("live_viewer_node")
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("segmented_image_topic", "/segmented_image")
        self.declare_parameter("result_topic", "/sam/result_json")
        self.declare_parameter("window_name", "SAM3 ROS Streaming")
        self.declare_parameter("display_fps", 30.0)
        self.declare_parameter("display_scale", 1.0)
        self.declare_parameter("display_max_width", 0)
        self.declare_parameter("record_overlay", False)
        self.declare_parameter("overlay_video_output", "overlays/ros/live_viewer_overlay.mp4")
        self.declare_parameter("overlay_video_fps", 15.0)
        self.declare_parameter("overlay_video_max_frames", 0)

        self.bridge = CvBridge()
        self.window_name = str(self.get_parameter("window_name").value)
        display_fps = float(self.get_parameter("display_fps").value)
        self.display_scale = float(self.get_parameter("display_scale").value)
        self.display_max_width = int(self.get_parameter("display_max_width").value)
        self.recorder = OverlayRecorder(
            enabled=bool(self.get_parameter("record_overlay").value),
            output_path=Path(str(self.get_parameter("overlay_video_output").value)),
            fps=float(self.get_parameter("overlay_video_fps").value),
            max_frames=int(self.get_parameter("overlay_video_max_frames").value),
        )
        self.latest_image: np.ndarray | None = None
        self.latest_segmented: np.ndarray | None = None
        self.latest_metrics: dict[str, Any] = {}
        self.segmented_times: deque[float] = deque(maxlen=60)
        self.gpu_monitor = TegrastatsMonitor()
        self.gpu_monitor.start()

        image_topic = self.get_parameter("image_topic").value
        segmented_image_topic = self.get_parameter("segmented_image_topic").value
        result_topic = self.get_parameter("result_topic").value
        self.image_subscription = self.create_subscription(Image, image_topic, self.on_image, 10)
        self.segmented_subscription = self.create_subscription(Image, segmented_image_topic, self.on_segmented, 10)
        self.result_subscription = self.create_subscription(String, result_topic, self.on_result, 10)
        self.timer = self.create_timer(1.0 / display_fps, self.display)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.get_logger().info(
            f"viewing {image_topic} beside {segmented_image_topic}; metrics from {result_topic}; "
            f"display_scale={self.display_scale:g} display_max_width={self.display_max_width}; "
            f"record_overlay={self.recorder.enabled}"
        )

    def on_image(self, msg: Image) -> None:
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def on_segmented(self, msg: Image) -> None:
        self.latest_segmented = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.segmented_times.append(self.get_clock().now().nanoseconds / 1_000_000_000.0)

    def on_result(self, msg: String) -> None:
        try:
            self.latest_metrics = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("failed to parse result JSON")

    def display(self) -> None:
        if self.latest_image is None or self.latest_segmented is None:
            return
        overlay = self.latest_segmented
        if self.latest_image.shape[:2] != overlay.shape[:2]:
            overlay = cv2.resize(
                overlay,
                (self.latest_image.shape[1], self.latest_image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        self.recorder.write(overlay)
        combined = _display_with_metrics(
            overlay,
            viewer_fps=self._viewer_fps(),
            metrics=self.latest_metrics,
            gpu_util=self.gpu_monitor.gpu_util,
        )
        combined = _scale_display(combined, self.display_scale, self.display_max_width)
        cv2.imshow(self.window_name, combined)
        key = cv2.waitKey(1) & 0xFF
        if key in {27, ord("q")}:
            raise SystemExit

    def _viewer_fps(self) -> float | None:
        if len(self.segmented_times) < 2:
            return None
        duration = self.segmented_times[-1] - self.segmented_times[0]
        if duration <= 0:
            return None
        return (len(self.segmented_times) - 1) / duration

    def destroy_node(self) -> bool:
        self.gpu_monitor.stop()
        self.recorder.release(self.get_logger())
        cv2.destroyAllWindows()
        return super().destroy_node()


class TegrastatsMonitor:
    def __init__(self) -> None:
        self.gpu_util: float | None = None
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if shutil.which("tegrastats") is None:
            return
        self.process = subprocess.Popen(
            ["tegrastats", "--interval", "1000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def _read_loop(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            if self._stop.is_set():
                return
            value = parse_tegrastats_gr3d(line)
            if value is not None:
                self.gpu_util = value


class OverlayRecorder:
    def __init__(self, enabled: bool, output_path: Path, fps: float, max_frames: int) -> None:
        self.enabled = enabled
        self.output_path = output_path
        self.fps = fps
        self.max_frames = max_frames
        self.writer: cv2.VideoWriter | None = None
        self.frames = 0

    def write(self, frame_rgb: np.ndarray) -> None:
        if not self.enabled:
            return
        if self.writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame_rgb.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"failed to create overlay video: {self.output_path}")
        self.writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        self.frames += 1
        if self.max_frames > 0 and self.frames >= self.max_frames:
            raise SystemExit

    def release(self, logger: Any) -> None:
        if self.writer is not None:
            self.writer.release()
            logger.info(f"wrote {self.frames} overlay frames to {self.output_path}")


def _display_with_metrics(
    overlay_rgb: np.ndarray,
    viewer_fps: float | None,
    metrics: dict[str, Any],
    gpu_util: float | None,
) -> np.ndarray:
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    panel = _metrics_panel(
        overlay_bgr.shape[0],
        [
            "Overlay",
            f"FPS: {_format_float(viewer_fps)}",
            f"SAM latency: {_format_float(metrics.get('latency_ms'))} ms",
            f"Callback: {_format_float(metrics.get('callback_total_ms'))} ms",
            f"End-to-end: {_format_float(metrics.get('end_to_end_ms'))} ms",
            f"GPU util: {_format_float(gpu_util)}%",
            f"CUDA alloc: {_format_float(metrics.get('cuda_allocated_mb'))} MB",
        ],
    )
    return np.hstack([overlay_bgr, panel])


def _metrics_panel(height: int, lines: list[str], width: int = 360) -> np.ndarray:
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (55, 55, 55), 1)
    y = 34
    for idx, line in enumerate(lines):
        scale = 0.66 if idx == 0 else 0.56
        _draw_text(panel, line, (16, y), scale=scale)
        y += 30 if idx == 0 else 24
    return panel


def _scale_display(image: np.ndarray, display_scale: float, display_max_width: int) -> np.ndarray:
    scale = display_scale if display_scale > 0 else 1.0
    if display_max_width > 0 and image.shape[1] * scale > display_max_width:
        scale = display_max_width / float(image.shape[1])
    if scale == 1.0:
        return image
    width = max(1, int(round(image.shape[1] * scale)))
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _draw_text(image_bgr: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.7) -> None:
    cv2.putText(image_bgr, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image_bgr, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _format_float(value: Any) -> str:
    if value in ("", None):
        return "n/a"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "n/a"


def main() -> None:
    rclpy.init()
    node = LiveViewerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
