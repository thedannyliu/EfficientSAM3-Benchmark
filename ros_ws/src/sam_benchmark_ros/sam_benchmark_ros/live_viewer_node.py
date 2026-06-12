from __future__ import annotations

import json
import shutil
import subprocess
import threading
from collections import deque
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

        self.bridge = CvBridge()
        self.window_name = str(self.get_parameter("window_name").value)
        display_fps = float(self.get_parameter("display_fps").value)
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
        self.get_logger().info(
            f"viewing {image_topic} beside {segmented_image_topic}; metrics from {result_topic}"
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
        left = self.latest_image
        right = self.latest_segmented
        if left.shape[:2] != right.shape[:2]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)
        combined = np.hstack([left, right])
        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        _draw_labels(combined, left_width=left.shape[1])
        _draw_metrics(
            combined,
            viewer_fps=self._viewer_fps(),
            metrics=self.latest_metrics,
            gpu_util=self.gpu_monitor.gpu_util,
        )
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


def _draw_labels(image_bgr: np.ndarray, left_width: int) -> None:
    _draw_text(image_bgr, "Original", (16, 32))
    _draw_text(image_bgr, "SAM3 mask overlay", (left_width + 16, 32))


def _draw_metrics(
    image_bgr: np.ndarray,
    viewer_fps: float | None,
    metrics: dict[str, Any],
    gpu_util: float | None,
) -> None:
    lines = [
        f"FPS: {_format_float(viewer_fps)}",
        f"SAM latency: {_format_float(metrics.get('latency_ms'))} ms",
        f"Callback: {_format_float(metrics.get('callback_total_ms'))} ms",
        f"End-to-end: {_format_float(metrics.get('end_to_end_ms'))} ms",
        f"GPU util: {_format_float(gpu_util)}%",
        f"CUDA alloc: {_format_float(metrics.get('cuda_allocated_mb'))} MB",
    ]
    x = image_bgr.shape[1] // 2 + 16
    y = 64
    width = 360
    height = 24 * len(lines) + 16
    cv2.rectangle(image_bgr, (x - 8, y - 22), (x + width, y + height), (0, 0, 0), -1)
    for idx, line in enumerate(lines):
        _draw_text(image_bgr, line, (x, y + idx * 24), scale=0.58)


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
