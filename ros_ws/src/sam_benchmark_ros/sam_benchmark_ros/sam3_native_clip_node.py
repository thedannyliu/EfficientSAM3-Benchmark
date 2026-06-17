from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from sam_backend.backends import _import_required, _prepend_repo_path
from sam_backend.overlay import overlay_prediction
from sam_backend.profiling import cuda_memory_mb, parameter_counts
from sam_backend.streaming import masks_to_mono8


class Sam3NativeClipNode(Node):
    def __init__(self) -> None:
        super().__init__("sam3_native_clip_node")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("result_topic", "/sam/result_json")
        self.declare_parameter("mask_topic", "/segmentation_mask")
        self.declare_parameter("segmented_image_topic", "/segmented_image")
        self.declare_parameter("overlay_topic", "/sam/overlay")
        self.declare_parameter("checkpoint_path", "checkpoints/sam3/sam3.pt")
        self.declare_parameter("external_repo", "external/sam3")
        self.declare_parameter("prompt", "monitor")
        self.declare_parameter("clip_frames", 120)
        self.declare_parameter("frame_dir", "results/thor/ros_camera/sam3_native_clip/frames")
        self.declare_parameter("version", "sam3")

        self.bridge = CvBridge()
        self.prompt = str(self.get_parameter("prompt").value)
        self.clip_frames = int(self.get_parameter("clip_frames").value)
        if self.clip_frames <= 0:
            raise ValueError("clip_frames must be positive")
        self.frame_dir = Path(str(self.get_parameter("frame_dir").value))
        self.frames: list[np.ndarray] = []
        self.headers: list[Any] = []
        self.result_times: deque[float] = deque(maxlen=60)
        self.processing_started = False

        external_repo = str(self.get_parameter("external_repo").value)
        checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        _prepend_repo_path(external_repo)
        torch_module = _import_required("torch")
        builder = _import_required("sam3.model_builder")
        self.torch_module = torch_module
        self.predictor = builder.build_sam3_predictor(
            checkpoint_path=checkpoint_path,
            version=str(self.get_parameter("version").value),
        )
        self.params = parameter_counts(getattr(self.predictor, "model", self.predictor))
        if torch_module.cuda.is_available():
            torch_module.cuda.reset_peak_memory_stats()

        image_topic = str(self.get_parameter("image_topic").value)
        result_topic = str(self.get_parameter("result_topic").value)
        mask_topic = str(self.get_parameter("mask_topic").value)
        segmented_image_topic = str(self.get_parameter("segmented_image_topic").value)
        overlay_topic = str(self.get_parameter("overlay_topic").value)
        self.result_publisher = self.create_publisher(String, result_topic, 10)
        self.mask_publisher = self.create_publisher(Image, mask_topic, 10)
        self.segmented_image_publisher = self.create_publisher(Image, segmented_image_topic, 10)
        self.overlay_publisher = self.create_publisher(Image, overlay_topic, 10) if overlay_topic else None
        self.subscription = self.create_subscription(Image, image_topic, self.on_image, 10)
        self.get_logger().info(
            f"capturing {self.clip_frames} frames from {image_topic}; "
            f"SAM3 native tracking will run after the clip is materialized"
        )

    def on_image(self, msg: Image) -> None:
        if self.processing_started:
            return
        frame_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.frames.append(frame_rgb.copy())
        self.headers.append(msg.header)
        if len(self.frames) % 30 == 0 or len(self.frames) == self.clip_frames:
            self.get_logger().info(f"captured {len(self.frames)}/{self.clip_frames} frames")
        if len(self.frames) >= self.clip_frames:
            self.processing_started = True
            self._process_clip()

    def _process_clip(self) -> None:
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in self.frame_dir.iterdir():
            if old_frame.is_file() and old_frame.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                old_frame.unlink()
        for idx, frame_rgb in enumerate(self.frames):
            path = self.frame_dir / f"{idx:06d}.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        session_id = None
        try:
            response = self.predictor.handle_request({"type": "start_session", "resource_path": str(self.frame_dir)})
            session_id = response["session_id"]
            self.predictor.handle_request(
                {"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": self.prompt}
            )
            iterator = self.predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "start_frame_index": 0,
                    "max_frame_num_to_track": self.clip_frames,
                }
            )
            last = perf_counter()
            for response in iterator:
                now = perf_counter()
                latency_ms = (now - last) * 1000.0
                last = now
                frame_index = int(response.get("frame_index", response.get("frame_idx", 0)))
                if frame_index >= len(self.frames):
                    continue
                masks = _sam3_output_masks(response.get("outputs", {}), self.frames[frame_index].shape[:2])
                self._publish_frame(frame_index, masks, latency_ms)
        finally:
            if session_id is not None:
                self.predictor.handle_request({"type": "close_session", "session_id": session_id})

    def _publish_frame(self, frame_index: int, masks: Any, latency_ms: float) -> None:
        frame = self.frames[frame_index]
        header = self.headers[frame_index]
        mask = masks_to_mono8(masks, frame.shape[:2])
        overlay = overlay_prediction(frame, masks)
        callback_total_ms = latency_ms
        end_to_end_ms = self._end_to_end_ms(header)
        self.result_times.append(self.get_clock().now().nanoseconds / 1_000_000_000.0)
        tracking_fps = self._tracking_fps()
        memory = cuda_memory_mb(self.torch_module)
        result = {
            "frame_index": frame_index,
            "stamp": {"sec": header.stamp.sec, "nanosec": header.stamp.nanosec},
            "frame_id": header.frame_id,
            "backend": "sam3",
            "stream_mode": "native_clip",
            "tracking_state": "tracking",
            "prompt_mode": "native_text",
            "prompt_text": self.prompt,
            "latency_ms": latency_ms,
            "callback_total_ms": callback_total_ms,
            "end_to_end_ms": end_to_end_ms,
            "tracking_fps": tracking_fps,
            "mask_count": _safe_len(masks),
            **memory,
            **self.params,
        }

        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        mask_msg.header = header
        self.mask_publisher.publish(mask_msg)

        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="rgb8")
        overlay_msg.header = header
        self.segmented_image_publisher.publish(overlay_msg)
        if self.overlay_publisher is not None:
            self.overlay_publisher.publish(overlay_msg)
        self.result_publisher.publish(String(data=json.dumps(result)))

    def _end_to_end_ms(self, header: Any) -> float:
        now_msg = self.get_clock().now().to_msg()
        return _stamp_delta_ms(header.stamp.sec, header.stamp.nanosec, now_msg.sec, now_msg.nanosec)

    def _tracking_fps(self) -> float | str:
        if len(self.result_times) < 2:
            return ""
        duration = self.result_times[-1] - self.result_times[0]
        if duration <= 0:
            return ""
        return (len(self.result_times) - 1) / duration

    def destroy_node(self) -> bool:
        if hasattr(self, "predictor") and hasattr(self.predictor, "shutdown"):
            self.predictor.shutdown()
        return super().destroy_node()


def _sam3_output_masks(outputs: dict[str, Any], frame_hw: tuple[int, int]) -> Any:
    for key in ("out_binary_masks", "pred_masks", "masks"):
        if key in outputs:
            return outputs[key]
    return np.zeros((0, frame_hw[0], frame_hw[1]), dtype=np.uint8)


def _safe_len(value: object) -> int:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _stamp_delta_ms(start_sec: int, start_nanosec: int, end_sec: int, end_nanosec: int) -> float:
    start_ns = start_sec * 1_000_000_000 + start_nanosec
    end_ns = end_sec * 1_000_000_000 + end_nanosec
    return (end_ns - start_ns) / 1_000_000.0


def main() -> None:
    rclpy.init()
    node = Sam3NativeClipNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
