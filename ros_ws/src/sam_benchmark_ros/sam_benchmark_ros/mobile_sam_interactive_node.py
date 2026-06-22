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

from sam_backend import BackendConfig, Prompt, create_backend
from sam_backend.overlay import overlay_prediction, to_numpy
from sam_backend.profiling import cuda_memory_mb, parameter_counts
from sam_backend.streaming import left_panel_click_to_image_point, masks_to_bbox_xyxy, masks_to_mono8


class MobileSamInteractiveNode(Node):
    def __init__(self) -> None:
        super().__init__("mobile_sam_interactive_node")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("result_topic", "/sam/result_json")
        self.declare_parameter("mask_topic", "/segmentation_mask")
        self.declare_parameter("segmented_image_topic", "/segmented_image")
        self.declare_parameter("overlay_topic", "/sam/overlay")
        self.declare_parameter("backend", "mobilesam")
        self.declare_parameter("checkpoint_path", "checkpoints/mobilesam/mobile_sam.pt")
        self.declare_parameter("external_repo", "external/MobileSAM")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("mobile_sam_model_type", "vit_t")
        self.declare_parameter("window_name", "MobileSAM RealSense")
        self.declare_parameter("display_fps", 30.0)
        self.declare_parameter("display_scale", 1.0)
        self.declare_parameter("display_max_width", 0)
        self.declare_parameter("record_overlay", False)
        self.declare_parameter("overlay_video_output", "overlays/ros/mobile_sam_overlay.mp4")
        self.declare_parameter("overlay_video_fps", 15.0)
        self.declare_parameter("overlay_video_max_frames", 0)
        self.declare_parameter("bbox_min_area", 25)
        self.declare_parameter("bbox_scale", 1.2)
        self.declare_parameter("enable_display", True)
        self.declare_parameter("auto_start", False)
        self.declare_parameter("initial_point_x", 0.5)
        self.declare_parameter("initial_point_y", 0.5)
        self.declare_parameter("initial_point_normalized", True)

        self.bridge = CvBridge()
        self.window_name = str(self.get_parameter("window_name").value)
        self.display_scale = float(self.get_parameter("display_scale").value)
        self.display_max_width = int(self.get_parameter("display_max_width").value)
        self.current_display_scale = 1.0
        self.recorder = OverlayRecorder(
            enabled=bool(self.get_parameter("record_overlay").value),
            output_path=Path(str(self.get_parameter("overlay_video_output").value)),
            fps=float(self.get_parameter("overlay_video_fps").value),
            max_frames=int(self.get_parameter("overlay_video_max_frames").value),
        )
        self.bbox_min_area = int(self.get_parameter("bbox_min_area").value)
        self.bbox_scale = float(self.get_parameter("bbox_scale").value)
        self.enable_display = bool(self.get_parameter("enable_display").value)
        self.auto_start = bool(self.get_parameter("auto_start").value)
        self.initial_point_x = float(self.get_parameter("initial_point_x").value)
        self.initial_point_y = float(self.get_parameter("initial_point_y").value)
        self.initial_point_normalized = bool(self.get_parameter("initial_point_normalized").value)
        self.pending_point: tuple[float, float] | None = None
        self.tracking_bbox: tuple[float, float, float, float] | None = None
        self.latest_frame: np.ndarray | None = None
        self.latest_display: np.ndarray | None = None
        self.latest_result: dict[str, Any] = {"tracking_state": "waiting_for_click"}
        self.frame_index = 0
        self.result_times: deque[float] = deque(maxlen=60)
        self.backend_name = str(self.get_parameter("backend").value)
        if self.backend_name not in {"mobilesam", "sam1"}:
            raise ValueError("backend must be one of: mobilesam, sam1")
        self.model_label = _model_label(self.backend_name, str(self.get_parameter("mobile_sam_model_type").value))

        self.backend = create_backend(
            BackendConfig(
                backend=self.backend_name,
                checkpoint_path=str(self.get_parameter("checkpoint_path").value),
                device=str(self.get_parameter("device").value),
                external_repo=str(self.get_parameter("external_repo").value),
                mobile_sam_model_type=str(self.get_parameter("mobile_sam_model_type").value),
            )
        )
        self.torch_module = getattr(self.backend, "torch", None)
        self.params = parameter_counts(getattr(self.backend, "model", None))

        image_topic = str(self.get_parameter("image_topic").value)
        result_topic = str(self.get_parameter("result_topic").value)
        mask_topic = str(self.get_parameter("mask_topic").value)
        segmented_image_topic = str(self.get_parameter("segmented_image_topic").value)
        overlay_topic = str(self.get_parameter("overlay_topic").value)
        display_fps = float(self.get_parameter("display_fps").value)

        self.result_publisher = self.create_publisher(String, result_topic, 10)
        self.mask_publisher = self.create_publisher(Image, mask_topic, 10)
        self.segmented_image_publisher = self.create_publisher(Image, segmented_image_topic, 10)
        self.overlay_publisher = self.create_publisher(Image, overlay_topic, 10) if overlay_topic else None
        self.image_subscription = self.create_subscription(Image, image_topic, self.on_image, 1)
        self.timer = self.create_timer(1.0 / display_fps, self.display) if self.enable_display else None

        if self.enable_display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.window_name, self.on_mouse)
            self.get_logger().info(
                f"listening on {image_topic}; click the image to initialize {self.model_label}; "
                f"display_scale={self.display_scale:g} display_max_width={self.display_max_width}; "
                f"record_overlay={self.recorder.enabled}"
            )
        elif self.auto_start:
            self.get_logger().info(f"listening on {image_topic}; auto-starting {self.model_label} from initial point")
        else:
            self.get_logger().info(f"listening on {image_topic}; display disabled and waiting for an external prompt")

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self.latest_frame is None:
            return
        scale = self.current_display_scale if self.current_display_scale > 0 else 1.0
        point = left_panel_click_to_image_point(x / scale, y / scale, self.latest_frame.shape[:2])
        if point is None:
            return
        self.pending_point = point
        self.tracking_bbox = None
        self.latest_result = {"tracking_state": "pending_click", "point_x": point[0], "point_y": point[1]}
        self.get_logger().info(f"received point prompt x={point[0]:.1f} y={point[1]:.1f}")

    def on_image(self, msg: Image) -> None:
        callback_start = perf_counter()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.latest_frame = frame
        prompt = self._next_prompt()
        if prompt is None:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            overlay = _status_overlay(frame, "Click left image to initialize")
            callback_total_ms = (perf_counter() - callback_start) * 1000.0
            result = self._result(
                msg,
                "waiting_for_click",
                "",
                None,
                0.0,
                callback_total_ms,
                self._end_to_end_ms(msg),
            )
        else:
            prediction_start = perf_counter()
            prediction = self.backend.predict(frame, prompt)
            latency_ms = (perf_counter() - prediction_start) * 1000.0
            mask = masks_to_mono8(prediction.masks, frame.shape[:2])
            next_bbox = masks_to_bbox_xyxy(
                prediction.masks,
                frame.shape[:2],
                min_area=self.bbox_min_area,
                scale=self.bbox_scale,
            )
            prompt_mode = "point" if prompt.points else "box"
            if next_bbox is None:
                self.tracking_bbox = None
                state = "lost"
                overlay = _status_overlay(frame, "Tracking lost; click left image")
            else:
                self.tracking_bbox = next_bbox
                state = "tracking"
                overlay = overlay_prediction(frame, prediction.masks, [next_bbox], prediction.scores)
                if prompt.points:
                    x, y = [int(round(value)) for value in prompt.points[0]]
                    cv2.circle(overlay, (x, y), 6, (255, 80, 30), -1, cv2.LINE_AA)
            result = self._result(
                msg,
                state,
                prompt_mode,
                prompt,
                latency_ms,
                (perf_counter() - callback_start) * 1000.0,
                self._end_to_end_ms(msg),
                bbox=next_bbox,
                prediction=prediction,
            )
        self.result_times.append(self.get_clock().now().nanoseconds / 1_000_000_000.0)
        result["tracking_fps"] = self._tracking_fps()
        self.recorder.write(overlay)
        display = _display_with_metrics(overlay, result, self.model_label)
        self.latest_display, self.current_display_scale = _scale_display(
            display,
            self.display_scale,
            self.display_max_width,
        )
        self._publish(msg, mask, overlay, result)
        self.frame_index += 1

    def display(self) -> None:
        if self.latest_display is None:
            return
        cv2.imshow(self.window_name, self.latest_display)
        key = cv2.waitKey(1) & 0xFF
        if key in {27, ord("q")}:
            raise SystemExit
        if key == ord("r"):
            self.pending_point = None
            self.tracking_bbox = None
            self.latest_result = {"tracking_state": "waiting_for_click"}
            self.get_logger().info(f"reset {self.model_label} tracking state")

    def _next_prompt(self) -> Prompt | None:
        if self.pending_point is not None:
            point = self.pending_point
            self.pending_point = None
            return Prompt(points=[point], labels=[1])
        if self.tracking_bbox is not None:
            return Prompt(boxes=[self.tracking_bbox])
        if self.auto_start and self.latest_frame is not None:
            self.auto_start = False
            height, width = self.latest_frame.shape[:2]
            if self.initial_point_normalized:
                point = (self.initial_point_x * float(width), self.initial_point_y * float(height))
            else:
                point = (self.initial_point_x, self.initial_point_y)
            return Prompt(points=[point], labels=[1])
        return None

    def _result(
        self,
        msg: Image,
        state: str,
        prompt_mode: str,
        prompt: Prompt | None,
        latency_ms: float,
        callback_total_ms: float,
        end_to_end_ms: float,
        bbox: tuple[float, float, float, float] | None = None,
        prediction: object | None = None,
    ) -> dict[str, Any]:
        scores = to_numpy(getattr(prediction, "scores", None))
        memory = cuda_memory_mb(self.torch_module) if self.torch_module is not None else cuda_memory_mb(None)
        point = prompt.points[0] if prompt and prompt.points else None
        return {
            "frame_index": self.frame_index,
            "stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec},
            "frame_id": msg.header.frame_id,
            "backend": self.backend_name,
            "tracking_state": state,
            "prompt_mode": prompt_mode,
            "point_x": point[0] if point else "",
            "point_y": point[1] if point else "",
            "box_x1": bbox[0] if bbox else "",
            "box_y1": bbox[1] if bbox else "",
            "box_x2": bbox[2] if bbox else "",
            "box_y2": bbox[3] if bbox else "",
            "latency_ms": latency_ms,
            "callback_total_ms": callback_total_ms,
            "end_to_end_ms": end_to_end_ms,
            "mask_count": _safe_len(getattr(prediction, "masks", None)),
            "box_count": 1 if bbox else 0,
            "score_max": float(scores.max()) if scores.size else None,
            **memory,
            **self.params,
        }

    def _publish(self, msg: Image, mask: np.ndarray, overlay: np.ndarray, result: dict[str, Any]) -> None:
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        mask_msg.header = msg.header
        self.mask_publisher.publish(mask_msg)

        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="rgb8")
        overlay_msg.header = msg.header
        self.segmented_image_publisher.publish(overlay_msg)
        if self.overlay_publisher is not None:
            self.overlay_publisher.publish(overlay_msg)
        self.result_publisher.publish(String(data=json.dumps(result)))

    def _end_to_end_ms(self, msg: Image) -> float:
        now_msg = self.get_clock().now().to_msg()
        return _stamp_delta_ms(msg.header.stamp.sec, msg.header.stamp.nanosec, now_msg.sec, now_msg.nanosec)

    def _tracking_fps(self) -> float | str:
        if len(self.result_times) < 2:
            return ""
        duration = self.result_times[-1] - self.result_times[0]
        if duration <= 0:
            return ""
        return (len(self.result_times) - 1) / duration

    def destroy_node(self) -> bool:
        if self.enable_display:
            cv2.destroyAllWindows()
        self.recorder.release(self.get_logger())
        return super().destroy_node()


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


def _display_with_metrics(overlay_rgb: np.ndarray, result: dict[str, Any], model_label: str) -> np.ndarray:
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    panel = _metrics_panel(
        overlay_bgr.shape[0],
        [
            model_label,
            f"State: {result.get('tracking_state', 'n/a')}",
            f"Prompt: {result.get('prompt_mode') or 'click image'}",
            f"FPS: {_format_float(result.get('tracking_fps'))}",
            f"Latency: {_format_float(result.get('latency_ms'))} ms",
            f"Callback: {_format_float(result.get('callback_total_ms'))} ms",
            f"End-to-end: {_format_float(result.get('end_to_end_ms'))} ms",
            f"CUDA: {_format_float(result.get('cuda_allocated_mb'))} MB",
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


def _scale_display(image: np.ndarray, display_scale: float, display_max_width: int) -> tuple[np.ndarray, float]:
    scale = display_scale if display_scale > 0 else 1.0
    if display_max_width > 0 and image.shape[1] * scale > display_max_width:
        scale = display_max_width / float(image.shape[1])
    if scale == 1.0:
        return image, 1.0
    width = max(1, int(round(image.shape[1] * scale)))
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), scale


def _status_overlay(frame_rgb: np.ndarray, status: str) -> np.ndarray:
    return frame_rgb.copy()


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


def _safe_len(value: object) -> int:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _stamp_delta_ms(start_sec: int, start_nanosec: int, end_sec: int, end_nanosec: int) -> float:
    start_ns = start_sec * 1_000_000_000 + start_nanosec
    end_ns = end_sec * 1_000_000_000 + end_nanosec
    return (end_ns - start_ns) / 1_000_000.0


def _model_label(backend_name: str, model_type: str) -> str:
    if backend_name == "sam1":
        return f"SAM1-{model_type.removeprefix('vit_').upper()}"
    return "MobileSAM"


def main() -> None:
    rclpy.init()
    node = MobileSamInteractiveNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
