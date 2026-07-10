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

from sam_benchmark_ros.video_recording import prompt_is_visible, stamp_to_seconds

from sam_backend.backends import _import_required, _prepend_repo_path
from sam_backend.overlay import overlay_prediction
from sam_backend.profiling import cuda_memory_mb, parameter_counts
from sam_backend.streaming import (
    left_panel_click_to_image_point,
    left_panel_drag_to_image_box,
    masks_to_mono8,
    parse_text_prompts,
)


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
        self.declare_parameter("prompts", "")
        self.declare_parameter("clip_frames", 120)
        self.declare_parameter("frame_dir", "results/thor/ros_camera/sam3_native_clip/frames")
        self.declare_parameter("version", "sam3")
        self.declare_parameter("prompt_mode", "text")
        self.declare_parameter("window_name", "SAM3 Native Memory")
        self.declare_parameter("display_fps", 30.0)
        self.declare_parameter("display_scale", 1.0)
        self.declare_parameter("display_max_width", 0)
        self.declare_parameter("enable_display", True)
        self.declare_parameter("auto_start", False)
        self.declare_parameter("initial_point_x", 0.5)
        self.declare_parameter("initial_point_y", 0.5)
        self.declare_parameter("initial_point_normalized", True)
        self.declare_parameter("box_drag_min_pixels", 5.0)
        self.declare_parameter("prompt_display_seconds", 0.5)

        self.bridge = CvBridge()
        self.prompt_mode = str(self.get_parameter("prompt_mode").value)
        if self.prompt_mode not in {"text", "point", "box", "interactive"}:
            raise ValueError("prompt_mode must be one of: text, point, box, interactive")
        self.prompt = str(self.get_parameter("prompt").value)
        self.prompt_texts = parse_text_prompts(self.prompt, str(self.get_parameter("prompts").value))
        if self.prompt_mode == "text" and not self.prompt_texts:
            raise ValueError("prompt or prompts must provide at least one text prompt")
        self.clip_frames = int(self.get_parameter("clip_frames").value)
        if self.clip_frames <= 0:
            raise ValueError("clip_frames must be positive")
        self.frame_dir = Path(str(self.get_parameter("frame_dir").value))
        self.window_name = str(self.get_parameter("window_name").value)
        self.display_scale = float(self.get_parameter("display_scale").value)
        self.display_max_width = int(self.get_parameter("display_max_width").value)
        self.enable_display = bool(self.get_parameter("enable_display").value)
        self.auto_start = bool(self.get_parameter("auto_start").value)
        self.initial_point_x = float(self.get_parameter("initial_point_x").value)
        self.initial_point_y = float(self.get_parameter("initial_point_y").value)
        self.initial_point_normalized = bool(self.get_parameter("initial_point_normalized").value)
        self.box_drag_min_pixels = float(self.get_parameter("box_drag_min_pixels").value)
        self.prompt_display_seconds = float(self.get_parameter("prompt_display_seconds").value)
        self.current_display_scale = 1.0

        self.latest_frame: np.ndarray | None = None
        self.latest_header: Any | None = None
        self.latest_display: np.ndarray | None = None
        self.frames: list[np.ndarray] = []
        self.headers: list[Any] = []
        self.geometry_prompt: dict[str, Any] | None = None
        self.prompt_display_start: float | None = None
        self.drag_start: tuple[float, float] | None = None
        self.result_times: deque[float] = deque(maxlen=60)
        self.state = "waiting_for_prompt" if self.prompt_mode != "text" else "capturing"
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
        self.subscription = self.create_subscription(Image, image_topic, self.on_image, 1)
        display_fps = float(self.get_parameter("display_fps").value)
        self.timer = self.create_timer(1.0 / display_fps, self.display) if self.enable_display else None

        if self.enable_display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.window_name, self.on_mouse)

        if self.prompt_mode == "text":
            self.get_logger().info(
                f"capturing {self.clip_frames} frames from {image_topic}; "
                f"SAM3 text native tracking will run after the clip is materialized"
            )
        elif self.auto_start:
            self.get_logger().info(f"listening on {image_topic}; auto-starting native SAM3 point tracking")
        else:
            self.get_logger().info(
                f"listening on {image_topic}; click for point or drag for box; clip_frames={self.clip_frames}"
            )

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if self.prompt_mode == "text" or self.latest_frame is None or self.state == "processing":
            return
        scale = self.current_display_scale if self.current_display_scale > 0 else 1.0
        point = left_panel_click_to_image_point(x / scale, y / scale, self.latest_frame.shape[:2])
        if point is None:
            if event == cv2.EVENT_LBUTTONUP:
                self.drag_start = None
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = point
            return
        if event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None and flags & cv2.EVENT_FLAG_LBUTTON:
            return
        if event != cv2.EVENT_LBUTTONUP or self.drag_start is None:
            return

        start = self.drag_start
        self.drag_start = None
        box = left_panel_drag_to_image_box(start, point, self.latest_frame.shape[:2], min_size=self.box_drag_min_pixels)
        if self.prompt_mode == "box" and box is None:
            self.get_logger().info("drag a larger box to start SAM3 box tracking")
            return
        if self.prompt_mode == "point" or box is None:
            self._start_capture({"prompt_mode": "point", "point": point, "label": 1})
            self.get_logger().info(f"received native SAM3 point prompt x={point[0]:.1f} y={point[1]:.1f}")
        else:
            self._start_capture({"prompt_mode": "box", "box": box})
            self.get_logger().info(
                f"received native SAM3 box prompt x1={box[0]:.1f} y1={box[1]:.1f} x2={box[2]:.1f} y2={box[3]:.1f}"
            )

    def on_image(self, msg: Image) -> None:
        if self.state == "processing":
            return
        frame_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.latest_frame = frame_rgb
        self.latest_header = msg.header

        if self.prompt_mode != "text" and self.auto_start and self.state == "waiting_for_prompt":
            self.auto_start = False
            height, width = frame_rgb.shape[:2]
            if self.initial_point_normalized:
                point = (self.initial_point_x * float(width), self.initial_point_y * float(height))
            else:
                point = (self.initial_point_x, self.initial_point_y)
            self._start_capture({"prompt_mode": "point", "point": point, "label": 1})
            return

        if self.state == "waiting_for_prompt":
            self.latest_display = _scale_display(
                _status_overlay(frame_rgb, "Click point or drag box"),
                self.display_scale,
                self.display_max_width,
            )[0]
            return

        if self.state == "capturing":
            self.frames.append(frame_rgb.copy())
            self.headers.append(msg.header)
            if len(self.frames) % 30 == 0 or len(self.frames) == self.clip_frames:
                self.get_logger().info(f"captured {len(self.frames)}/{self.clip_frames} frames")
        if self.state == "capturing" and len(self.frames) >= self.clip_frames:
            self.processing_started = True
            self.state = "processing"
            self._process_clip()

    def _start_capture(self, prompt: dict[str, Any]) -> None:
        if self.latest_frame is None or self.latest_header is None:
            return
        self.geometry_prompt = prompt
        self.prompt_display_start = stamp_to_seconds(self.latest_header.stamp)
        self.frames = [self.latest_frame.copy()]
        self.headers = [self.latest_header]
        self.state = "capturing"

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
            self._sync()
            start_init = perf_counter()
            response = self.predictor.handle_request({"type": "start_session", "resource_path": str(self.frame_dir)})
            self._sync()
            init_ms = (perf_counter() - start_init) * 1000.0
            session_id = response["session_id"]
            start_prompt = perf_counter()
            prompt_record = self._add_prompt(session_id)
            self._sync()
            add_prompt_ms = (perf_counter() - start_prompt) * 1000.0
            iterator = self.predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "start_frame_index": 0,
                    "max_frame_num_to_track": self.clip_frames,
                }
            )
            while True:
                start = perf_counter()
                try:
                    response = next(iterator)
                    self._sync()
                except StopIteration:
                    break
                latency_ms = (perf_counter() - start) * 1000.0
                frame_index = int(response.get("frame_index", response.get("frame_idx", 0)))
                if frame_index >= len(self.frames):
                    continue
                masks = _sam3_output_masks(response.get("outputs", {}), self.frames[frame_index].shape[:2])
                self._publish_frame(frame_index, masks, latency_ms, init_ms, add_prompt_ms, prompt_record)
        finally:
            if session_id is not None:
                self.predictor.handle_request({"type": "close_session", "session_id": session_id})
            self.state = "done"
            self.get_logger().info("native SAM3 clip tracking complete; click or drag again to reset and run another clip")

    def _add_prompt(self, session_id: str) -> dict[str, Any]:
        if self.prompt_mode == "text":
            for prompt in self.prompt_texts:
                self.predictor.handle_request(
                    {"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": prompt}
                )
            return {"prompt_mode": "native_text", "prompt_text": ",".join(self.prompt_texts)}
        if self.geometry_prompt is None:
            raise RuntimeError("geometry prompt is missing")
        prompt = self.geometry_prompt
        if prompt["prompt_mode"] == "box":
            box_xywh = _normalized_xywh(prompt["box"], self.frames[0].shape[:2])
            self.predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "bounding_boxes": [box_xywh],
                    "bounding_box_labels": [1],
                }
            )
        else:
            self.predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "points": [prompt["point"]],
                    "point_labels": [prompt.get("label", 1)],
                    "obj_id": 1,
                    "rel_coordinates": False,
                }
            )
        return prompt

    def _publish_frame(
        self,
        frame_index: int,
        masks: Any,
        latency_ms: float,
        init_ms: float,
        add_prompt_ms: float,
        prompt: dict[str, Any],
    ) -> None:
        frame = self.frames[frame_index]
        header = self.headers[frame_index]
        mask = masks_to_mono8(masks, frame.shape[:2])
        overlay = overlay_prediction(frame, masks)
        if prompt_is_visible(self.prompt_display_start, header.stamp, self.prompt_display_seconds):
            _draw_prompt(overlay, prompt)
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
            "prompt_mode": prompt["prompt_mode"],
            "prompt_text": prompt.get("prompt_text", ""),
            "prompt_count": len(self.prompt_texts) if prompt["prompt_mode"] == "native_text" else 1,
            "point_x": prompt.get("point", ("", ""))[0] if prompt["prompt_mode"] == "point" else "",
            "point_y": prompt.get("point", ("", ""))[1] if prompt["prompt_mode"] == "point" else "",
            "box_x1": prompt.get("box", ("", "", "", ""))[0] if prompt["prompt_mode"] == "box" else "",
            "box_y1": prompt.get("box", ("", "", "", ""))[1] if prompt["prompt_mode"] == "box" else "",
            "box_x2": prompt.get("box", ("", "", "", ""))[2] if prompt["prompt_mode"] == "box" else "",
            "box_y2": prompt.get("box", ("", "", "", ""))[3] if prompt["prompt_mode"] == "box" else "",
            "latency_ms": latency_ms,
            "init_state_ms": init_ms if frame_index == 0 else "",
            "add_prompt_ms": add_prompt_ms if frame_index == 0 else "",
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
        display, scale = _scale_display(_display_with_metrics(overlay, result), self.display_scale, self.display_max_width)
        self.latest_display = display
        self.current_display_scale = scale

    def display(self) -> None:
        if self.latest_display is None:
            return
        cv2.imshow(self.window_name, self.latest_display)
        key = cv2.waitKey(1) & 0xFF
        if key in {27, ord("q")}:
            raise SystemExit
        if key == ord("r"):
            self.state = "waiting_for_prompt" if self.prompt_mode != "text" else "capturing"
            self.processing_started = False
            self.geometry_prompt = None
            self.prompt_display_start = None
            self.frames = []
            self.headers = []
            self.drag_start = None
            self.latest_display = None
            self.get_logger().info("reset native SAM3 tracking state")

    def _sync(self) -> None:
        cuda = getattr(self.torch_module, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.synchronize()

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
        if self.enable_display:
            cv2.destroyAllWindows()
        if hasattr(self, "predictor") and hasattr(self.predictor, "shutdown"):
            self.predictor.shutdown()
        return super().destroy_node()


def _sam3_output_masks(outputs: dict[str, Any], frame_hw: tuple[int, int]) -> Any:
    for key in ("out_binary_masks", "pred_masks", "masks"):
        if key in outputs:
            return outputs[key]
    return np.zeros((0, frame_hw[0], frame_hw[1]), dtype=np.uint8)


def _normalized_xywh(box_xyxy: tuple[float, float, float, float], frame_hw: tuple[int, int]) -> list[float]:
    height, width = frame_hw
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    return [
        max(0.0, min(1.0, x1 / float(width))),
        max(0.0, min(1.0, y1 / float(height))),
        max(0.0, min(1.0, (x2 - x1 + 1.0) / float(width))),
        max(0.0, min(1.0, (y2 - y1 + 1.0) / float(height))),
    ]


def _draw_prompt(image_rgb: np.ndarray, prompt: dict[str, Any]) -> None:
    if prompt["prompt_mode"] == "box":
        box = prompt["box"]
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (30, 220, 255), 1, cv2.LINE_AA)
    elif prompt["prompt_mode"] == "point":
        point = prompt["point"]
        x = int(round(point[0]))
        y = int(round(point[1]))
        cv2.circle(image_rgb, (x, y), 9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(image_rgb, (x, y), 5, (255, 80, 30), -1, cv2.LINE_AA)


def _status_overlay(frame_rgb: np.ndarray, status: str) -> np.ndarray:
    overlay = frame_rgb.copy()
    cv2.putText(overlay, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(overlay, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def _display_with_metrics(overlay_rgb: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    panel = np.full((overlay_bgr.shape[0], 360, 3), 24, dtype=np.uint8)
    lines = [
        "SAM3 native memory",
        f"Prompt: {result.get('prompt_mode', '')}",
        f"Frame: {result.get('frame_index', '')}",
        f"State: {result.get('tracking_state', '')}",
        f"Latency: {_format_float(result.get('latency_ms'))} ms",
        f"FPS: {_format_float(result.get('tracking_fps'))}",
        f"CUDA: {_format_float(result.get('cuda_allocated_mb'))} MB",
    ]
    y = 34
    for idx, line in enumerate(lines):
        scale = 0.66 if idx == 0 else 0.56
        cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += 30 if idx == 0 else 24
    return np.hstack([overlay_bgr, panel])


def _scale_display(image: np.ndarray, display_scale: float, display_max_width: int) -> tuple[np.ndarray, float]:
    scale = display_scale if display_scale > 0 else 1.0
    if display_max_width > 0 and image.shape[1] * scale > display_max_width:
        scale = display_max_width / float(image.shape[1])
    if scale == 1.0:
        return image, 1.0
    width = max(1, int(round(image.shape[1] * scale)))
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), scale


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
