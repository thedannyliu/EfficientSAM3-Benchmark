from __future__ import annotations

import json
import traceback
from collections import deque
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from sam_backend.backends import _import_required, _prepend_repo_path
from sam_backend.profiling import cuda_memory_mb, parameter_counts
from sam_backend.sam3_online import (
    initialize_sam3_online_state,
    prune_sam3_online_state,
    run_sam3_online_step,
)
from sam_backend.streaming import (
    left_panel_click_to_image_point,
    left_panel_drag_to_image_box,
    masks_to_mono8,
)
from sam_benchmark_ros.sam2_online_tracking_node import _overlay_multi_object
from sam_benchmark_ros.sam3_native_clip_node import (
    _draw_prompt,
    _format_float,
    _normalized_xywh,
    _safe_len,
    _scale_display,
    _stamp_delta_ms,
    _status_overlay,
)
from sam_benchmark_ros.text_prompt_ui import TextPromptEditor
from sam_benchmark_ros.video_recording import prompt_is_visible, stamp_to_seconds


class Sam3OnlineTrackingNode(Node):
    def __init__(self) -> None:
        super().__init__("sam3_online_tracking_node")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("result_topic", "/sam/result_json")
        self.declare_parameter("mask_topic", "/segmentation_mask")
        self.declare_parameter("segmented_image_topic", "/segmented_image")
        self.declare_parameter("overlay_topic", "/sam/overlay")
        self.declare_parameter("checkpoint_path", "checkpoints/sam3/sam3.pt")
        self.declare_parameter("external_repo", "external/efficientsam3")
        self.declare_parameter("version", "sam3")
        self.declare_parameter("instinctsam_text_checkpoint", "")
        self.declare_parameter("instinctsam_vision_checkpoint", "")
        self.declare_parameter("prompt_mode", "text")
        self.declare_parameter("prompt", "monitor")
        self.declare_parameter("text_prompt_topic", "/sam/text_prompt")
        self.declare_parameter("input_queue_size", 1)
        self.declare_parameter("image_qos_reliability", "best_effort")
        self.declare_parameter("memory_history_size", 32)
        self.declare_parameter("nms_backend", "auto")
        self.declare_parameter("window_name", "SAM3 Online Memory")
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
        self.prompt_mode = str(self.get_parameter("prompt_mode").value).strip().lower()
        if self.prompt_mode not in {"text", "point", "box", "interactive"}:
            raise ValueError("prompt_mode must be one of: text, point, box, interactive")
        self.geometry_prompt_mode = (
            self.prompt_mode if self.prompt_mode in {"point", "box"} else "interactive"
        )
        self.prompt_text = str(self.get_parameter("prompt").value).strip()
        if self.prompt_mode == "text" and not self.prompt_text:
            raise ValueError("prompt must not be empty in text mode")
        self.input_queue_size = max(1, int(self.get_parameter("input_queue_size").value))
        self.memory_history_size = max(
            1, int(self.get_parameter("memory_history_size").value)
        )
        self.window_name = str(self.get_parameter("window_name").value)
        self.display_scale = float(self.get_parameter("display_scale").value)
        self.display_max_width = int(self.get_parameter("display_max_width").value)
        self.enable_display = bool(self.get_parameter("enable_display").value)
        self.auto_start = bool(self.get_parameter("auto_start").value)
        self.initial_point_x = float(self.get_parameter("initial_point_x").value)
        self.initial_point_y = float(self.get_parameter("initial_point_y").value)
        self.initial_point_normalized = bool(
            self.get_parameter("initial_point_normalized").value
        )
        self.box_drag_min_pixels = float(
            self.get_parameter("box_drag_min_pixels").value
        )
        self.prompt_display_seconds = float(
            self.get_parameter("prompt_display_seconds").value
        )
        self.current_display_scale = 1.0

        self.latest_frame: np.ndarray | None = None
        self.latest_header: Any | None = None
        self.latest_display: np.ndarray | None = None
        self.inference_state: dict[str, Any] | None = None
        self.active_prompt: dict[str, Any] | None = None
        self.prompt_display_start: float | None = None
        self.drag_start: tuple[float, float] | None = None
        self.frame_index = -1
        self.result_times: deque[float] = deque(maxlen=60)
        self.state = "waiting_for_frame" if self.prompt_mode == "text" else "waiting_for_prompt"
        self.text_prompt_editor = TextPromptEditor()

        self.predictor, self.model, self.torch_module, self.backend_name = self._build_model()
        self.params = parameter_counts(self.model)
        self.image_mean = self._tensor(list(self.model.image_mean))[:, None, None]
        self.image_std = self._tensor(list(self.model.image_std))[:, None, None]
        if self.torch_module.cuda.is_available():
            self.torch_module.cuda.reset_peak_memory_stats()

        image_topic = str(self.get_parameter("image_topic").value)
        self.result_publisher = self.create_publisher(
            String, str(self.get_parameter("result_topic").value), 10
        )
        self.mask_publisher = self.create_publisher(
            Image, str(self.get_parameter("mask_topic").value), 10
        )
        self.segmented_image_publisher = self.create_publisher(
            Image, str(self.get_parameter("segmented_image_topic").value), 10
        )
        overlay_topic = str(self.get_parameter("overlay_topic").value)
        self.overlay_publisher = (
            self.create_publisher(Image, overlay_topic, 10) if overlay_topic else None
        )
        self.subscription = self.create_subscription(
            Image, image_topic, self.on_image, self._image_qos()
        )
        text_prompt_topic = str(self.get_parameter("text_prompt_topic").value)
        self.text_prompt_subscription = self.create_subscription(
            String, text_prompt_topic, self.on_text_prompt, 10
        )
        self.text_prompt_publisher = self.create_publisher(String, text_prompt_topic, 10)
        display_fps = float(self.get_parameter("display_fps").value)
        self.timer = (
            self.create_timer(1.0 / display_fps, self.display)
            if self.enable_display
            else None
        )

        if self.enable_display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.window_name, self.on_mouse)
        self.get_logger().info(
            f"listening on {image_topic}; native SAM3 online memory tracking processes "
            f"each arriving frame; prompt_mode={self.prompt_mode} "
            f"input_queue_size={self.input_queue_size} "
            f"memory_history_size={self.memory_history_size}; press t for text, "
            "click for point, or drag for box"
        )

    def _build_model(self) -> tuple[Any, Any, Any, str]:
        external_repo = str(self.get_parameter("external_repo").value)
        checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        version = str(self.get_parameter("version").value)
        if version != "sam3":
            raise ValueError("SAM3 online tracking currently requires version:=sam3")
        _prepend_repo_path(external_repo)
        torch_module = _import_required("torch")
        builder = _import_required("sam3.model_builder")
        if not hasattr(builder, "build_sam3_video_predictor"):
            raise RuntimeError(
                "SAM3 online tracking requires build_sam3_video_predictor from external/efficientsam3"
            )

        from sam_backend.sam3_runtime import configure_sam3_nms

        self.nms_backend = configure_sam3_nms(
            torch_module, str(self.get_parameter("nms_backend").value)
        )
        self.get_logger().info(f"SAM3 mask NMS backend: {self.nms_backend}")
        if self.nms_backend == "torch":
            self.get_logger().info("SAM3 connected components backend: cpu")

        predictor = builder.build_sam3_video_predictor(
            checkpoint_path=checkpoint_path,
            gpus_to_use=[0],
        )
        model = predictor.model
        text_checkpoint = (
            str(self.get_parameter("instinctsam_text_checkpoint").value) or None
        )
        vision_checkpoint = (
            str(self.get_parameter("instinctsam_vision_checkpoint").value) or None
        )
        backend_name = "sam3"
        if text_checkpoint or vision_checkpoint:
            from sam_backend.instinctsam import install_instinctsam_video_components

            install_instinctsam_video_components(
                model,
                builder,
                text_checkpoint=text_checkpoint,
                vision_checkpoint=vision_checkpoint,
                device=str(predictor.device),
            )
            backend_name = "instinctsam"
        return predictor, model, torch_module, backend_name

    def _image_qos(self) -> QoSProfile:
        value = str(self.get_parameter("image_qos_reliability").value).strip().lower()
        if value in {"best_effort", "besteffort", "best-effort", "0", "false"}:
            reliability = ReliabilityPolicy.BEST_EFFORT
        elif value in {"reliable", "1", "true"}:
            reliability = ReliabilityPolicy.RELIABLE
        else:
            raise ValueError("image_qos_reliability must be 'best_effort' or 'reliable'")
        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self.input_queue_size,
            reliability=reliability,
            durability=DurabilityPolicy.VOLATILE,
        )

    def on_text_prompt(self, msg: String) -> None:
        prompt = msg.data.strip()
        if not prompt:
            self.get_logger().warning("ignoring empty runtime text prompt")
            return
        self.prompt_text = prompt
        self.prompt_mode = "text"
        if self.latest_frame is None or self.latest_header is None:
            self.state = "waiting_for_frame"
            return
        self._start_tracking(
            {"prompt_mode": "text", "prompt_text": prompt},
            self.latest_frame,
            self.latest_header,
        )

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if (
            self.latest_frame is None
            or self.latest_header is None
            or self.state == "processing"
            or self.text_prompt_editor.active
        ):
            return
        scale = self.current_display_scale if self.current_display_scale > 0 else 1.0
        point = left_panel_click_to_image_point(
            x / scale, y / scale, self.latest_frame.shape[:2]
        )
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
        box = left_panel_drag_to_image_box(
            start,
            point,
            self.latest_frame.shape[:2],
            min_size=self.box_drag_min_pixels,
        )
        if self.geometry_prompt_mode == "box" and box is None:
            self.get_logger().info("drag a larger box to start SAM3 box tracking")
            return
        if self.geometry_prompt_mode == "point" or box is None:
            prompt = {"prompt_mode": "point", "point": point, "label": 1}
        else:
            prompt = {"prompt_mode": "box", "box": box}
        self.prompt_mode = prompt["prompt_mode"]
        self._start_tracking(prompt, self.latest_frame, self.latest_header)

    def on_image(self, msg: Image) -> None:
        if self.state == "processing":
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.latest_frame = frame
        self.latest_header = msg.header

        if self.state == "waiting_for_frame" and self.prompt_mode == "text":
            self._start_tracking(
                {"prompt_mode": "text", "prompt_text": self.prompt_text},
                frame,
                msg.header,
            )
            return
        if self.auto_start and self.state == "waiting_for_prompt":
            self.auto_start = False
            height, width = frame.shape[:2]
            if self.initial_point_normalized:
                point = (
                    self.initial_point_x * float(width),
                    self.initial_point_y * float(height),
                )
            else:
                point = (self.initial_point_x, self.initial_point_y)
            self._start_tracking(
                {"prompt_mode": "point", "point": point, "label": 1}, frame, msg.header
            )
            return
        if self.state == "waiting_for_prompt":
            display, scale = _scale_display(
                _status_overlay(frame, "Press t for text, click point, or drag box"),
                self.display_scale,
                self.display_max_width,
            )
            self.latest_display = display
            self.current_display_scale = scale
            return
        if self.state == "error":
            display, scale = _scale_display(
                _status_overlay(frame, "Tracking error; press r or enter a new prompt"),
                self.display_scale,
                self.display_max_width,
            )
            self.latest_display = display
            self.current_display_scale = scale
            return
        if self.state == "tracking":
            self._track_frame(frame, msg.header)

    def _start_tracking(
        self, prompt: dict[str, Any], frame: np.ndarray, header: Any
    ) -> None:
        self.state = "processing"
        try:
            self._sync()
            init_start = perf_counter()
            inference_state = self.model.init_state(
                resource_path=[PILImage.fromarray(frame).convert("RGB")]
            )
            initialize_sam3_online_state(inference_state)
            self._sync()
            init_ms = (perf_counter() - init_start) * 1000.0

            prompt_start = perf_counter()
            _, outputs = self._apply_prompt(inference_state, prompt)
            self._sync()
            prompt_ms = (perf_counter() - prompt_start) * 1000.0

            self.inference_state = inference_state
            self.active_prompt = prompt
            self.prompt_display_start = stamp_to_seconds(header.stamp)
            self.frame_index = 0
            self.result_times.clear()
            self._publish_frame(
                frame,
                header,
                0,
                outputs,
                prompt_ms,
                init_ms,
                prompt_ms,
            )
            self.state = "tracking"
            self.get_logger().info(
                f"started native SAM3 online tracking from {prompt['prompt_mode']} prompt"
            )
        except Exception:
            self.inference_state = None
            self.active_prompt = None
            self.state = "error"
            self.get_logger().error(
                f"failed to start SAM3 online tracking:\n{traceback.format_exc()}"
            )

    def _apply_prompt(
        self, inference_state: dict[str, Any], prompt: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        if prompt["prompt_mode"] == "text":
            return self.model.add_prompt(
                inference_state, frame_idx=0, text_str=prompt["prompt_text"]
            )
        if prompt["prompt_mode"] == "box":
            return self.model.add_prompt(
                inference_state,
                frame_idx=0,
                boxes_xywh=[
                    _normalized_xywh(
                        prompt["box"],
                        (
                            inference_state["orig_height"],
                            inference_state["orig_width"],
                        ),
                    )
                ],
                box_labels=[1],
            )
        return self.model.add_prompt(
            inference_state,
            frame_idx=0,
            points=[prompt["point"]],
            point_labels=[prompt.get("label", 1)],
            obj_id=1,
            rel_coordinates=False,
        )

    def _track_frame(self, frame: np.ndarray, header: Any) -> None:
        if self.inference_state is None or self.active_prompt is None:
            self.state = "waiting_for_prompt"
            return
        self.state = "processing"
        try:
            self._sync()
            start = perf_counter()
            frame_tensor = self._preprocess_frame(frame)
            frame_idx, outputs = run_sam3_online_step(
                self.model,
                self.inference_state,
                frame_tensor,
                self.torch_module,
            )
            self._sync()
            latency_ms = (perf_counter() - start) * 1000.0
            self.frame_index = frame_idx
            prune_sam3_online_state(
                self.inference_state, frame_idx, self.memory_history_size
            )
            self._publish_frame(frame, header, frame_idx, outputs, latency_ms, "", "")
            self.state = "tracking"
        except Exception:
            self.inference_state = None
            self.state = "error"
            self.get_logger().error(
                f"failed during SAM3 online tracking:\n{traceback.format_exc()}"
            )

    def _preprocess_frame(self, frame_rgb: np.ndarray) -> Any:
        image_size = int(self.model.image_size)
        image = PILImage.fromarray(frame_rgb).convert("RGB").resize((image_size, image_size))
        array = np.asarray(image).copy()
        tensor = self.torch_module.from_numpy(array).permute(2, 0, 1)
        tensor = tensor.to(
            device=self.model.device,
            dtype=self.torch_module.float16,
            non_blocking=True,
        )
        tensor /= 255.0
        tensor -= self.image_mean
        tensor /= self.image_std
        return tensor

    def _publish_frame(
        self,
        frame: np.ndarray,
        header: Any,
        frame_index: int,
        outputs: dict[str, Any],
        latency_ms: float,
        init_state_ms: float | str,
        add_prompt_ms: float | str,
    ) -> None:
        masks = outputs.get("out_binary_masks", np.zeros((0, *frame.shape[:2]), dtype=bool))
        obj_ids = [int(value) for value in outputs.get("out_obj_ids", [])]
        overlay = _overlay_multi_object(frame, masks, obj_ids)
        if (
            self.active_prompt is not None
            and self.active_prompt["prompt_mode"] in {"point", "box"}
            and prompt_is_visible(
                self.prompt_display_start, header.stamp, self.prompt_display_seconds
            )
        ):
            _draw_prompt(overlay, self.active_prompt)
        mask = masks_to_mono8(masks, frame.shape[:2])
        self.result_times.append(self.get_clock().now().nanoseconds / 1_000_000_000.0)
        memory = cuda_memory_mb(self.torch_module)
        prompt = self.active_prompt or {"prompt_mode": ""}
        result = {
            "frame_index": frame_index,
            "source_frame_index": frame_index,
            "stamp": {"sec": header.stamp.sec, "nanosec": header.stamp.nanosec},
            "frame_id": header.frame_id,
            "backend": self.backend_name,
            "stream_mode": "native_online",
            "tracking_state": "tracking",
            "object_count": len(obj_ids),
            "object_ids": ",".join(str(value) for value in obj_ids),
            "prompt_mode": prompt["prompt_mode"],
            "prompt_text": prompt.get("prompt_text", ""),
            "latency_ms": latency_ms,
            "init_state_ms": init_state_ms,
            "add_prompt_ms": add_prompt_ms,
            "callback_total_ms": latency_ms,
            "end_to_end_ms": self._end_to_end_ms(header),
            "tracking_fps": self._tracking_fps(),
            "mask_count": _safe_len(masks),
            "memory_history_size": self.memory_history_size,
            "queue_depth": self.input_queue_size,
            "nms_backend": self.nms_backend,
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
        display, scale = _scale_display(
            _display_with_metrics(overlay, result),
            self.display_scale,
            self.display_max_width,
        )
        self.latest_display = display
        self.current_display_scale = scale

    def display(self) -> None:
        if self.latest_display is None:
            return
        display = self.latest_display.copy()
        self.text_prompt_editor.draw(display)
        cv2.imshow(self.window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if self.text_prompt_editor.active:
            prompt = self.text_prompt_editor.handle_key(key)
            if prompt is not None:
                self.text_prompt_publisher.publish(String(data=prompt))
            return
        if key == ord("t"):
            self.text_prompt_editor.start(self.prompt_text)
            return
        if key in {27, ord("q")}:
            raise SystemExit
        if key == ord("r"):
            self._clear_state()
            self.get_logger().info("reset native SAM3 online tracking state")

    def _clear_state(self) -> None:
        self.inference_state = None
        self.active_prompt = None
        self.prompt_display_start = None
        self.drag_start = None
        self.frame_index = -1
        self.result_times.clear()
        self.state = "waiting_for_frame" if self.prompt_mode == "text" else "waiting_for_prompt"

    def _tensor(self, values: list[float]) -> Any:
        return self.torch_module.tensor(
            values,
            dtype=self.torch_module.float16,
            device=self.model.device,
        )

    def _sync(self) -> None:
        if self.torch_module.cuda.is_available():
            self.torch_module.cuda.synchronize()

    def _end_to_end_ms(self, header: Any) -> float:
        now = self.get_clock().now().to_msg()
        return _stamp_delta_ms(
            header.stamp.sec, header.stamp.nanosec, now.sec, now.nanosec
        )

    def _tracking_fps(self) -> float | str:
        if len(self.result_times) < 2:
            return ""
        duration = self.result_times[-1] - self.result_times[0]
        return (len(self.result_times) - 1) / duration if duration > 0 else ""

    def destroy_node(self) -> bool:
        if self.enable_display:
            cv2.destroyAllWindows()
        if hasattr(self.predictor, "shutdown"):
            self.predictor.shutdown()
        return super().destroy_node()


def _display_with_metrics(overlay_rgb: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    panel = np.full((overlay_bgr.shape[0], 360, 3), 24, dtype=np.uint8)
    lines = [
        f"{result.get('backend', 'sam3')} online memory",
        f"Prompt: {result.get('prompt_mode', '')}",
        f"Objects: {result.get('object_count', 0)}",
        f"Frame: {result.get('frame_index', '')}",
        f"Latency: {_format_float(result.get('latency_ms'))} ms",
        f"FPS: {_format_float(result.get('tracking_fps'))}",
        f"CUDA: {_format_float(result.get('cuda_allocated_mb'))} MB",
        "t: text  click: point  drag: box",
    ]
    y = 34
    for idx, line in enumerate(lines):
        scale = 0.66 if idx == 0 else 0.52
        cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += 30 if idx == 0 else 24
    return np.hstack([overlay_bgr, panel])


def main() -> None:
    rclpy.init()
    node = Sam3OnlineTrackingNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
