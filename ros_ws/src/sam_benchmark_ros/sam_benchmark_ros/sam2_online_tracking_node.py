from __future__ import annotations

import json
import traceback
from collections import OrderedDict, deque
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

from sam_backend.overlay import overlay_prediction
from sam_backend.profiling import cuda_memory_mb, parameter_counts
from sam_backend.streaming import (
    left_panel_click_to_image_point,
    left_panel_drag_to_image_box,
    masks_to_mono8,
)
from sam_benchmark_ros.sam2_native_clip_node import (
    _backend_label,
    _binary_masks,
    _build_predictor_from_params,
    _draw_prompt,
    _format_float,
    _safe_len,
    _scale_display,
    _stamp_delta_ms,
    _status_overlay,
)
from sam_benchmark_ros.video_recording import prompt_is_visible, stamp_to_seconds


class Sam2OnlineTrackingNode(Node):
    def __init__(self) -> None:
        super().__init__("sam2_online_tracking_node")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("result_topic", "/sam/result_json")
        self.declare_parameter("mask_topic", "/segmentation_mask")
        self.declare_parameter("segmented_image_topic", "/segmented_image")
        self.declare_parameter("overlay_topic", "/sam/overlay")
        self.declare_parameter("checkpoint_path", "checkpoints/sam2/sam2.1_hiera_large.pt")
        self.declare_parameter("sam2_checkpoint_path", "")
        self.declare_parameter("model_config", "configs/sam2.1/sam2.1_hiera_l.yaml")
        self.declare_parameter("external_repo", "external/sam2")
        self.declare_parameter("sam2_distill_root", "external/SAM2-Distillation-Pipeline")
        self.declare_parameter("model_kind", "sam2")
        self.declare_parameter("tinyvit_checkpoint", "")
        self.declare_parameter("tinyvit_model_name", "tiny_vit_21m_512.dist_in22k_ft_in1k")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("input_queue_size", 3)
        self.declare_parameter("image_qos_reliability", "best_effort")
        self.declare_parameter("memory_history_size", 32)
        self.declare_parameter("window_name", "SAM2 Online Memory")
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
        self.input_queue_size = max(1, int(self.get_parameter("input_queue_size").value))
        self.memory_history_size = max(1, int(self.get_parameter("memory_history_size").value))
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
        self.prompt: dict[str, Any] | None = None
        self.prompt_display_start: float | None = None
        self.drag_start: tuple[float, float] | None = None
        self.state = "waiting_for_prompt"
        self.frame_index = -1
        self.inference_state: dict[str, Any] | None = None
        self.original_frames: dict[int, np.ndarray] = {}
        self.headers: dict[int, Any] = {}
        self.result_times: deque[float] = deque(maxlen=60)

        self.predictor, self.torch_module, self.load_summary = _build_predictor_from_params(self)
        self.params = parameter_counts(self.predictor)
        self.image_mean = self._tensor([0.485, 0.456, 0.406])[:, None, None]
        self.image_std = self._tensor([0.229, 0.224, 0.225])[:, None, None]
        if self.torch_module.cuda.is_available():
            self.torch_module.cuda.reset_peak_memory_stats()

        image_topic = str(self.get_parameter("image_topic").value)
        image_qos = self._image_qos()
        result_topic = str(self.get_parameter("result_topic").value)
        mask_topic = str(self.get_parameter("mask_topic").value)
        segmented_image_topic = str(self.get_parameter("segmented_image_topic").value)
        overlay_topic = str(self.get_parameter("overlay_topic").value)
        display_fps = float(self.get_parameter("display_fps").value)

        self.result_publisher = self.create_publisher(String, result_topic, 10)
        self.mask_publisher = self.create_publisher(Image, mask_topic, 10)
        self.segmented_image_publisher = self.create_publisher(Image, segmented_image_topic, 10)
        self.overlay_publisher = self.create_publisher(Image, overlay_topic, 10) if overlay_topic else None
        self.subscription = self.create_subscription(Image, image_topic, self.on_image, image_qos)
        self.timer = self.create_timer(1.0 / display_fps, self.display) if self.enable_display else None

        if self.enable_display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.window_name, self.on_mouse)
        self.get_logger().info(
            f"listening on {image_topic}; click for point or drag for box; "
            f"input_queue_size={self.input_queue_size} "
            f"image_qos_reliability={self.get_parameter('image_qos_reliability').value} "
            f"memory_history_size={self.memory_history_size}"
        )

    def _image_qos(self) -> QoSProfile:
        reliability_value = str(self.get_parameter("image_qos_reliability").value).strip().lower()
        if reliability_value in {"reliable", "1", "true"}:
            reliability = ReliabilityPolicy.RELIABLE
        elif reliability_value in {"best_effort", "besteffort", "best-effort", "0", "false"}:
            reliability = ReliabilityPolicy.BEST_EFFORT
        else:
            raise ValueError("image_qos_reliability must be 'best_effort' or 'reliable'")
        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self.input_queue_size,
            reliability=reliability,
            durability=DurabilityPolicy.VOLATILE,
        )

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if self.latest_frame is None or self.state == "processing":
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
        if box is None:
            if self._reset_and_start({"prompt_mode": "point", "point": point, "label": 1}):
                self.get_logger().info(f"reset online SAM2 tracking from point x={point[0]:.1f} y={point[1]:.1f}")
        else:
            if self._reset_and_start({"prompt_mode": "box", "box": box}):
                self.get_logger().info(
                    f"reset online SAM2 tracking from box x1={box[0]:.1f} y1={box[1]:.1f} x2={box[2]:.1f} y2={box[3]:.1f}"
                )

    def on_image(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.latest_frame = frame
        self.latest_header = msg.header

        if self.auto_start and self.state == "waiting_for_prompt":
            self.auto_start = False
            height, width = frame.shape[:2]
            if self.initial_point_normalized:
                point = (self.initial_point_x * float(width), self.initial_point_y * float(height))
            else:
                point = (self.initial_point_x, self.initial_point_y)
            self._reset_and_start({"prompt_mode": "point", "point": point, "label": 1})
            return

        if self.state == "waiting_for_prompt":
            self.latest_display = _scale_display(_status_overlay(frame, "Click point or drag box"), self.display_scale, self.display_max_width)[0]
            return

        if self.state != "tracking":
            return

        self._track_frame(frame, msg.header)

    def _reset_and_start(self, prompt: dict[str, Any]) -> bool:
        if self.latest_frame is None or self.latest_header is None:
            return False
        self.state = "processing"
        self.prompt = prompt
        self.prompt_display_start = stamp_to_seconds(self.latest_header.stamp)
        self.frame_index = 0
        self.original_frames = {0: self.latest_frame.copy()}
        self.headers = {0: self.latest_header}
        self.inference_state = self._new_inference_state(self.latest_frame)

        try:
            self._sync()
            start = perf_counter()
            if prompt["prompt_mode"] == "box":
                _, obj_ids, video_res_masks = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=0,
                    obj_id=1,
                    box=np.asarray(prompt["box"], dtype=np.float32),
                )
            else:
                _, obj_ids, video_res_masks = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=0,
                    obj_id=1,
                    points=np.asarray([prompt["point"]], dtype=np.float32),
                    labels=np.asarray([prompt.get("label", 1)], dtype=np.int32),
                )
            self.predictor.propagate_in_video_preflight(self.inference_state)
            self._sync()
            add_prompt_ms = (perf_counter() - start) * 1000.0
            masks = _binary_masks(video_res_masks)
            self._publish_frame(0, masks, obj_ids, add_prompt_ms, add_prompt_ms, self.prompt)
            self.state = "tracking"
            return True
        except Exception:
            self.state = "waiting_for_prompt"
            self.inference_state = None
            self.get_logger().error(f"failed to start online SAM2 tracking:\n{traceback.format_exc()}")
            return False

    def _track_frame(self, frame: np.ndarray, header: Any) -> None:
        if self.inference_state is None or self.prompt is None:
            self.state = "waiting_for_prompt"
            return
        self.state = "processing"
        frame_idx = self.frame_index + 1
        self.frame_index = frame_idx
        self.original_frames[frame_idx] = frame.copy()
        self.headers[frame_idx] = header
        self.inference_state["images"].append(self._preprocess_frame(frame))
        self.inference_state["num_frames"] = frame_idx + 1

        try:
            self._sync()
            start = perf_counter()
            obj_ids, video_res_masks = self._run_online_step(frame_idx)
            self._sync()
            latency_ms = (perf_counter() - start) * 1000.0
            self._prune_history(frame_idx)
            masks = _binary_masks(video_res_masks)
            self._publish_frame(frame_idx, masks, obj_ids, latency_ms, "", self.prompt)
        except Exception:
            self.state = "waiting_for_prompt"
            self.inference_state = None
            self.get_logger().error(f"failed during online SAM2 tracking:\n{traceback.format_exc()}")
        finally:
            if self.state == "processing":
                self.state = "tracking"

    def _run_online_step(self, frame_idx: int) -> tuple[list[Any], Any]:
        assert self.inference_state is not None
        obj_ids = self.inference_state["obj_ids"]
        batch_size = self.predictor._get_obj_num(self.inference_state)
        if self._uses_global_output_dict():
            with self.torch_module.inference_mode():
                output_dict = self.inference_state["output_dict"]
                current_out, pred_masks = self.predictor._run_single_frame_inference(
                    inference_state=self.inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                )
                output_dict["non_cond_frame_outputs"][frame_idx] = current_out
                self.predictor._add_output_per_object(
                    self.inference_state,
                    frame_idx,
                    current_out,
                    "non_cond_frame_outputs",
                )
                self.inference_state["frames_already_tracked"][frame_idx] = {"reverse": False}
            _, video_res_masks = self.predictor._get_orig_video_res_output(self.inference_state, pred_masks)
            return obj_ids, video_res_masks

        pred_masks_per_obj = []
        with self.torch_module.inference_mode():
            for obj_idx in range(batch_size):
                obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
                current_out, pred_masks = self.predictor._run_single_frame_inference(
                    inference_state=self.inference_state,
                    output_dict=obj_output_dict,
                    frame_idx=frame_idx,
                    batch_size=1,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                )
                obj_output_dict["non_cond_frame_outputs"][frame_idx] = current_out
                self.inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": False}
                self.inference_state["frames_already_tracked"][frame_idx] = {"reverse": False}
                if obj_idx == 0 and "output_dict" in self.inference_state:
                    self.inference_state["output_dict"]["non_cond_frame_outputs"][frame_idx] = current_out
                pred_masks_per_obj.append(pred_masks)

        if len(pred_masks_per_obj) > 1:
            all_pred_masks = self.torch_module.cat(pred_masks_per_obj, dim=0)
        else:
            all_pred_masks = pred_masks_per_obj[0]
        _, video_res_masks = self.predictor._get_orig_video_res_output(self.inference_state, all_pred_masks)
        return obj_ids, video_res_masks

    def _uses_global_output_dict(self) -> bool:
        external_repo = str(self.get_parameter("external_repo").value).lower()
        return "edgetam" in external_repo

    def _new_inference_state(self, frame: np.ndarray) -> dict[str, Any]:
        compute_device = self.predictor.device
        height, width = frame.shape[:2]
        return {
            "images": [self._preprocess_frame(frame)],
            "num_frames": 1,
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
            "video_height": int(height),
            "video_width": int(width),
            "device": compute_device,
            "storage_device": compute_device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict": {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {
                "cond_frame_outputs": set(),
                "non_cond_frame_outputs": set(),
            },
            "tracking_has_started": False,
            "frames_already_tracked": {},
            "frames_tracked_per_obj": {},
        }

    def _preprocess_frame(self, frame_rgb: np.ndarray) -> Any:
        image_size = int(self.predictor.image_size)
        image = PILImage.fromarray(frame_rgb).convert("RGB").resize((image_size, image_size))
        array = np.asarray(image).copy()
        if array.dtype != np.uint8:
            raise RuntimeError(f"unexpected RGB frame dtype: {array.dtype}")
        tensor = self.torch_module.from_numpy(array).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.predictor.device, non_blocking=True)
        return (tensor - self.image_mean) / self.image_std

    def _prune_history(self, frame_idx: int) -> None:
        if self.inference_state is None:
            return
        min_keep = max(1, frame_idx - self.memory_history_size + 1)
        for obj_idx, obj_output_dict in self.inference_state["output_dict_per_obj"].items():
            non_cond = obj_output_dict["non_cond_frame_outputs"]
            for old_idx in [idx for idx in non_cond if idx < min_keep]:
                non_cond.pop(old_idx, None)
            frames_tracked = self.inference_state["frames_tracked_per_obj"].get(obj_idx, {})
            for old_idx in [idx for idx in frames_tracked if idx < min_keep]:
                frames_tracked.pop(old_idx, None)
        if "output_dict" in self.inference_state:
            non_cond = self.inference_state["output_dict"]["non_cond_frame_outputs"]
            for old_idx in [idx for idx in non_cond if idx < min_keep]:
                non_cond.pop(old_idx, None)
        if "frames_already_tracked" in self.inference_state:
            for old_idx in [idx for idx in self.inference_state["frames_already_tracked"] if idx < min_keep]:
                self.inference_state["frames_already_tracked"].pop(old_idx, None)
        for old_idx in [idx for idx in self.original_frames if idx < min_keep]:
            self.original_frames.pop(old_idx, None)
            self.headers.pop(old_idx, None)
            if old_idx < len(self.inference_state["images"]):
                self.inference_state["images"][old_idx] = None
        self.inference_state["cached_features"] = {
            idx: value for idx, value in self.inference_state["cached_features"].items() if idx >= min_keep
        }

    def _publish_frame(
        self,
        frame_index: int,
        masks: Any,
        obj_ids: Any,
        latency_ms: float,
        add_prompt_ms: float | str,
        prompt: dict[str, Any],
    ) -> None:
        frame = self.original_frames.get(frame_index, self.latest_frame)
        header = self.headers.get(frame_index, self.latest_header)
        if frame is None or header is None:
            return
        overlay = overlay_prediction(frame, masks)
        if prompt_is_visible(self.prompt_display_start, header.stamp, self.prompt_display_seconds):
            _draw_prompt(overlay, prompt)
        mask = masks_to_mono8(masks, frame.shape[:2])
        self.result_times.append(self.get_clock().now().nanoseconds / 1_000_000_000.0)
        memory = cuda_memory_mb(self.torch_module)
        result = {
            "frame_index": frame_index,
            "source_frame_index": frame_index,
            "stamp": {"sec": header.stamp.sec, "nanosec": header.stamp.nanosec},
            "frame_id": header.frame_id,
            "backend": _backend_label(str(self.get_parameter("external_repo").value)),
            "stream_mode": "native_online",
            "tracking_state": "tracking",
            "prompt_mode": prompt["prompt_mode"],
            "point_x": prompt.get("point", ("", ""))[0] if prompt["prompt_mode"] == "point" else "",
            "point_y": prompt.get("point", ("", ""))[1] if prompt["prompt_mode"] == "point" else "",
            "box_x1": prompt.get("box", ("", "", "", ""))[0] if prompt["prompt_mode"] == "box" else "",
            "box_y1": prompt.get("box", ("", "", "", ""))[1] if prompt["prompt_mode"] == "box" else "",
            "box_x2": prompt.get("box", ("", "", "", ""))[2] if prompt["prompt_mode"] == "box" else "",
            "box_y2": prompt.get("box", ("", "", "", ""))[3] if prompt["prompt_mode"] == "box" else "",
            "latency_ms": latency_ms,
            "init_state_ms": "" if frame_index else 0.0,
            "add_prompt_ms": add_prompt_ms if frame_index == 0 else "",
            "callback_total_ms": latency_ms,
            "end_to_end_ms": self._end_to_end_ms(header),
            "tracking_fps": self._tracking_fps(),
            "mask_count": _safe_len(masks),
            "object_ids": ",".join(str(obj_id) for obj_id in obj_ids),
            "model_kind": str(self.get_parameter("model_kind").value),
            "memory_history_size": self.memory_history_size,
            "queue_depth": self.input_queue_size,
            **self.load_summary,
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
            self._clear_state()
            self.get_logger().info("reset native SAM2 online tracking state")

    def _clear_state(self) -> None:
        self.state = "waiting_for_prompt"
        self.prompt = None
        self.prompt_display_start = None
        self.drag_start = None
        self.frame_index = -1
        self.inference_state = None
        self.original_frames = {}
        self.headers = {}
        self.latest_display = None

    def _tensor(self, values: list[float]) -> Any:
        return self.torch_module.tensor(values, dtype=self.torch_module.float32, device=self.predictor.device)

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
        return super().destroy_node()


def _display_with_metrics(overlay_rgb: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    panel = np.full((overlay_bgr.shape[0], 360, 3), 24, dtype=np.uint8)
    lines = [
        "SAM2 online memory",
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


def main() -> None:
    rclpy.init()
    node = Sam2OnlineTrackingNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
