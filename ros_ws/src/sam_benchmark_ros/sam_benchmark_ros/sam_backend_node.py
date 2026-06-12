from __future__ import annotations

import json
from time import perf_counter

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from sam_backend import BackendConfig, Prompt, create_backend
from sam_backend.overlay import overlay_prediction, to_numpy
from sam_backend.profiling import component_timer, cuda_memory_mb, parameter_counts
from sam_backend.streaming import masks_to_mono8


class SamBackendNode(Node):
    def __init__(self) -> None:
        super().__init__("sam_backend_node")
        self.declare_parameter("backend", "null")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("backbone_type", "efficientvit")
        self.declare_parameter("model_name", "b0")
        self.declare_parameter("text_encoder_type", "")
        self.declare_parameter("text_encoder_context_length", 77)
        self.declare_parameter("text_encoder_pos_embed_table_size", 0)
        self.declare_parameter("interpolate_pos_embed", False)
        self.declare_parameter("autocast_dtype", "bfloat16")
        self.declare_parameter("model_config", "")
        self.declare_parameter("external_repo", "")
        self.declare_parameter("mobile_sam_model_type", "vit_t")
        self.declare_parameter("prompt_mode", "auto")
        self.declare_parameter("prompt", "person")
        self.declare_parameter("point_x", 0.5)
        self.declare_parameter("point_y", 0.5)
        self.declare_parameter("point_normalized", True)
        self.declare_parameter("point_label", 1)
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("result_topic", "/sam/result_json")
        self.declare_parameter("overlay_topic", "")
        self.declare_parameter("mask_topic", "")
        self.declare_parameter("segmented_image_topic", "")

        backend_name = self.get_parameter("backend").value
        checkpoint_path = self.get_parameter("checkpoint_path").value or None
        device = self.get_parameter("device").value or None
        backbone_type = self.get_parameter("backbone_type").value
        model_name = self.get_parameter("model_name").value
        text_encoder_type = self.get_parameter("text_encoder_type").value or None
        text_encoder_context_length = int(self.get_parameter("text_encoder_context_length").value)
        text_encoder_pos_embed_table_size = int(self.get_parameter("text_encoder_pos_embed_table_size").value) or None
        interpolate_pos_embed = bool(self.get_parameter("interpolate_pos_embed").value)
        autocast_dtype = str(self.get_parameter("autocast_dtype").value)
        model_config = self.get_parameter("model_config").value or None
        external_repo = self.get_parameter("external_repo").value or None
        mobile_sam_model_type = self.get_parameter("mobile_sam_model_type").value
        self.backend_name = str(backend_name)
        self.prompt_mode = self._resolve_prompt_mode(str(self.get_parameter("prompt_mode").value))
        self.prompt_text = str(self.get_parameter("prompt").value)
        self.point_x = float(self.get_parameter("point_x").value)
        self.point_y = float(self.get_parameter("point_y").value)
        self.point_normalized = bool(self.get_parameter("point_normalized").value)
        self.point_label = int(self.get_parameter("point_label").value)
        self.bridge = CvBridge()
        self.backend = create_backend(
            BackendConfig(
                backend=backend_name,
                checkpoint_path=checkpoint_path,
                device=device,
                backbone_type=backbone_type,
                model_name=model_name,
                text_encoder_type=text_encoder_type,
                text_encoder_context_length=text_encoder_context_length,
                text_encoder_pos_embed_table_size=text_encoder_pos_embed_table_size,
                interpolate_pos_embed=interpolate_pos_embed,
                autocast_dtype=autocast_dtype,
                model_config=model_config,
                external_repo=external_repo,
                mobile_sam_model_type=mobile_sam_model_type,
            )
        )
        self.torch_module = getattr(self.backend, "torch", None)
        self.params = parameter_counts(getattr(self.backend, "model", None))
        self.frame_index = 0

        image_topic = self.get_parameter("image_topic").value
        result_topic = self.get_parameter("result_topic").value
        overlay_topic = self.get_parameter("overlay_topic").value
        mask_topic = self.get_parameter("mask_topic").value
        segmented_image_topic = self.get_parameter("segmented_image_topic").value
        self.publisher = self.create_publisher(String, result_topic, 10)
        self.overlay_publisher = self.create_publisher(Image, overlay_topic, 10) if overlay_topic else None
        self.mask_publisher = self.create_publisher(Image, mask_topic, 10) if mask_topic else None
        self.segmented_image_publisher = (
            self.create_publisher(Image, segmented_image_topic, 10) if segmented_image_topic else None
        )
        self.subscription = self.create_subscription(Image, image_topic, self.on_image, 10)
        self.get_logger().info(f"listening on {image_topic}, publishing {result_topic}")
        if overlay_topic:
            self.get_logger().info(f"publishing overlays on {overlay_topic}")
        if mask_topic:
            self.get_logger().info(f"publishing mono8 masks on {mask_topic}")
        if segmented_image_topic:
            self.get_logger().info(f"publishing segmented images on {segmented_image_topic}")

    def on_image(self, msg: Image) -> None:
        callback_start = perf_counter()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        prompt = self._make_prompt(frame)
        profile = {}
        if self.torch_module is not None:
            with component_timer(getattr(self.backend, "model", None), self.torch_module) as profile:
                prediction = self.backend.predict(frame, prompt)
        else:
            prediction = self.backend.predict(frame, prompt)
        callback_total_ms = (perf_counter() - callback_start) * 1000.0
        component_total_ms = sum(profile.values())
        memory = cuda_memory_mb(self.torch_module) if self.torch_module is not None else cuda_memory_mb(None)
        scores = to_numpy(prediction.scores)
        now_msg = self.get_clock().now().to_msg()
        end_to_end_ms = _stamp_delta_ms(msg.header.stamp.sec, msg.header.stamp.nanosec, now_msg.sec, now_msg.nanosec)
        result = {
            "frame_index": self.frame_index,
            "stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec},
            "frame_id": msg.header.frame_id,
            "latency_ms": prediction.latency_ms,
            "callback_total_ms": callback_total_ms,
            "end_to_end_ms": end_to_end_ms,
            "image_encoder_ms": profile.get("image_encoder_ms", 0.0),
            "text_encoder_ms": profile.get("text_encoder_ms", 0.0),
            "prompt_encoder_ms": profile.get("prompt_encoder_ms", 0.0),
            "mask_decoder_ms": profile.get("mask_decoder_ms", 0.0),
            "transformer_ms": profile.get("transformer_ms", 0.0),
            "geometry_encoder_ms": profile.get("geometry_encoder_ms", 0.0),
            "segmentation_head_ms": profile.get("segmentation_head_ms", 0.0),
            "grounding_ms": profile.get("grounding_ms", 0.0),
            "detector_ms": profile.get("detector_ms", 0.0),
            "memory_attention_ms": profile.get("memory_attention_ms", 0.0),
            "memory_encoder_ms": profile.get("memory_encoder_ms", 0.0),
            "other_ms": max(0.0, callback_total_ms - component_total_ms),
            "prompt_mode": self.prompt_mode,
            "prompt_text": self.prompt_text if self.prompt_mode == "text" else "",
            "point_x": prompt.points[0][0] if prompt.points else "",
            "point_y": prompt.points[0][1] if prompt.points else "",
            "mask_count": _safe_len(prediction.masks),
            "box_count": _safe_len(prediction.boxes),
            "score_max": float(scores.max()) if scores.size else None,
            **memory,
            **self.params,
        }
        self.publisher.publish(String(data=json.dumps(result)))
        if self.mask_publisher is not None:
            mask = masks_to_mono8(prediction.masks, frame.shape[:2])
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            mask_msg.header = msg.header
            self.mask_publisher.publish(mask_msg)
        if self.overlay_publisher is not None or self.segmented_image_publisher is not None:
            overlay = overlay_prediction(frame, prediction.masks, prediction.boxes, prediction.scores)
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="rgb8")
            overlay_msg.header = msg.header
            if self.overlay_publisher is not None:
                self.overlay_publisher.publish(overlay_msg)
            if self.segmented_image_publisher is not None:
                self.segmented_image_publisher.publish(overlay_msg)
        self.frame_index += 1

    def _resolve_prompt_mode(self, value: str) -> str:
        if value == "auto":
            if self.backend_name in {"sam2", "efficient-sam2", "efficienttam", "mobilesam"}:
                return "point"
            return "text"
        if value not in {"text", "point"}:
            raise ValueError("prompt_mode must be one of: auto, text, point")
        return value

    def _make_prompt(self, frame: object) -> Prompt:
        if self.prompt_mode == "text":
            return Prompt(text=self.prompt_text)
        height, width = frame.shape[:2]  # type: ignore[attr-defined]
        if self.point_normalized:
            x = self.point_x * float(width)
            y = self.point_y * float(height)
        else:
            x = self.point_x
            y = self.point_y
        return Prompt(points=[(x, y)], labels=[self.point_label])


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
    node = SamBackendNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
