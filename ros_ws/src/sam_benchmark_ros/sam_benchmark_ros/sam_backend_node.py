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
        self.declare_parameter("prompt", "person")
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("result_topic", "/sam/result_json")
        self.declare_parameter("overlay_topic", "")

        backend_name = self.get_parameter("backend").value
        checkpoint_path = self.get_parameter("checkpoint_path").value or None
        device = self.get_parameter("device").value or None
        backbone_type = self.get_parameter("backbone_type").value
        model_name = self.get_parameter("model_name").value
        text_encoder_type = self.get_parameter("text_encoder_type").value or None
        text_encoder_context_length = int(self.get_parameter("text_encoder_context_length").value)
        text_encoder_pos_embed_table_size = int(self.get_parameter("text_encoder_pos_embed_table_size").value) or None
        interpolate_pos_embed = bool(self.get_parameter("interpolate_pos_embed").value)
        self.prompt = Prompt(text=self.get_parameter("prompt").value)
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
            )
        )
        self.torch_module = getattr(self.backend, "torch", None)
        self.params = parameter_counts(getattr(self.backend, "model", None))
        self.frame_index = 0

        image_topic = self.get_parameter("image_topic").value
        result_topic = self.get_parameter("result_topic").value
        overlay_topic = self.get_parameter("overlay_topic").value
        self.publisher = self.create_publisher(String, result_topic, 10)
        self.overlay_publisher = self.create_publisher(Image, overlay_topic, 10) if overlay_topic else None
        self.subscription = self.create_subscription(Image, image_topic, self.on_image, 10)
        self.get_logger().info(f"listening on {image_topic}, publishing {result_topic}")
        if overlay_topic:
            self.get_logger().info(f"publishing overlays on {overlay_topic}")

    def on_image(self, msg: Image) -> None:
        callback_start = perf_counter()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        profile = {}
        if self.torch_module is not None:
            with component_timer(getattr(self.backend, "model", None), self.torch_module) as profile:
                prediction = self.backend.predict(frame, self.prompt)
        else:
            prediction = self.backend.predict(frame, self.prompt)
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
            "grounding_ms": profile.get("grounding_ms", 0.0),
            "other_ms": max(0.0, callback_total_ms - component_total_ms),
            "mask_count": _safe_len(prediction.masks),
            "box_count": _safe_len(prediction.boxes),
            "score_max": float(scores.max()) if scores.size else None,
            **memory,
            **self.params,
        }
        self.publisher.publish(String(data=json.dumps(result)))
        if self.overlay_publisher is not None:
            overlay = overlay_prediction(frame, prediction.masks, prediction.boxes, prediction.scores)
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="rgb8")
            overlay_msg.header = msg.header
            self.overlay_publisher.publish(overlay_msg)
        self.frame_index += 1


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
