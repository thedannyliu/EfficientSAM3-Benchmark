from __future__ import annotations

import json

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from sam_backend import BackendConfig, Prompt, create_backend


class SamBackendNode(Node):
    def __init__(self) -> None:
        super().__init__("sam_backend_node")
        self.declare_parameter("backend", "null")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("prompt", "person")
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("result_topic", "/sam/result_json")

        backend_name = self.get_parameter("backend").value
        checkpoint_path = self.get_parameter("checkpoint_path").value or None
        device = self.get_parameter("device").value or None
        self.prompt = Prompt(text=self.get_parameter("prompt").value)
        self.bridge = CvBridge()
        self.backend = create_backend(
            BackendConfig(backend=backend_name, checkpoint_path=checkpoint_path, device=device)
        )

        image_topic = self.get_parameter("image_topic").value
        result_topic = self.get_parameter("result_topic").value
        self.publisher = self.create_publisher(String, result_topic, 10)
        self.subscription = self.create_subscription(Image, image_topic, self.on_image, 10)
        self.get_logger().info(f"listening on {image_topic}, publishing {result_topic}")

    def on_image(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        prediction = self.backend.predict(frame, self.prompt)
        result = {
            "stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec},
            "frame_id": msg.header.frame_id,
            "latency_ms": prediction.latency_ms,
            "mask_count": _safe_len(prediction.masks),
            "box_count": _safe_len(prediction.boxes),
        }
        self.publisher.publish(String(data=json.dumps(result)))


def _safe_len(value: object) -> int:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def main() -> None:
    rclpy.init()
    node = SamBackendNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
