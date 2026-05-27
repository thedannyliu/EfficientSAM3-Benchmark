from __future__ import annotations

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraStreamNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_stream_node")
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("gstreamer_pipeline", "")
        self.declare_parameter("width", 0)
        self.declare_parameter("height", 0)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("frame_id", "camera")

        topic = self.get_parameter("image_topic").value
        camera_index = int(self.get_parameter("camera_index").value)
        pipeline = str(self.get_parameter("gstreamer_pipeline").value)
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        fps = float(self.get_parameter("fps").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, topic, 10)
        if pipeline:
            self.capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            source_label = "GStreamer pipeline"
        else:
            self.capture = cv2.VideoCapture(camera_index)
            source_label = f"camera index {camera_index}"
        if width > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            self.capture.set(cv2.CAP_PROP_FPS, fps)
        if not self.capture.isOpened():
            raise RuntimeError(f"failed to open {source_label}")

        self.timer = self.create_timer(1.0 / fps, self.publish_frame)
        self.get_logger().info(f"publishing {source_label} to {topic} at {fps:g} FPS")

    def publish_frame(self) -> None:
        ok, frame = self.capture.read()
        if not ok:
            self.get_logger().warning("camera read failed")
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        msg = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.publisher.publish(msg)

    def destroy_node(self) -> bool:
        self.capture.release()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = CameraStreamNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
