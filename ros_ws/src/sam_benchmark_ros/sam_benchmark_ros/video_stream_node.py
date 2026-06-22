from __future__ import annotations

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class VideoStreamNode(Node):
    def __init__(self) -> None:
        super().__init__("video_stream_node")
        self.declare_parameter("video_path", "")
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("fps", 15.0)
        self.declare_parameter("frame_id", "camera")
        self.declare_parameter("resize_width", 0)
        self.declare_parameter("resize_height", 0)

        video_path = self.get_parameter("video_path").value
        topic = self.get_parameter("image_topic").value
        fps = float(self.get_parameter("fps").value)
        self.frame_id = self.get_parameter("frame_id").value
        self.resize_width = int(self.get_parameter("resize_width").value)
        self.resize_height = int(self.get_parameter("resize_height").value)
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, topic, 10)
        self.capture = cv2.VideoCapture(video_path if video_path else 0)
        if not self.capture.isOpened():
            raise RuntimeError(f"failed to open video source: {video_path or 0}")
        self.timer = self.create_timer(1.0 / fps, self.publish_frame)
        if self.resize_width > 0 or self.resize_height > 0:
            self.get_logger().info(
                f"publishing {topic} at {fps:g} FPS; resizing frames to {self.resize_width}x{self.resize_height}"
            )
        else:
            self.get_logger().info(f"publishing {topic} at {fps:g} FPS")

    def publish_frame(self) -> None:
        ok, frame = self.capture.read()
        if not ok:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
            if not ok:
                return
        frame = self._resize_frame(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        msg = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.publisher.publish(msg)

    def destroy_node(self) -> bool:
        self.capture.release()
        return super().destroy_node()

    def _resize_frame(self, frame):
        if self.resize_width <= 0 and self.resize_height <= 0:
            return frame
        height, width = frame.shape[:2]
        if self.resize_width > 0 and self.resize_height > 0:
            size = (self.resize_width, self.resize_height)
        elif self.resize_width > 0:
            scale = self.resize_width / float(width)
            size = (self.resize_width, max(1, int(round(height * scale))))
        else:
            scale = self.resize_height / float(height)
            size = (max(1, int(round(width * scale))), self.resize_height)
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def main() -> None:
    rclpy.init()
    node = VideoStreamNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
